"""
Inference script for ICPR 2026 LRLPR Challenge submission.

Generate predictions for test set and create submission file.

Usage:
    python inference.py
    python inference.py --model_path custom_model.pth
    python inference.py --test_dir ../test_data
    python inference.py --output predictions.txt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np

# Import from baseline
from config import Config
from models import Phase1Recognizer, Phase2Recognizer
from models.recognizer_v3 import Phase3Recognizer
from utils import decode_predictions
from transforms import get_val_transforms
from postprocess import batch_beam_search_decode, validate_plate_format


class TestDataset(Dataset):
    """Dataset for test set (no ground truth labels)."""
    
    def __init__(self, test_dir, transform=None, num_frames=5):
        self.test_dir = test_dir
        self.transform = transform
        self.num_frames = num_frames
        
        # Scan all track directories
        self.tracks = []
        for track_name in sorted(os.listdir(test_dir)):
            track_path = os.path.join(test_dir, track_name)
            if os.path.isdir(track_path) and track_name.startswith('track_'):
                # Check if LR frames exist
                lr_frames = [f"lr-{i:03d}.jpg" for i in range(1, num_frames + 1)]
                if all(os.path.exists(os.path.join(track_path, f)) for f in lr_frames):
                    self.tracks.append({
                        'track_id': track_name,
                        'track_path': track_path,
                        'lr_frames': lr_frames
                    })
        
        print(f"✅ Found {len(self.tracks)} test tracks")
    
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        track = self.tracks[idx]
        
        # Load LR frames
        images = []
        for frame_name in track['lr_frames']:
            img_path = os.path.join(track['track_path'], frame_name)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
            
            images.append(img)
        
        # Stack frames: [num_frames, C, H, W]
        images = torch.stack(images, dim=0)
        
        return images, track['track_id']


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"\n📦 Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config
    model_config = checkpoint.get('config', {})
    model_type = model_config.get('model', 'Phase1Recognizer')
    
    print(f"   Model Type: {model_type}")
    print(f"   STN: {model_config.get('stn', True)}")
    print(f"   Backbone: {model_config.get('backbone', 'ResNet34')}")
    
    # Create model based on type
    p10 = checkpoint.get('phase10_config', {})
    if model_type == 'Phase3Recognizer' or 'phase3' in str(model_path).lower() or any(k in str(model_path).lower() for k in ['phase8', 'phase9', 'phase10', 'phase11', 'phase12', 'phase13']):
        print(f"   Loading Phase 3 model (BiLSTM)...")
        model = Phase3Recognizer(
            num_classes=len(Config.CHARS) + 1,
            use_stn=model_config.get('stn', True),
            use_resnet_backbone=True,
            hidden_size=p10.get('hidden_size', 256),
            num_lstm_layers=p10.get('num_lstm_layers', 2),
            dropout=0.0,
            fusion_reduction=p10.get('fusion_reduction', 16),
            use_sr_branch=False,
        )
    elif model_type == 'Phase2Recognizer' or 'phase2' in str(model_path).lower():
        print(f"   Loading Phase 2 model...")
        model = Phase2Recognizer(
            num_classes=len(Config.CHARS) + 1,
            use_stn=model_config.get('stn', True),
            use_resnet_backbone=model_config.get('backbone') == 'ResNet34',
            d_model=model_config.get('d_model', 512),
            nhead=model_config.get('nhead', 4),
            num_transformer_layers=model_config.get('num_transformer_layers', 2),
            dim_feedforward=model_config.get('dim_feedforward', 1024),
            fusion_reduction=model_config.get('fusion_reduction', 16),
            dropout=model_config.get('dropout', 0.1),
            use_sr_branch=False,
        )
    else:
        print(f"   Loading Phase 1 model...")
        model = Phase1Recognizer(
            num_classes=len(Config.CHARS) + 1,
            use_stn=model_config.get('stn', True),
            use_resnet_backbone=model_config.get('backbone') == 'ResNet34'
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Epoch: {checkpoint['epoch'] + 1}")
    print(f"   Accuracy: {checkpoint['accuracy']:.2f}%")
    print(f"   CER: {checkpoint['cer']:.4f}")
    
    return model


def run_inference(model, test_loader, device, idx2char, use_beam_search=True,
                  beam_width=10):
    """Run inference on test set with optional beam search decoding."""
    mode = "beam search" if use_beam_search else "greedy"
    print(f"\n🔮 Running inference ({mode}, beam_width={beam_width})...")
    
    predictions = {}
    
    model.eval()
    with torch.no_grad():
        for images, track_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)  # [B, T, C] log-probs
            
            if use_beam_search:
                # Beam search with plate format bonus
                decoded, confidences = batch_beam_search_decode(
                    outputs, idx2char, beam_width=beam_width, blank=0,
                    format_bonus_weight=1.5
                )
                for track_id, pred_text, conf in zip(track_ids, decoded, confidences):
                    predictions[track_id] = {
                        'text': pred_text,
                        'confidence': conf
                    }
            else:
                # Greedy decoding (fallback)
                probs = torch.softmax(outputs, dim=2)
                preds = torch.argmax(probs, dim=2)
                decoded = decode_predictions(preds, idx2char)
                
                for i, (track_id, pred_text) in enumerate(zip(track_ids, decoded)):
                    max_probs = torch.max(probs[i], dim=1)[0]
                    non_blank_mask = preds[i] != 0
                    if non_blank_mask.sum() > 0:
                        confidence = max_probs[non_blank_mask].mean().item()
                    else:
                        confidence = 0.0
                    predictions[track_id] = {
                        'text': pred_text,
                        'confidence': confidence
                    }
    
    return predictions


def save_predictions(predictions, output_path):
    """Save predictions to submission file in ICPR format."""
    print(f"\n💾 Saving predictions to: {output_path}")
    
    # Sort by track_id for consistency
    sorted_predictions = sorted(predictions.items(), key=lambda x: int(x[0].split('_')[1]))
    
    # ICPR 2026 format: track_id,plate_text;confidence
    with open(output_path, 'w') as f:
        for track_id, pred_data in sorted_predictions:
            text = pred_data['text']
            confidence = pred_data['confidence']
            f.write(f"{track_id},{text};{confidence:.4f}\n")
    
    print(f"   ✅ Saved {len(predictions)} predictions")
    
    # Show sample predictions
    print(f"\n📝 Sample predictions (ICPR 2026 format):")
    for track_id, pred_data in sorted_predictions[:10]:
        text = pred_data['text']
        confidence = pred_data['confidence']
        print(f"   {track_id},{text};{confidence:.4f}")


def main():
    parser = argparse.ArgumentParser(description='ICPR 2026 LRLPR Inference')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model_phase2.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_dir', type=str, default='Pa7a3Hin-test-public',
                        help='Path to test directory')
    parser.add_argument('--output', type=str, default='outputs/predictions/predictions.txt',
                        help='Output file for predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--beam_width', type=int, default=10,
                        help='Beam width for beam search (0=greedy)')
    
    args = parser.parse_args()
    
    # Setup
    print("="*80)
    print("🎯 ICPR 2026 LRLPR Challenge - Inference Script")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Prepare transforms
    transform = get_val_transforms()
    
    # Create test dataset
    print(f"\n📁 Loading test data from: {args.test_dir}")
    test_dataset = TestDataset(
        test_dir=args.test_dir,
        transform=transform,
        num_frames=5  # Standard number of frames in ICPR 2026 dataset
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run inference
    use_beam = args.beam_width > 0
    predictions = run_inference(model, test_loader, device, Config.IDX2CHAR,
                               use_beam_search=use_beam,
                               beam_width=args.beam_width if use_beam else 1)
    
    # Save results
    save_predictions(predictions, args.output)
    
    print(f"\n{'='*80}")
    print("✅ INFERENCE COMPLETED!")
    print(f"{'='*80}")
    print(f"📄 Submission file: {args.output}")
    print(f"📊 Total predictions: {len(predictions)}")


if __name__ == '__main__':
    main()
