"""
Evaluation script for Multi-Frame CRNN License Plate Recognition.

This script loads the trained model and evaluates it on the validation/test dataset.

Usage:
    python test.py                    # Evaluate on validation set
    python test.py --test             # Evaluate on test set (if available)
    python test.py --model_path path  # Use custom model path
    python test.py --config_path path # Use custom config path
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Support both running as module and direct script execution
try:
    from .config import Config
    from .dataset import AdvancedMultiFrameDataset
    from .models import MultiFrameCRNN
    from .utils import seed_everything, decode_predictions
except ImportError:
    from config import Config
    from dataset import AdvancedMultiFrameDataset
    from models import MultiFrameCRNN
    from utils import seed_everything, decode_predictions


class ModelEvaluator:
    """Class to handle model evaluation and metrics calculation."""
    
    def __init__(self, model, device, idx2char):
        self.model = model
        self.device = device
        self.idx2char = idx2char
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
    def evaluate(self, data_loader, dataset_name="Validation"):
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: PyTorch DataLoader
            dataset_name: Name of dataset for display
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        predictions_list = []
        ground_truth_list = []
        error_cases = []
        
        print(f"\n{'='*80}")
        print(f"EVALUATING ON {dataset_name.upper()} SET")
        print(f"{'='*80}")
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"{dataset_name} Evaluation", unit="batch")
            
            for batch_idx, batch_data in enumerate(pbar):
                images = batch_data[0].to(self.device)
                targets = batch_data[1].to(self.device)
                target_lengths = batch_data[2]
                labels_text = batch_data[3]
                
                # Forward pass
                preds = self.model(images)
                
                # Calculate loss
                input_lengths = torch.full(
                    (images.size(0),), 
                    preds.size(1), 
                    dtype=torch.long
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                total_loss += loss.item()
                
                # Decode predictions
                predicted_text = decode_predictions(
                    torch.argmax(preds, dim=2),
                    self.idx2char
                )
                
                # Calculate accuracy
                for i in range(len(labels_text)):
                    ground_truth_list.append(labels_text[i])
                    predictions_list.append(predicted_text[i])
                    
                    if predicted_text[i] == labels_text[i]:
                        total_correct += 1
                    else:
                        error_cases.append({
                            'ground_truth': labels_text[i],
                            'prediction': predicted_text[i]
                        })
                    
                    total_samples += 1
                
                # Update progress bar
                accuracy = (total_correct / total_samples) * 100
                pbar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': f'{accuracy:.2f}%'
                })
        
        avg_loss = total_loss / len(data_loader)
        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        
        metrics = {
            'dataset_name': dataset_name,
            'total_samples': total_samples,
            'correct_predictions': total_correct,
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'error_rate': 100 - accuracy,
            'error_cases': error_cases[:100],  # Store first 100 errors
            'outputs/predictions/predictions': predictions_list,
            'ground_truth': ground_truth_list
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics in a formatted way."""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS - {metrics['dataset_name'].upper()}")
        print(f"{'='*80}")
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"  ├─ Total Samples: {metrics['total_samples']}")
        print(f"  ├─ Correct Predictions: {metrics['correct_predictions']}")
        print(f"  ├─ Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  ├─ Error Rate: {metrics['error_rate']:.2f}%")
        print(f"  └─ Average Loss: {metrics['avg_loss']:.4f}")
        
        if metrics['error_cases']:
            print(f"\n❌ SAMPLE ERROR CASES (showing first 10):")
            for i, case in enumerate(metrics['error_cases'][:10]):
                print(f"  {i+1}. Expected: '{case['ground_truth']}' | Got: '{case['prediction']}'")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, metrics, output_file="evaluation_results.json"):
        """Save evaluation results to JSON file."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_name': metrics['dataset_name'],
            'total_samples': metrics['total_samples'],
            'correct_predictions': metrics['correct_predictions'],
            'accuracy': metrics['accuracy'],
            'error_rate': metrics['error_rate'],
            'avg_loss': metrics['avg_loss'],
            'error_cases': metrics['error_cases']
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_file}")


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"📂 Loading model from: {model_path}")
    
    model = MultiFrameCRNN(num_classes=Config.NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both direct state_dict and checkpoint with state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✅ Model loaded successfully")
    return model


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate License Plate Recognition Model')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Path to dataset root (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for evaluation (default: from config)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of workers for data loading (default: from config)')
    parser.add_argument('--mode', type=str, choices=['val', 'test'], default='val',
                        help='Evaluation mode: val or test')
    parser.add_argument('--save_results', action='store_true',
                        help='Save evaluation results to JSON')
    parser.add_argument('--show_errors', type=int, default=10,
                        help='Number of error cases to display')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_everything(Config.SEED)
    
    # Set device
    device = Config.DEVICE
    print(f"🚀 EVALUATION START | Device: {device}")
    print(f"{'='*80}\n")
    
    # Load model
    try:
        model = load_model(args.model_path, device)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Prepare dataset
    dataset_root = args.dataset_root or Config.DATA_ROOT
    batch_size = args.batch_size or Config.BATCH_SIZE
    num_workers = args.num_workers or Config.NUM_WORKERS
    
    print(f"📂 Dataset Root: {dataset_root}")
    print(f"📋 Batch Size: {batch_size}")
    print(f"👷 Num Workers: {num_workers}\n")
    
    # Load validation/test dataset
    print(f"Loading {args.mode} dataset...")
    try:
        eval_dataset = AdvancedMultiFrameDataset(
            dataset_root,
            mode=args.mode,
            split_ratio=0.8
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    if len(eval_dataset) == 0:
        print(f"❌ {args.mode.upper()} dataset is empty!")
        return
    
    # Create data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AdvancedMultiFrameDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device, Config.IDX2CHAR)
    
    # Run evaluation
    metrics = evaluator.evaluate(eval_loader, dataset_name=args.mode.upper())
    
    # Print results
    evaluator.print_metrics(metrics)
    
    # Save results if requested
    if args.save_results:
        output_file = f"outputs/evaluations/evaluation_results_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        evaluator.save_results(metrics, output_file)
    
    print(f"{'='*80}")
    print(f"✅ EVALUATION COMPLETED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
