"""
Training script for Multi-Frame CRNN License Plate Recognition.

Usage:
    python train.py

The data directory should be configured in config.py (DATA_ROOT).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Support both running as module and direct script execution
try:
    from .config import Config
    from .dataset import AdvancedMultiFrameDataset
    from .models import Phase1Recognizer
    from .utils import (
        seed_everything, decode_predictions, calculate_cer, 
        calculate_accuracy, get_prediction_confidence, calculate_confidence_gap
    )
except ImportError:
    from config import Config
    from dataset import AdvancedMultiFrameDataset
    from models import Phase1Recognizer
    from utils import (
        seed_everything, decode_predictions, calculate_cer,
        calculate_accuracy, get_prediction_confidence, calculate_confidence_gap
    )


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (accuracy), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        # Check if improved
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_pipeline():
    """Main training pipeline with Phase 1 upgrades."""
    seed_everything(Config.SEED)
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(Config.LOG_DIR, f'phase1_{timestamp}')
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs: {log_dir}")
    
    print(f"🚀 PHASE 1 TRAINING START | Device: {Config.DEVICE}")
    
    # Check data directory
    if not os.path.exists(Config.DATA_ROOT):
        print(f"❌ LỖI: Sai đường dẫn DATA_ROOT: {Config.DATA_ROOT}")
        writer.close()
        return

    # Create datasets
    train_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='train', split_ratio=0.8)
    val_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='val', split_ratio=0.8)
    
    if len(train_ds) == 0: 
        print("❌ Dataset Train rỗng!")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=AdvancedMultiFrameDataset.collate_fn, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )
    
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=AdvancedMultiFrameDataset.collate_fn, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=True
        )
    else:
        print("⚠️ CẢNH BÁO: Validation Set rỗng. Sẽ bỏ qua bước validate.")
        val_loader = None

    # Initialize Phase 1 model with upgrades
    model = Phase1Recognizer(
        num_classes=Config.NUM_CLASSES,
        use_stn=Config.USE_STN,
        use_resnet_backbone=Config.USE_RESNET_BACKBONE
    ).to(Config.DEVICE)
    
    print(f"\n📋 Model Config: {model.get_model_info()}")
    print(f"   - STN: {Config.USE_STN}")
    print(f"   - Backbone: {'ResNet-34' if Config.USE_RESNET_BACKBONE else 'VanillaCNN'}")
    print(f"   - Image Size: {Config.IMG_HEIGHT}×{Config.IMG_WIDTH}\n")
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, mode='max')
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=Config.LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=Config.EPOCHS
    )
    scaler = GradScaler()

    best_acc = 0.0
    best_cer = float('inf')
    best_conf_gap = -float('inf')
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{Config.EPOCHS}")
        for images, targets, target_lengths, _ in pbar:
            images = images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                preds = model(images)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),), 
                    fill_value=preds.size(1), 
                    dtype=torch.long
                )
                loss = criterion(preds_permuted, targets, input_lengths, target_lengths)

            scaler_scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Only step scheduler if optimizer actually stepped
            if scaler.get_scale() >= scaler_scale_before:
                scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Log training metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Validation with enhanced metrics
        val_acc = 0
        val_cer = 0
        val_conf_gap = 0
        avg_val_loss = 0
        
        if val_loader:
            model.eval()
            val_loss = 0
            all_predictions = []
            all_targets = []
            all_confidences = []
            
            with torch.no_grad():
                for images, targets, target_lengths, labels_text in val_loader:
                    images = images.to(Config.DEVICE)
                    targets = targets.to(Config.DEVICE)
                    preds = model(images)
                    
                    loss = criterion(
                        preds.permute(1, 0, 2), 
                        targets, 
                        torch.full((images.size(0),), preds.size(1), dtype=torch.long), 
                        target_lengths
                    )
                    val_loss += loss.item()
                    
                    # Decode predictions
                    decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)
                    all_predictions.extend(decoded)
                    all_targets.extend(labels_text)
                    
                    # Get confidence scores
                    confidences = get_prediction_confidence(preds)
                    all_confidences.extend(confidences.tolist())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = calculate_accuracy(all_predictions, all_targets) * 100
            val_cer = calculate_cer(all_predictions, all_targets)
            
            # Calculate confidence gap
            is_correct = [pred == target for pred, target in zip(all_predictions, all_targets)]
            val_conf_gap = calculate_confidence_gap(all_confidences, is_correct)
            
            # Log validation metrics
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('CER/val', val_cer, epoch)
            writer.add_scalar('ConfidenceGap/val', val_conf_gap, epoch)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | CER: {val_cer:.4f} | ConfGap: {val_conf_gap:.4f}")
        
        # Save best model (by accuracy)
        if val_acc > best_acc:
            best_acc = val_acc
            best_cer = val_cer
            best_conf_gap = val_conf_gap
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'cer': val_cer,
                'confidence_gap': val_conf_gap,
                'config': model.get_model_info()
            }, "checkpoints/best_model_phase1.pth")
            print(f" -> ⭐ Saved Best Model! (Acc: {val_acc:.2f}%, CER: {val_cer:.4f}, Gap: {val_conf_gap:.4f})")
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"   Best Acc: {best_acc:.2f}% | Best CER: {best_cer:.4f} | Best Gap: {best_conf_gap:.4f}")
            break


    writer.close()
    print(f"\n✅ Training completed!")
    print(f"   Best Accuracy: {best_acc:.2f}%")
    print(f"   Best CER: {best_cer:.4f}")
    print(f"   Best Confidence Gap: {best_conf_gap:.4f}")
    print(f"   TensorBoard logs: {log_dir}")


if __name__ == "__main__":
    train_pipeline()
