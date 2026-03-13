"""
🚀 OPTIMIZED TRAINING FOR A30 GPU (24GB VRAM)
================================================
Key Optimizations:
- Batch size: 96 → 256 (2.67x larger)
- Num workers: 4 → 12 (parallel data loading)
- Mixed Precision: AMP (automatic)
- cudnn.benchmark: True (faster convolution)
- Pin memory: True (faster GPU transfer)
- Persistent workers: True (reduce worker startup)
- torch.compile: JIT optimization (PyTorch 2.0+)
- Gradient accumulation: Optional for even larger effective batch

Expected speedup: 2-3x faster training
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add baseline to path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dataset import LPRDataset
from models import Phase1Recognizer
from utils import decode_predictions, calculate_accuracy, calculate_cer, calculate_confidence_gap

# Get config values
TRAIN_ROOT = Config.DATA_ROOT
VAL_SPLIT_FILE = Config.VAL_SPLIT_FILE
CHARACTERS = Config.CHARS
NUM_EPOCHS = Config.EPOCHS
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = Config.EARLY_STOPPING_PATIENCE
LOG_DIR = Config.LOG_DIR
MAX_FRAMES = 5
USE_STN = Config.USE_STN
USE_RESNET_BACKBONE = Config.USE_RESNET_BACKBONE

# ============================================================================
# A30 GPU OPTIMIZED CONFIG (Parallel Training Mode)
# ============================================================================
BATCH_SIZE_A30 = 64           # 64 for parallel training (~5GB VRAM)
NUM_WORKERS_A30 = 6           # 6 workers (reduced for parallel training)
PIN_MEMORY = True             # Faster CPU→GPU transfer
PERSISTENT_WORKERS = True     # Reduce worker startup time
PREFETCH_FACTOR = 4           # Prefetch 4 batches per worker
GRADIENT_ACCUMULATION = 4     # 4x accumulation → effective batch = 256
USE_TORCH_COMPILE = True      # PyTorch 2.0+ JIT compilation

# Learning rate scaling (linear with effective batch size)
BASE_LR = 1e-3                # Original LR for batch=96
EFFECTIVE_BATCH = BATCH_SIZE_A30 * GRADIENT_ACCUMULATION  # 64 × 4 = 256
LR_A30 = BASE_LR * (EFFECTIVE_BATCH / 96)  # Scale LR proportionally

# Image size (can increase for better accuracy)
IMG_HEIGHT_A30 = 48           # Keep 48 or increase to 64
IMG_WIDTH_A30 = 192           # 160 → 192 (more sequence length)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, mode='max', delta=0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.mode == 'max':
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, gradient_accum_steps=1):
    """Training loop with gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Ep {epoch}/{NUM_EPOCHS}")
    for batch_idx, (images, labels, input_lengths, target_lengths) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images)  # [B, T, C]
            log_probs = outputs.log_softmax(2)
            log_probs = log_probs.permute(1, 0, 2)  # [T, B, C]
            
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            loss = loss / gradient_accum_steps  # Scale loss for accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every gradient_accum_steps
        if (batch_idx + 1) % gradient_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accum_steps
        pbar.set_postfix({
            'loss': f"{loss.item() * gradient_accum_steps:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validation with all metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_confidences = []
    
    for images, labels, input_lengths, target_lengths in tqdm(dataloader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)
            log_probs = outputs.log_softmax(2)
            log_probs_t = log_probs.permute(1, 0, 2)
            
            loss = criterion(log_probs_t, labels, input_lengths, target_lengths)
            total_loss += loss.item()
        
        # Decode predictions
        preds = decode_predictions(log_probs, CHARACTERS)
        targets = []
        start_idx = 0
        for length in target_lengths:
            target_seq = labels[start_idx:start_idx + length].cpu().numpy()
            target_text = ''.join([CHARACTERS[idx] for idx in target_seq])
            targets.append(target_text)
            start_idx += length
        
        all_preds.extend(preds)
        all_targets.extend(targets)
        
        # Calculate confidences
        probs = torch.exp(log_probs)
        max_probs, _ = probs.max(dim=2)
        confidences = max_probs.mean(dim=1).cpu().numpy()
        all_confidences.extend(confidences)
    
    # Calculate metrics
    accuracy = calculate_accuracy(all_preds, all_targets)
    cer = calculate_cer(all_preds, all_targets)
    conf_gap = calculate_confidence_gap(all_preds, all_targets, all_confidences)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, cer, conf_gap


def main():
    # Fix random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True  # 🚀 Auto-tune convolution algorithms
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("🚀 PHASE 1 TRAINING - A30 GPU OPTIMIZED (PARALLEL MODE)")
    print("="*80)
    print(f"📦 Batch Size: {BATCH_SIZE_A30} (Physical)")
    print(f"🔄 Gradient Accumulation: {GRADIENT_ACCUMULATION}x")
    print(f"⚡ Effective Batch: {EFFECTIVE_BATCH} (= {BATCH_SIZE_A30} × {GRADIENT_ACCUMULATION})")
    print(f"👷 Workers: {NUM_WORKERS_A30}")
    print(f"📐 Image Size: {IMG_HEIGHT_A30}×{IMG_WIDTH_A30}")
    print(f"📚 Learning Rate: {LR_A30:.2e} (Scaled from {BASE_LR:.2e})")
    print(f"🎯 Device: {device}")
    if torch.cuda.is_available():
        print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        used_vram = torch.cuda.memory_allocated() / 1e9
        free_vram = total_vram - used_vram
        print(f"🗄️  VRAM: {used_vram:.1f} GB used / {total_vram:.1f} GB total ({free_vram:.1f} GB free)")
    print("="*80 + "\n")
    
    # Create datasets with optimized config
    print("📂 Loading datasets...")
    train_dataset = LPRDataset(
        root_dir=TRAIN_ROOT,
        mode='train',
        split_ratio=0.8
    )
    
    val_dataset = LPRDataset(
        root_dir=TRAIN_ROOT,
        mode='val',
        split_ratio=0.8
    )
    
    # Optimized DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_A30,
        shuffle=True,
        num_workers=NUM_WORKERS_A30,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_A30,
        shuffle=False,
        num_workers=NUM_WORKERS_A30,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=val_dataset.collate_fn
    )
    
    print(f"✅ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"✅ Val: {len(val_dataset)} samples, {len(val_loader)} batches\n")
    
    # Initialize model
    print("🏗️  Building model...")
    model = Phase1Recognizer(
        num_classes=len(CHARACTERS),
        use_stn=USE_STN,
        use_resnet_backbone=USE_RESNET_BACKBONE
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total params: {total_params:,} ({total_params * 4 / 1e6:.1f} MB)")
    print(f"📊 Trainable params: {trainable_params:,}\n")
    
    # Compile model for faster execution (PyTorch 2.0+)
    if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        print("⚡ Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode='max-autotune')  # Max optimization
            print("✅ Model compiled successfully!\n")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
            print("   Continuing without compilation...\n")
    
    # Setup training
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR_A30, weight_decay=WEIGHT_DECAY)
    
    # OneCycle LR scheduler (optimal for fast convergence)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_A30,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader) // GRADIENT_ACCUMULATION,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    scaler = GradScaler()  # Mixed precision
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='max')
    
    # TensorBoard
    log_dir = f"{LOG_DIR}/phase1_fast_a30_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs: {log_dir}\n")
    
    # Training loop
    print("🚀 Starting training...\n")
    best_accuracy = 0
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, GRADIENT_ACCUMULATION
        )
        
        # Validate
        val_loss, accuracy, cer, conf_gap = validate(model, val_loader, criterion, device)
        
        # Update LR
        if GRADIENT_ACCUMULATION == 1:
            scheduler.step()
        else:
            for _ in range(len(train_loader) // GRADIENT_ACCUMULATION):
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('Accuracy', accuracy, epoch)
        writer.add_scalar('CER', cer, epoch)
        writer.add_scalar('ConfidenceGap', conf_gap, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Print results
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Acc: {accuracy:.2f}% | CER: {cer:.4f} | ConfGap: {conf_gap:.4f}")
        print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
                'cer': cer,
                'confidence_gap': conf_gap,
                'config': {
                    'batch_size': BATCH_SIZE_A30,
                    'img_height': IMG_HEIGHT_A30,
                    'img_width': IMG_WIDTH_A30,
                    'num_workers': NUM_WORKERS_A30,
                    'lr': LR_A30
                }
            }
            save_path = os.path.join(os.path.dirname(__file__), 'checkpoints/best_model_phase1_a30.pth')
            torch.save(checkpoint, save_path)
            print(f" -> ⭐ Saved Best Model! (Acc: {accuracy:.2f}%, CER: {cer:.4f}, Gap: {conf_gap:.4f})")
        
        # Early stopping
        early_stopping(accuracy, epoch)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            print(f"   Best epoch: {early_stopping.best_epoch} with accuracy: {early_stopping.best_score:.2f}%")
            break
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("✅ Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"🏆 Best accuracy: {best_accuracy:.2f}%")
    print(f"📊 Average time/epoch: {total_time/epoch:.1f}s")
    print("="*80)
    
    writer.close()


if __name__ == '__main__':
    main()
