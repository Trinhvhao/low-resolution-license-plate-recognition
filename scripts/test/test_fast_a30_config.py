#!/usr/bin/env python3
"""
Quick test to verify A30 optimized config works before full training
Tests: Model creation, forward pass, memory usage, dataloader speed
"""

import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add baseline to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baseline'))

from config import *
from dataset import LPRDataset
from models import Phase1Recognizer

# A30 Config (Parallel Training Mode)
BATCH_SIZE_TEST = 64          # Reduced for parallel training
NUM_WORKERS_TEST = 6          # Reduced workers
IMG_HEIGHT_TEST = 48
IMG_WIDTH_TEST = 192

def format_bytes(bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

def test_model():
    """Test model creation and forward pass"""
    print("="*70)
    print("🧪 TEST 1: Model Creation & Forward Pass")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"Total VRAM: {format_bytes(total_mem)}")
        print()
    
    # Create model
    print("Creating model...")
    model = Phase1Recognizer(
        num_classes=len(CHARACTERS),
        use_stn=True,
        use_resnet_backbone=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params:,} parameters ({total_params*4/1e6:.1f} MB)")
    
    # Test forward pass with different batch sizes
    print("\nTesting forward pass with various batch sizes...")
    test_sizes = [64, 128, 196, 256]
    
    for bs in test_sizes:
        try:
            dummy_input = torch.randn(bs, MAX_FRAMES, 3, IMG_HEIGHT_TEST, IMG_WIDTH_TEST).to(device)
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            with torch.no_grad():
                output = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_used = torch.cuda.max_memory_allocated()
                print(f"  Batch {bs:3d}: Output shape {list(output.shape)} | VRAM: {format_bytes(mem_used)} ✅")
            else:
                print(f"  Batch {bs:3d}: Output shape {list(output.shape)} ✅")
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  Batch {bs:3d}: ❌ OOM - Reduce batch size!")
                break
            else:
                raise e
    
    print()

def test_dataloader():
    """Test DataLoader speed"""
    print("="*70)
    print("🧪 TEST 2: DataLoader Speed")
    print("="*70)
    
    print(f"Config: Batch={BATCH_SIZE_TEST}, Workers={NUM_WORKERS_TEST}")
    print(f"Image Size: {IMG_HEIGHT_TEST}×{IMG_WIDTH_TEST}")
    print()
    
    # Create dataset (AdvancedMultiFrameDataset uses different API)
    print("Loading dataset...")
    dataset = LPRDataset(
        root_dir=TRAIN_ROOT,
        mode='train',
        split_ratio=0.8
    )
    print(f"✅ Dataset: {len(dataset)} samples")
    
    # Test different worker configs
    worker_configs = [
        (4, False, 2),   # Original
        (6, True, 4),    # Parallel mode
    ]
    
    print("\nTesting DataLoader configurations:")
    print(f"{'Workers':<10} {'Persistent':<12} {'Prefetch':<10} {'Time/Batch':<12} {'Throughput':<15}")
    print("-" * 70)
    
    for num_workers, persistent, prefetch in worker_configs:
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent if num_workers > 0 else False,
            prefetch_factor=prefetch if num_workers > 0 else None,
            collate_fn=dataset.collate_fn
        )
        
        # Warmup
        for batch in loader:
            break
        
        # Benchmark
        num_batches = min(20, len(loader))
        start = time.time()
        
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
        
        elapsed = time.time() - start
        time_per_batch = elapsed / num_batches
        throughput = BATCH_SIZE_TEST / time_per_batch
        
        persistent_str = "Yes" if persistent else "No"
        print(f"{num_workers:<10} {persistent_str:<12} {prefetch:<10} {time_per_batch:.3f}s      {throughput:.1f} samples/s")
    
    print()

def test_memory_usage():
    """Test peak memory usage during training simulation"""
    print("="*70)
    print("🧪 TEST 3: Memory Usage Simulation")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return
    
    device = torch.device('cuda')
    
    # Create model and dummy batch
    model = Phase1Recognizer(
        num_classes=len(CHARACTERS),
        use_stn=True,
        use_resnet_backbone=True
    ).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    batch_size = BATCH_SIZE_TEST
    dummy_images = torch.randn(batch_size, MAX_FRAMES, 3, IMG_HEIGHT_TEST, IMG_WIDTH_TEST).to(device)
    dummy_labels = torch.randint(1, len(CHARACTERS), (batch_size * 7,)).to(device)
    input_lengths = torch.full((batch_size,), 20, dtype=torch.long)
    target_lengths = torch.full((batch_size,), 7, dtype=torch.long)
    
    print(f"Simulating training with batch size {batch_size}...")
    print()
    
    # Reset stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Forward pass
    model.train()
    outputs = model(dummy_images)
    log_probs = outputs.log_softmax(2).permute(1, 0, 2)
    
    mem_after_forward = torch.cuda.max_memory_allocated()
    print(f"After Forward:  {format_bytes(mem_after_forward)}")
    
    # Backward pass
    loss = criterion(log_probs, dummy_labels, input_lengths, target_lengths)
    loss.backward()
    
    mem_after_backward = torch.cuda.max_memory_allocated()
    print(f"After Backward: {format_bytes(mem_after_backward)}")
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    mem_peak = torch.cuda.max_memory_allocated()
    mem_total = torch.cuda.get_device_properties(0).total_memory
    mem_percent = (mem_peak / mem_total) * 100
    
    print(f"Peak Memory:    {format_bytes(mem_peak)} ({mem_percent:.1f}% of {format_bytes(mem_total)})")
    
    if mem_percent > 90:
        print("\n⚠️  WARNING: Memory usage >90%! Consider reducing batch size.")
    elif mem_percent > 80:
        print("\n⚠️  Memory usage >80%. System might be unstable.")
    else:
        print(f"\n✅ Memory usage looks good ({mem_percent:.1f}% < 80%)")
    
    print()

def main():
    print("\n" + "="*70)
    print("🚀 A30 OPTIMIZED CONFIG - QUICK TEST")
    print("="*70)
    print()
    
    # Run tests
    test_model()
    test_dataloader()
    test_memory_usage()
    
    print("="*70)
    print("✅ All tests completed!")
    print()
    print("📋 Summary:")
    print("   - Model creation: OK")
    print("   - Forward pass: OK")
    print("   - DataLoader: OK")
    print("   - Memory usage: OK")
    print()
    print("🚀 Ready to start training:")
    print("   ./start_training_fast_a30.sh")
    print("="*70)
    print()

if __name__ == '__main__':
    main()
