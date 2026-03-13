"""
Quick test script for Phase 1 implementation.
Validates model creation, forward pass, and basic functionality.
"""

import sys
import os

# Add baseline to path
sys.path.insert(0, os.path.dirname(__file__))

import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# Test imports
print("\n📦 Testing imports...")
try:
    from config import Config
    print("  ✓ Config imported")
    
    from models import Phase1Recognizer, STN, ResNet34Backbone
    print("  ✓ Phase1Recognizer imported")
    print("  ✓ STN imported")
    print("  ✓ ResNet34Backbone imported")
    
    from utils import calculate_cer, calculate_accuracy, calculate_confidence_gap
    print("  ✓ Metrics imported")
    
except Exception as e:
    print(f"  ❌ Import error: {e}")
    sys.exit(1)

# Test config changes
print(f"\n⚙️ Config check:")
print(f"  - Image size: {Config.IMG_HEIGHT}×{Config.IMG_WIDTH}")
print(f"  - Batch size: {Config.BATCH_SIZE}")
print(f"  - USE_STN: {Config.USE_STN}")
print(f"  - USE_RESNET_BACKBONE: {Config.USE_RESNET_BACKBONE}")
print(f"  - Early stopping patience: {Config.EARLY_STOPPING_PATIENCE}")

# Test model creation
print(f"\n🏗️ Testing model creation...")
try:
    # Test with STN + ResNet
    model = Phase1Recognizer(
        num_classes=Config.NUM_CLASSES,
        use_stn=True,
        use_resnet_backbone=True
    )
    print(f"  ✓ Phase1Recognizer created (STN + ResNet)")
    print(f"  ✓ Model info: {model.get_model_info()}")
    
    # Test without STN
    model_no_stn = Phase1Recognizer(
        num_classes=Config.NUM_CLASSES,
        use_stn=False,
        use_resnet_backbone=True
    )
    print(f"  ✓ Model created without STN")
    
    # Test with vanilla CNN
    model_vanilla = Phase1Recognizer(
        num_classes=Config.NUM_CLASSES,
        use_stn=False,
        use_resnet_backbone=False
    )
    print(f"  ✓ Model created with VanillaCNN")
    
except Exception as e:
    print(f"  ❌ Model creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test forward pass
print(f"\n🔄 Testing forward pass...")
try:
    model.eval()
    
    # Create dummy input [B, T, C, H, W]
    batch_size = 2
    num_frames = 5
    dummy_input = torch.randn(batch_size, num_frames, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    
    print(f"  Input shape: {list(dummy_input.shape)}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Expected: [B={batch_size}, W, num_classes={Config.NUM_CLASSES}]")
    
    # Validate output shape
    assert output.size(0) == batch_size, "Batch size mismatch"
    assert output.size(2) == Config.NUM_CLASSES, "Num classes mismatch"
    
    print(f"  ✓ Forward pass successful")
    
except Exception as e:
    print(f"  ❌ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test metrics
print(f"\n📊 Testing metrics...")
try:
    predictions = ["ABC1234", "XYZ5678", "ABC1234"]
    targets = ["ABC1234", "XYZ5679", "ABC1235"]
    
    acc = calculate_accuracy(predictions, targets)
    cer = calculate_cer(predictions, targets)
    
    print(f"  Predictions: {predictions}")
    print(f"  Targets:     {targets}")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  CER: {cer:.4f}")
    
    # Test confidence gap
    confidences = [0.95, 0.85, 0.90]
    is_correct = [True, False, False]
    gap = calculate_confidence_gap(confidences, is_correct)
    print(f"  Confidence Gap: {gap:.4f}")
    
    print(f"  ✓ Metrics working")
    
except Exception as e:
    print(f"  ❌ Metrics error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test parameter count
print(f"\n📈 Model statistics:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

print(f"\n✅ All Phase 1 tests passed!")
print(f"\n📝 Next steps:")
print(f"  1. Install dependencies: pip install -r requirements.txt")
print(f"  2. Start training: python train.py")
print(f"  3. Monitor with TensorBoard: tensorboard --logdir runs")
