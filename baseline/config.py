import os
import torch


class Config:
    """Configuration class for LPR training."""
    
    # Data paths
    DATA_ROOT = "/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/train"
    VAL_SPLIT_FILE = "/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/val_tracks.json"
    
    # Image settings - Phase 1: Increased resolution for better feature extraction
    IMG_HEIGHT = 48
    IMG_WIDTH = 160
    
    # Character set (Brazilian + Mercosur plates: 7 characters)
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Training hyperparameters (optimized for A30 GPU - 24GB VRAM)
    BATCH_SIZE = 96  # Reduced due to larger image size
    LEARNING_RATE = 0.001
    EPOCHS = 50
    SEED = 42
    NUM_WORKERS = 8  # A30 có bandwidth tốt, tăng workers
    
    # Phase 1: New training parameters
    EARLY_STOPPING_PATIENCE = 7
    SAVE_TOP_K_MODELS = 3
    
    # Phase 1: Model architecture flags
    USE_STN = True  # Spatial Transformer Network
    USE_RESNET_BACKBONE = True  # ResNet-34 instead of vanilla CNN
    
    # Logging
    LOG_DIR = "runs"  # TensorBoard logs
    
    # Mixed precision training
    USE_AMP = True  # A30 hỗ trợ Tensor Cores tốt
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Character mappings (computed from CHARS)
    CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
    IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank
