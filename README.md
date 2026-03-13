# Low-Resolution License Plate Recognition - ICPR

Một dự án nhận dạng biển số xe từ ảnh độ phân giải thấp sử dụng các kỹ thuật Deep Learning tiên tiến.

## 📋 Mô Tả Dự Án

Dự án này được phát triển cho cuộc thi ICPR (International Conference on Pattern Recognition) nhằm giải quyết bài toán nhận dạng biển số xe từ các ảnh có độ phân giải thấp. Bao gồm:

- **Super-Resolution (SR)**: Nâng cao độ phân giải ảnh biển số
- **Detection**: Phát hiện vị trí biển số trong ảnh
- **Recognition (OCR)**: Nhận dạng ký tự trên biển số
- **Post-processing**: Xử lý sau và gộp kết quả

## 🎯 Đặc Điểm Chính

- Multi-phase training pipeline với 13 giai đoạn tinh chỉnh
- Ensemble models cho kết quả tốt hơn
- SOTA architectures: Transformer, Attention mechanisms, STN
- Support cho Scenario A và Scenario B datasets
- Training trên GPU A30/A100 với distributed computing

## 📁 Cấu Trúc Thư Mục

```
.
├── baseline/                 # Core model implementation
│   ├── models/              # Model architectures
│   ├── scripts/             # Training & inference scripts
│   ├── dataset.py           # Data loading
│   ├── config.py            # Configuration
│   ├── pipeline.py          # Inference pipeline
│   └── transforms.py        # Data augmentation
├── best_models/             # Trained model checkpoints
├── dataset/                 # Training/validation data
│   ├── Scenario-A/
│   └── Scenario-B/
├── configs/                 # Configuration files (PM2, etc)
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

## 🚀 Cài Đặt

### Yêu Cầu

- Python 3.8+
- PyTorch >= 1.10
- CUDA 11.0+ (tùy chọn, nhưng khuyên dùng)

### Setup

```bash
# Clone repository
git clone https://github.com/Trinhvhao/low-resolution-license-plate-recognition.git
cd low-resolution-license-plate-recognition

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc cài từ baseline
pip install -r baseline/requirements.txt
```

## 🏋️ Training

```bash
# Training phase 1
cd baseline
python scripts/train/train_phase1.py

# Hoặc các giai đoạn khác
python scripts/train/train_phase10.py
```

## 🔍 Inference

```bash
python baseline/inference.py --model best_models/best_model_phase10.pth --input image.jpg
```

Hoặc sử dụng pipeline:

```python
from baseline.pipeline import OCRPipeline

pipeline = OCRPipeline(model_path='best_models/best_model_phase10.pth')
result = pipeline.predict('image.jpg')
print(result)
```

## 📊 Kết Quả

- **Best Model Phase 10**: Đạt hiệu năng tốt nhất trên test set
- Multi-phase optimization với SWA (Stochastic Weight Averaging)
- Ensemble kết quả từ nhiều mô hình

## 🛠️ Công Nghệ Sử Dụng

- PyTorch - Deep Learning framework
- CRNN - Convolutional Recurrent Neural Network
- Transformer - Attention-based architecture
- STN - Spatial Transformer Network
- Super-Resolution - Nâng cao độ phân giải

## 📝 Ghi Chú

- Dataset được lưu trong thư mục `dataset/`
- Model checkpoints có kích thước lớn, tải riêng nếu cần
- Xem `docs/` để biết thêm chi tiết về project structure

## 👨‍💻 Tác Giả

Trình Văn Hào
