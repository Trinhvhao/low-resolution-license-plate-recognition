# Phân Tích Dataset - License Plate Recognition

## 📊 Tổng Quan

**Dataset**: wYe7pBJ7-train.zip  
**Tổng số tracks**: 20,000  
**Kích thước**: ~2.5GB (compressed)

---

## 🗂️ Cấu Trúc Dataset

```
train/
├── Scenario-A/          # (nếu có)
└── Scenario-B/
    └── Brazilian/       # 20,000 tracks
        └── track_xxxxx/
            ├── annotations.json
            ├── lr-001.jpg   # Low-resolution frames (5 frames)
            ├── lr-002.jpg
            ├── lr-003.jpg
            ├── lr-004.jpg
            ├── lr-005.jpg
            ├── hr-001.jpg   # High-resolution frames (5 frames)
            ├── hr-002.jpg
            ├── hr-003.jpg
            ├── hr-004.jpg
            └── hr-005.jpg
```

---

## 📈 Thống Kê Chi Tiết

### Scenarios
- **Scenario-B**: 20,000 tracks (100%)
- **Scenario-A**: 0 tracks

### Plate Layouts
- **Brazilian**: 20,000 tracks (100%)
- Format: 3 chữ cái + 4 số (ví dụ: ATU7819)

### Plate Text Characteristics
- **Tổng số plates**: 20,000
- **Độ dài**: 7 ký tự (cố định)
- **Format**: `[A-Z]{3}[0-9]{4}`
- **Ký tự unique**: 36 (A-Z: 26, 0-9: 10)

### Character Distribution (Top 10)
| Ký tự | Số lần xuất hiện | Tỷ lệ |
|-------|------------------|-------|
| A | 684 | 68.4% |
| 7 | 443 | 44.3% |
| 2 | 425 | 42.5% |
| 9 | 413 | 41.3% |
| 5 | 409 | 40.9% |
| 1 | 405 | 40.5% |
| B | 399 | 39.9% |
| 0 | 392 | 39.2% |
| 6 | 389 | 38.9% |
| 3 | 382 | 38.2% |

**Nhận xét**: Ký tự 'A' xuất hiện rất nhiều (có thể là prefix của Brazilian plates)

### Frame Statistics
- **LR Frames**: 5 frames/track (cố định)
- **HR Frames**: 5 frames/track (cố định)
- **Tổng frames**: 200,000 (100K LR + 100K HR)

---

## 🔍 Sample Data

### Annotation Format
```json
{
  "plate_layout": "Brazilian",
  "plate_text": "ATU7819",
  "corners": {}
}
```

### Sample Plate Texts
```
ATU7819, AYY4046, GIY3981, ADP6372, AXV6959
AYX3391, AJT8959, ASZ2219, AZK4197, AWL0312
ABE5754, AKP0704, ACS8373, IJS7643, ESS1574
AYK1296, CWF5744, ARQ2220, ATQ3529, BCA9261
```

---

## ✅ Đánh Giá Dataset

### Điểm Mạnh
1. **Kích thước lớn**: 20K tracks là đủ để train model tốt
2. **Cân bằng frames**: Mỗi track có đủ 5 LR + 5 HR frames
3. **Format nhất quán**: Tất cả đều Brazilian plates, 7 ký tự
4. **Chất lượng tốt**: Có cả LR và HR cho augmentation

### Điểm Yếu
1. **Thiếu đa dạng**: Chỉ có Brazilian plates
2. **Không có Scenario-A**: Có thể thiếu một phần data
3. **Character imbalance**: Ký tự 'A' xuất hiện quá nhiều
4. **Fixed length**: Chỉ 7 ký tự, không linh hoạt

### Khuyến Nghị
1. **Data split**: 80% train (16K), 20% val (4K)
2. **Augmentation**: Cần thiết để tăng diversity
3. **Class weighting**: Cân nhắc weight cho ký tự hiếm
4. **Validation strategy**: K-fold hoặc stratified split

---

## 🎯 Cấu Hình Training

### Character Set
```python
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Không cần dấu '-' vì Brazilian plates không có
```

### Image Settings
- **Height**: 32 pixels
- **Width**: 128 pixels
- **Channels**: 3 (RGB)
- **Frames per sample**: 5

### Training Settings
```python
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
TRAIN_SPLIT = 0.8  # 16,000 tracks
VAL_SPLIT = 0.2    # 4,000 tracks
```

### Expected Performance
- **Baseline accuracy**: 85-90%
- **Target accuracy**: 95%+
- **Training time**: ~2-3 hours (với GPU)

---

## 📝 Notes

1. Dataset này phù hợp với baseline model hiện tại
2. Format cố định (7 ký tự) giúp model học nhanh hơn
3. Cần monitor character-level accuracy để phát hiện bias
4. Có thể cần thêm data từ Scenario-A nếu có
