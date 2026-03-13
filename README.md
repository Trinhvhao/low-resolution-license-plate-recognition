# Google Drive Folder Downloader

Script Python để tải các file từ Google Drive folder về máy.

## Cài đặt

```bash
pip install -r requirements.txt
```

Hoặc cài trực tiếp:

```bash
pip install gdown
```

## Sử dụng

```bash
python download_drive_files.py
```

File sẽ được tải về thư mục `downloaded_files/`

## Lưu ý

- Folder Google Drive phải được set là **public** (Anyone with the link can view)
- Nếu folder là private, bạn cần sử dụng Google Drive API với authentication
- Script sẽ tự động tạo thư mục `downloaded_files` nếu chưa có

## Troubleshooting

Nếu gặp lỗi, thử:

1. Cập nhật gdown: `pip install --upgrade gdown`
2. Kiểm tra quyền truy cập folder
3. Thử tải từng file riêng lẻ nếu folder quá lớn
