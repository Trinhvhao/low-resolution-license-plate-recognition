module.exports = {
  apps: [{
    name: 'lpr-fast-a30',
    script: 'train_fast_a30.py',
    cwd: '/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/baseline',
    interpreter: 'python',
    instances: 1,
    autorestart: false,
    watch: false,
    max_memory_restart: '20G',
    env: {
      'CUDA_VISIBLE_DEVICES': '0',
      'PYTHONUNBUFFERED': '1'
    },
    error_file: '/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/logs/fast-a30-error.log',
    out_file: '/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/logs/fast-a30-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    time: true
  }]
};
