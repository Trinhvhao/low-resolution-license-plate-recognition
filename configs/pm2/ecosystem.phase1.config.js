module.exports = {
  apps: [{
    name: 'lpr-phase1',
    script: 'python',
    args: 'train.py',
    cwd: '/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/OCR_ICPR/baseline',
    interpreter: 'none',
    instances: 1,
    autorestart: false,
    watch: false,
    max_memory_restart: '20G',
    env: {
      CUDA_VISIBLE_DEVICES: '0',
      PYTHONUNBUFFERED: '1'
    },
    error_file: '../logs/phase1-error.log',
    out_file: '../logs/phase1-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
    merge_logs: true,
    time: true
  }]
};
