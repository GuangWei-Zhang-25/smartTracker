@echo off
set POLARS_SKIP_CPU_CHECK=1
py C:\Users\guangwei\Documents\SmartTracker\train_yolo.py D:\Project\Vergin_female_parental\mouse_dataset\dataset.yaml --model n --epochs 50 --imgsz 640 --batch 8 --device 0 --name yolon_seg_final
pause
