#!/usr/bin/env python3
"""训练YOLOv8麻将牌检测模型"""
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

model = YOLO('yolov8n.pt')

results = model.train(
    data='dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
    name='train_mahjong',
    project='runs',
    exist_ok=True,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5,
    translate=0.1,
    scale=0.2,
    shear=2,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    device='cpu'
)
print('Training complete!')
