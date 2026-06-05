#!/usr/bin/env python3
"""导出ONNX模型"""
from ultralytics import YOLO
import os

model = YOLO('runs/train_mahjong/weights/best.pt')
model.export(format='onnx', opset=17, imgsz=640)

# 复制到models目录
os.makedirs('models', exist_ok=True)
os.system('cp runs/train_mahjong/weights/best.onnx models/mahjong_detect.onnx')
print('ONNX export complete!')
