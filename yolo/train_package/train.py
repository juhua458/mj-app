#!/usr/bin/env python3
"""YOLOv8n麻将牌识别训练脚本 - AutoDL一键运行"""
import os
import sys
import shutil

def main():
    print("=" * 50)
    print(" 麻将YOLOv8n训练")
    print("=" * 50)
    
    # 检查GPU
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU内存: {gpu_mem:.1f} GB")
    
    # 检查数据集
    data_yaml = "/root/mahjong_train/dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print("数据集未准备好，先运行 prepare_data.py")
        sys.exit(1)
    
    # 开始训练
    from ultralytics import YOLO
    print("加载YOLOv8n预训练模型...")
    model = YOLO("yolov8n.pt")
    
    print("开始训练 (约30-60分钟)...")
    results = model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=64,
        device=0,
        workers=8,
        project="/root/mahjong_train/runs",
        name="mahjong_detect",
        patience=30,
        save_period=10,
    )
    
    # 导出TFLite
    print("训练完成！导出TFLite模型...")
    best_pt = "/root/mahjong_train/runs/mahjong_detect/weights/best.pt"
    best_model = YOLO(best_pt)
    best_model.export(format="tflite", imgsz=640)
    
    # 复制到方便下载的位置
    tflite_dst = "/root/mahjong_train/mahjong_best.tflite"
    pt_dst = "/root/mahjong_train/mahjong_best.pt"
    
    # 查找tflite文件
    for root, dirs, files in os.walk("/root/mahjong_train/runs"):
        for f in files:
            if f.endswith('.tflite'):
                src = os.path.join(root, f)
                shutil.copy2(src, tflite_dst)
                size_mb = os.path.getsize(tflite_dst) / 1024 / 1024
                print(f"TFLite模型: {tflite_dst} ({size_mb:.1f} MB)")
                break
    
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, pt_dst)
        size_mb = os.path.getsize(pt_dst) / 1024 / 1024
        print(f"PyTorch模型: {pt_dst} ({size_mb:.1f} MB)")
    
    print("")
    print("=" * 50)
    print(" 全部完成！")
    print("=" * 50)
    print("请从AutoDL文件管理器下载:")
    print(f"  1. {tflite_dst}")
    print(f"  2. {pt_dst}")

if __name__ == "__main__":
    main()
