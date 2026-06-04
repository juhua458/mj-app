#!/usr/bin/env python3
"""
麻将牌检测模型训练脚本
功能：
1. 下载YOLOv8预训练模型
2. 使用标注数据训练
3. 导出ONNX格式（opset=17）
4. 验证模型效果
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def install_ultralytics():
    """安装ultralytics库"""
    try:
        import ultralytics
        print("[INFO] ultralytics已安装")
    except ImportError:
        print("[INFO] 正在安装ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
        print("[INFO] ultralytics安装完成")


def train_model(data_yaml: str, epochs: int = 100, imgsz: int = 640,
                batch: int = 4, model_size: str = "n"):
    """
    训练YOLOv8模型

    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: 批次大小
        model_size: 模型大小 (n/s/m/l/x)
    """
    from ultralytics import YOLO

    # 加载预训练模型
    model_name = f"yolov8{model_size}.pt"
    print(f"[INFO] 加载预训练模型: {model_name}")

    # 检查模型是否存在，不存在则下载
    if not os.path.exists(model_name):
        print(f"[INFO] 下载预训练模型...")

    model = YOLO(model_name)

    # 训练
    print(f"[INFO] 开始训练...")
    print(f"[INFO] 数据集: {data_yaml}")
    print(f"[INFO] 轮数: {epochs}")
    print(f"[INFO] 图像尺寸: {imgsz}")
    print(f"[INFO] 批次大小: {batch}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,  # 早停耐心值
        save=True,
        project="/workspace/mj-project/runs",
        name="train_mahjong",
        exist_ok=True,
        pretrained=True,
        optimizer="Adam",  # 使用Adam优化器
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 最终学习率
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,  # 框损失增益
        cls=0.5,  # 分类损失增益
        dfl=1.5,  # 分布焦点损失增益
        augment=True,  # 启用数据增强
        mosaic=1.0,  # mosaic增强
        mixup=0.0,  # mixup增强
        copy_paste=0.0,  # copy-paste增强
        degrees=5.0,  # 旋转角度
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        shear=2.0,  # 剪切
        perspective=0.0,  # 透视
        flipud=0.0,  # 上下翻转
        fliplr=0.5,  # 左右翻转
        bgr=0.0,  # BGR通道
        hsv_h=0.015,  # HSV色调
        hsv_s=0.7,  # HSV饱和度
        hsv_v=0.4,  # HSV亮度
    )

    print(f"[INFO] 训练完成!")
    print(f"[INFO] 最佳模型: {results.best}")

    return results


def export_onnx(model_path: str, imgsz: int = 640, opset: int = 17):
    """
    导出ONNX模型

    Args:
        model_path: 训练好的模型路径
        imgsz: 输入图像尺寸
        opset: ONNX opset版本
    """
    from ultralytics import YOLO

    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] 导出ONNX格式...")
    print(f"[INFO] opset版本: {opset}")

    # 导出ONNX
    model.export(
        format='onnx',
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,
    )

    # 获取导出路径
    onnx_path = os.path.splitext(model_path)[0] + '.onnx'

    if os.path.exists(onnx_path):
        print(f"[INFO] ONNX导出成功: {onnx_path}")

        # 显示文件大小
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[INFO] 文件大小: {size_mb:.2f} MB")

        return onnx_path
    else:
        print(f"[ERROR] ONNX导出失败")
        return None


def validate_model(model_path: str, data_yaml: str, imgsz: int = 640):
    """
    验证模型效果

    Args:
        model_path: 模型路径
        data_yaml: 数据集配置
        imgsz: 图像尺寸
    """
    from ultralytics import YOLO

    print(f"[INFO] 验证模型: {model_path}")
    model = YOLO(model_path)

    # 验证
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=4,
        conf=0.25,
        iou=0.45,
        max_det=300,
    )

    print(f"\n{'='*60}")
    print("验证结果")
    print(f"{'='*60}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"{'='*60}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="麻将牌检测模型训练")
    parser.add_argument("--data", "-d", default="/workspace/mj-project/dataset/dataset.yaml",
                       help="数据集配置文件")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", "-s", type=int, default=640, help="图像尺寸")
    parser.add_argument("--batch", "-b", type=int, default=4, help="批次大小")
    parser.add_argument("--model", "-m", default="n", help="模型大小 (n/s/m/l/x)")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--export", action="store_true", help="导出ONNX")
    parser.add_argument("--validate", action="store_true", help="验证模型")
    parser.add_argument("--model-path", help="模型路径（用于导出/验证）")

    args = parser.parse_args()

    # 安装依赖
    install_ultralytics()

    # 训练
    if args.train:
        results = train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            model_size=args.model
        )

        # 自动导出ONNX
        best_model = os.path.join("/workspace/mj-project/runs", "train_mahjong", "weights", "best.pt")
        if os.path.exists(best_model):
            export_onnx(best_model, args.imgsz, opset=17)

    # 导出ONNX
    if args.export and args.model_path:
        export_onnx(args.model_path, args.imgsz, opset=17)

    # 验证
    if args.validate and args.model_path:
        validate_model(args.model_path, args.data, args.imgsz)


if __name__ == "__main__":
    main()
