#!/usr/bin/env python3
"""
麻将牌数据增强工具
功能：
1. 对现有截图进行多种数据增强
2. 生成更多训练样本
3. 保持YOLO格式标注同步变换
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class MahjongAugmenter:
    """麻将牌数据增强器"""

    def __init__(self, image_dir: str, label_dir: str, output_dir: str):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.output_images_dir = os.path.join(output_dir, "images")
        self.output_labels_dir = os.path.join(output_dir, "labels")

        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)

        # 定义增强变换
        self.transform = A.Compose([
            # 颜色变换
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            ], p=0.8),

            # 噪声
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),

            # 模糊
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # 几何变换（保持bbox同步）
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),

            # 随机裁剪
            A.RandomResizedCrop(
                height=832,
                width=1758,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.3
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def load_yolo_labels(self, label_path: str) -> Tuple[List, List]:
        """加载YOLO格式标注"""
        bboxes = []
        class_labels = []

        if not os.path.exists(label_path):
            return bboxes, class_labels

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)

        return bboxes, class_labels

    def save_yolo_labels(self, label_path: str, bboxes: List, class_labels: List):
        """保存YOLO格式标注"""
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    def augment_image(self, image_path: str, label_path: str, num_augmentations: int = 10):
        """
        对单张图片进行数据增强

        Args:
            image_path: 原图路径
            label_path: 标注文件路径
            num_augmentations: 增强数量
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] 无法读取图片: {image_path}")
            return

        # 读取标注
        bboxes, class_labels = self.load_yolo_labels(label_path)

        if not bboxes:
            print(f"[WARNING] 没有标注: {label_path}")
            return

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 保存原图
        orig_img_path = os.path.join(self.output_images_dir, f"{base_name}_orig.jpg")
        orig_label_path = os.path.join(self.output_labels_dir, f"{base_name}_orig.txt")
        cv2.imwrite(orig_img_path, image)
        self.save_yolo_labels(orig_label_path, bboxes, class_labels)

        # 生成增强样本
        for i in range(num_augmentations):
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_class_labels = transformed['class_labels']

                # 保存增强后的图片
                aug_img_path = os.path.join(self.output_images_dir, f"{base_name}_aug_{i:03d}.jpg")
                aug_label_path = os.path.join(self.output_labels_dir, f"{base_name}_aug_{i:03d}.txt")

                cv2.imwrite(aug_img_path, aug_image)
                self.save_yolo_labels(aug_label_path, aug_bboxes, aug_class_labels)

            except Exception as e:
                print(f"[WARNING] 增强失败 {base_name}_{i}: {e}")

        print(f"[INFO] {base_name}: 原图 + {num_augmentations} 张增强")

    def process_all(self, num_augmentations: int = 10):
        """处理所有图片"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(self.image_dir).glob(ext))

        print(f"[INFO] 找到 {len(image_files)} 张图片")

        for img_path in image_files:
            img_path = str(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, f"{base_name}.txt")

            self.augment_image(img_path, label_path, num_augmentations)

        # 统计结果
        num_output_images = len(list(Path(self.output_images_dir).glob('*.jpg')))
        num_output_labels = len(list(Path(self.output_labels_dir).glob('*.txt')))

        print(f"\n[INFO] 数据增强完成!")
        print(f"[INFO] 输出图片: {num_output_images}")
        print(f"[INFO] 输出标注: {num_output_labels}")


def create_train_val_split(dataset_dir: str, train_ratio: float = 0.8):
    """
    划分训练集和验证集

    Args:
        dataset_dir: 数据集目录
        train_ratio: 训练集比例
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    # 获取所有图片
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    # 随机打乱
    random.shuffle(image_files)

    # 划分
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # 创建目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    # 移动文件
    for f in train_files:
        base_name = os.path.splitext(f)[0]
        # 移动图片
        src_img = os.path.join(images_dir, f)
        dst_img = os.path.join(images_dir, "train", f)
        os.rename(src_img, dst_img)

        # 移动标注
        src_label = os.path.join(labels_dir, f"{base_name}.txt")
        dst_label = os.path.join(labels_dir, "train", f"{base_name}.txt")
        if os.path.exists(src_label):
            os.rename(src_label, dst_label)

    for f in val_files:
        base_name = os.path.splitext(f)[0]
        src_img = os.path.join(images_dir, f)
        dst_img = os.path.join(images_dir, "val", f)
        os.rename(src_img, dst_img)

        src_label = os.path.join(labels_dir, f"{base_name}.txt")
        dst_label = os.path.join(labels_dir, "val", f"{base_name}.txt")
        if os.path.exists(src_label):
            os.rename(src_label, dst_label)

    print(f"[INFO] 数据集划分完成:")
    print(f"[INFO] 训练集: {len(train_files)}")
    print(f"[INFO] 验证集: {len(val_files)}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="麻将牌数据增强工具")
    parser.add_argument("--images", "-i", default="/workspace/mj-project/test_images", help="图片目录")
    parser.add_argument("--labels", "-l", default="/workspace/mj-project/dataset/labels", help="标注目录")
    parser.add_argument("--output", "-o", default="/workspace/mj-project/dataset", help="输出目录")
    parser.add_argument("--num-aug", "-n", type=int, default=20, help="每张图增强数量")
    parser.add_argument("--split", action="store_true", help="划分训练集和验证集")

    args = parser.parse_args()

    # 数据增强
    augmenter = MahjongAugmenter(args.images, args.labels, args.output)
    augmenter.process_all(args.num_aug)

    # 划分数据集
    if args.split:
        create_train_val_split(args.output)


if __name__ == "__main__":
    main()
