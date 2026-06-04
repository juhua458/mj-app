#!/usr/bin/env python3
"""
麻将牌数据标注工具
功能：
1. 定义麻将牌类别
2. 手动标注截图中的麻将牌
3. 生成YOLO格式标注文件
4. 可视化标注结果
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


# 麻将牌类别定义（贵阳捉鸡麻将）
MAHJONG_CLASSES = {
    # 万子 (0-8)
    0: "1wan", 1: "2wan", 2: "3wan", 3: "4wan", 4: "5wan",
    5: "6wan", 6: "7wan", 7: "8wan", 8: "9wan",
    # 筒子 (9-17)
    9: "1tong", 10: "2tong", 11: "3tong", 12: "4tong", 13: "5tong",
    14: "6tong", 15: "7tong", 16: "8tong", 17: "9tong",
    # 条子 (18-26)
    18: "1tiao", 19: "2tiao", 20: "3tiao", 21: "4tiao", 22: "5tiao",
    23: "6tiao", 24: "7tiao", 25: "8tiao", 26: "9tiao",
    # 风牌 (27-30)
    27: "dong", 28: "nan", 29: "xi", 30: "bei",
    # 箭牌 (31-33)
    31: "zhong", 32: "fa", 33: "bai",
    # 花牌（可选）
    34: "chun", 35: "xia", 36: "qiu", 37: "dong_ji",
    38: "mei", 39: "lan", 40: "zhu", 41: "ju",
}

# 反向映射
CLASS_TO_ID = {v: k for k, v in MAHJONG_CLASSES.items()}


class MahjongLabeler:
    """麻将牌标注器"""

    def __init__(self, image_dir: str, output_dir: str):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.labels_dir = os.path.join(output_dir, "labels")
        self.visualized_dir = os.path.join(output_dir, "visualized")

        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.visualized_dir, exist_ok=True)

        self.current_image = None
        self.current_annotations = []
        self.drawing = False
        self.start_point = None
        self.current_class = 0

    def create_manual_annotations(self):
        """
        为4张截图手动创建标注
        基于观察到的麻将牌位置和类别
        """
        # 标注数据（基于截图观察）
        annotations = {
            "bc37418d-0a40-4dfb-bbfb-5cfeab3fb643_Screenshot_2026-06-04-01-16-44-27_e39d2c7de19156b0683cd93e8735f348.jpg": {
                "size": (1758, 832),
                "hand_cards": [
                    # 手牌区域 (底部)
                    {"class": "4wan", "bbox": [0.080, 0.820, 0.130, 0.950]},
                    {"class": "8wan", "bbox": [0.140, 0.820, 0.190, 0.950]},
                    {"class": "2tiao", "bbox": [0.200, 0.820, 0.250, 0.950]},
                    {"class": "3tiao", "bbox": [0.260, 0.820, 0.310, 0.950]},
                    {"class": "4tiao", "bbox": [0.320, 0.820, 0.370, 0.950]},
                    {"class": "5tiao", "bbox": [0.380, 0.820, 0.430, 0.950]},
                    {"class": "6tiao", "bbox": [0.440, 0.820, 0.490, 0.950]},
                    {"class": "7tiao", "bbox": [0.500, 0.820, 0.550, 0.950]},
                    {"class": "5tong", "bbox": [0.560, 0.820, 0.610, 0.950]},
                    {"class": "6tong", "bbox": [0.620, 0.820, 0.670, 0.950]},
                    {"class": "7tong", "bbox": [0.680, 0.820, 0.730, 0.950]},
                    {"class": "8tong", "bbox": [0.740, 0.820, 0.790, 0.950]},
                    {"class": "9tong", "bbox": [0.800, 0.820, 0.850, 0.950]},
                    {"class": "1tong", "bbox": [0.860, 0.820, 0.910, 0.950]},
                ],
                "table_cards": [
                    # 桌面上的牌
                    {"class": "hongzhong", "bbox": [0.420, 0.350, 0.470, 0.430]},
                    {"class": "9tiao", "bbox": [0.520, 0.280, 0.570, 0.350]},
                    {"class": "6tong", "bbox": [0.580, 0.400, 0.630, 0.480]},
                ]
            },
            "13329393-1852-4599-a767-f648fe07db9f_Screenshot_2026-06-04-04-01-42-98_e39d2c7de19156b0683cd93e8735f348.jpg": {
                "size": (1758, 832),
                "hand_cards": [
                    {"class": "2wan", "bbox": [0.080, 0.820, 0.130, 0.950]},
                    {"class": "4wan", "bbox": [0.140, 0.820, 0.190, 0.950]},
                    {"class": "5wan", "bbox": [0.200, 0.820, 0.250, 0.950]},
                    {"class": "8wan", "bbox": [0.260, 0.820, 0.310, 0.950]},
                    {"class": "2tiao", "bbox": [0.320, 0.820, 0.370, 0.950]},
                    {"class": "3tiao", "bbox": [0.380, 0.820, 0.430, 0.950]},
                    {"class": "4tiao", "bbox": [0.440, 0.820, 0.490, 0.950]},
                    {"class": "6tiao", "bbox": [0.500, 0.820, 0.550, 0.950]},
                    {"class": "7tiao", "bbox": [0.560, 0.820, 0.610, 0.950]},
                    {"class": "5tong", "bbox": [0.620, 0.820, 0.670, 0.950]},
                    {"class": "6tong", "bbox": [0.680, 0.820, 0.730, 0.950]},
                    {"class": "7tong", "bbox": [0.740, 0.820, 0.790, 0.950]},
                    {"class": "8tong", "bbox": [0.800, 0.820, 0.850, 0.950]},
                    {"class": "2tong", "bbox": [0.860, 0.820, 0.910, 0.950]},
                ],
                "table_cards": [
                    {"class": "hongzhong", "bbox": [0.420, 0.350, 0.470, 0.430]},
                ]
            },
            "9eb2eb58-6968-461e-9f90-9c7d217aad56_Screenshot_2026-06-04-02-36-26-31_e39d2c7de19156b0683cd93e8735f348.jpg": {
                "size": (1758, 832),
                "hand_cards": [
                    {"class": "2wan", "bbox": [0.080, 0.820, 0.130, 0.950]},
                    {"class": "4wan", "bbox": [0.140, 0.820, 0.190, 0.950]},
                    {"class": "5wan", "bbox": [0.200, 0.820, 0.250, 0.950]},
                    {"class": "8wan", "bbox": [0.260, 0.820, 0.310, 0.950]},
                    {"class": "2tiao", "bbox": [0.320, 0.820, 0.370, 0.950]},
                    {"class": "3tiao", "bbox": [0.380, 0.820, 0.430, 0.950]},
                    {"class": "4tiao", "bbox": [0.440, 0.820, 0.490, 0.950]},
                    {"class": "6tiao", "bbox": [0.500, 0.820, 0.550, 0.950]},
                    {"class": "7tiao", "bbox": [0.560, 0.820, 0.610, 0.950]},
                    {"class": "5tong", "bbox": [0.620, 0.820, 0.670, 0.950]},
                    {"class": "6tong", "bbox": [0.680, 0.820, 0.730, 0.950]},
                    {"class": "7tong", "bbox": [0.740, 0.820, 0.790, 0.950]},
                    {"class": "8tong", "bbox": [0.800, 0.820, 0.850, 0.950]},
                    {"class": "2tong", "bbox": [0.860, 0.820, 0.910, 0.950]},
                ],
                "table_cards": [
                    {"class": "hongzhong", "bbox": [0.420, 0.350, 0.470, 0.430]},
                ]
            },
            "c84831e8-02e1-488f-b2d9-a1028e47b85b_Screenshot_2026-06-04-03-18-17-06_e39d2c7de19156b0683cd93e8735f348.jpg": {
                "size": (1758, 832),
                "hand_cards": [
                    {"class": "2wan", "bbox": [0.080, 0.820, 0.130, 0.950]},
                    {"class": "2wan", "bbox": [0.140, 0.820, 0.190, 0.950]},
                    {"class": "3wan", "bbox": [0.200, 0.820, 0.250, 0.950]},
                    {"class": "4wan", "bbox": [0.260, 0.820, 0.310, 0.950]},
                    {"class": "5wan", "bbox": [0.320, 0.820, 0.370, 0.950]},
                    {"class": "6wan", "bbox": [0.380, 0.820, 0.430, 0.950]},
                    {"class": "7wan", "bbox": [0.440, 0.820, 0.490, 0.950]},
                    {"class": "8wan", "bbox": [0.500, 0.820, 0.550, 0.950]},
                    {"class": "8wan", "bbox": [0.560, 0.820, 0.610, 0.950]},
                    {"class": "2tiao", "bbox": [0.620, 0.820, 0.670, 0.950]},
                    {"class": "5tong", "bbox": [0.680, 0.820, 0.730, 0.950]},
                    {"class": "6tong", "bbox": [0.740, 0.820, 0.790, 0.950]},
                    {"class": "7tong", "bbox": [0.800, 0.820, 0.850, 0.950]},
                    {"class": "8tong", "bbox": [0.860, 0.820, 0.910, 0.950]},
                ],
                "table_cards": [
                    {"class": "6tong", "bbox": [0.400, 0.300, 0.450, 0.380]},
                    {"class": "3tong", "bbox": [0.520, 0.250, 0.570, 0.330]},
                    {"class": "8tong", "bbox": [0.580, 0.400, 0.630, 0.480]},
                ]
            },
        }

        return annotations

    def convert_to_yolo_format(self, annotations: Dict):
        """将标注转换为YOLO格式"""
        for img_name, data in annotations.items():
            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"[WARNING] 图片不存在: {img_path}")
                continue

            img_h, img_w = data["size"]
            label_lines = []

            # 处理手牌和桌面牌
            for card in data.get("hand_cards", []) + data.get("table_cards", []):
                class_name = card["class"]
                if class_name not in CLASS_TO_ID:
                    print(f"[WARNING] 未知类别: {class_name}")
                    continue

                class_id = CLASS_TO_ID[class_name]
                bbox = card["bbox"]

                # YOLO格式: <class_id> <x_center> <y_center> <width> <height>
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 保存标注文件
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_name)

            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            print(f"[INFO] 已生成标注: {label_path} ({len(label_lines)} 个目标)")

    def visualize_annotations(self, annotations: Dict):
        """可视化标注结果"""
        for img_name, data in annotations.items():
            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            # 绘制所有标注
            for card in data.get("hand_cards", []) + data.get("table_cards", []):
                class_name = card["class"]
                bbox = card["bbox"]

                # 转换归一化坐标到像素坐标
                x1 = int(bbox[0] * img_w)
                y1 = int(bbox[1] * img_h)
                x2 = int(bbox[2] * img_w)
                y2 = int(bbox[3] * img_h)

                # 绘制矩形
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制类别标签
                label = f"{class_name}"
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 保存可视化结果
            vis_path = os.path.join(self.visualized_dir, img_name)
            cv2.imwrite(vis_path, img)
            print(f"[INFO] 已保存可视化: {vis_path}")

    def create_dataset_yaml(self, output_path: str = "dataset.yaml"):
        """创建YOLO数据集配置文件"""
        yaml_content = f"""path: {self.output_dir}
train: images
train: images
val: images
test: images

nc: {len(MAHJONG_CLASSES)}
names:
"""
        for i, name in MAHJONG_CLASSES.items():
            yaml_content += f"  {i}: {name}\n"

        with open(output_path, "w") as f:
            f.write(yaml_content)

        print(f"[INFO] 已生成数据集配置: {output_path}")


def main():
    """主函数"""
    image_dir = "/workspace/mj-project/test_images"
    output_dir = "/workspace/mj-project/dataset"

    labeler = MahjongLabeler(image_dir, output_dir)

    # 获取手动标注
    annotations = labeler.create_manual_annotations()

    # 转换为YOLO格式
    labeler.convert_to_yolo_format(annotations)

    # 可视化
    labeler.visualize_annotations(annotations)

    # 创建数据集配置
    labeler.create_dataset_yaml(os.path.join(output_dir, "dataset.yaml"))

    print("\n[INFO] 数据标注完成!")


if __name__ == "__main__":
    main()
