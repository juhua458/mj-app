#!/usr/bin/env python3
"""将Camerash/mahjong-dataset转为YOLO格式"""
import os
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

SRC_IMG = "/root/mahjong_train/images_raw"
CSV_FILE = "/root/mahjong_train/data.csv"
OUT_DIR = "/root/mahjong_train/dataset"

# 42类映射
LABEL_MAP = {
    "characters-1": 0, "characters-2": 1, "characters-3": 2,
    "characters-4": 3, "characters-5": 4, "characters-6": 5,
    "characters-7": 6, "characters-8": 7, "characters-9": 8,
    "dots-1": 9, "dots-2": 10, "dots-3": 11,
    "dots-4": 12, "dots-5": 13, "dots-6": 14,
    "dots-7": 15, "dots-8": 16, "dots-9": 17,
    "bamboo-1": 18, "bamboo-2": 19, "bamboo-3": 20,
    "bamboo-4": 21, "bamboo-5": 22, "bamboo-6": 23,
    "bamboo-7": 24, "bamboo-8": 25, "bamboo-9": 26,
    "honors-east": 27, "honors-south": 28, "honors-west": 29, "honors-north": 30,
    "honors-red": 31, "honors-green": 32, "honors-white": 33,
    "bonus-spring": 34, "bonus-summer": 35, "bonus-autumn": 36, "bonus-winter": 37,
    "bonus-plum": 38, "bonus-orchid": 39, "bonus-chrysanthemum": 40, "bonus-bamboo": 41,
}

NAMES_CN = [
    "yi_wan","er_wan","san_wan","si_wan","wu_wan","liu_wan","qi_wan","ba_wan","jiu_wan",
    "yi_tong","er_tong","san_tong","si_tong","wu_tong","liu_tong","qi_tong","ba_tong","jiu_tong",
    "yi_tiao","er_tiao","san_tiao","si_tiao","wu_tiao","liu_tiao","qi_tiao","ba_tiao","jiu_tiao",
    "dong_feng","nan_feng","xi_feng","bei_feng","hong_zhong","fa_cai","bai_ban",
    "chun","xia","qiu","dong","mei","lan","ju","zhu"
]

def main():
    for split in ['train', 'val']:
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

    samples = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['image-name']
            label_name = row['label-name']
            if label_name in LABEL_MAP:
                samples.append((img_name, LABEL_MAP[label_name]))

    print(f"总样本数: {len(samples)}")
    
    by_class = defaultdict(list)
    for img_name, cls_id in samples:
        by_class[cls_id].append(img_name)
    
    train_imgs = set()
    val_imgs = set()
    for cls_id, imgs in by_class.items():
        n = len(imgs)
        n_val = max(1, n // 5)
        random.seed(42)
        random.shuffle(imgs)
        for img in imgs[:n_val]:
            val_imgs.add(img)
        for img in imgs[n_val:]:
            train_imgs.add(img)
    
    print(f"训练集: {len(train_imgs)} 张, 验证集: {len(val_imgs)} 张")
    
    def create_label(img_name, cls_id, split):
        label_line = f"{cls_id} 0.5 0.5 0.9 0.9\n"
        src_path = f"{SRC_IMG}/{img_name}"
        dst_img = f"{OUT_DIR}/images/{split}/{img_name}"
        dst_lbl = f"{OUT_DIR}/labels/{split}/{img_name.rsplit('.',1)[0]}.txt"
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_img)
            with open(dst_lbl, 'w') as f:
                f.write(label_line)
            return True
        return False

    train_count = 0
    val_count = 0
    for img_name, cls_id in samples:
        if img_name in train_imgs:
            if create_label(img_name, cls_id, 'train'):
                train_count += 1
        else:
            if create_label(img_name, cls_id, 'val'):
                val_count += 1

    print(f"实际复制: 训练集 {train_count} 张, 验证集 {val_count} 张")

    yaml_content = f"""path: {OUT_DIR}
train: images/train
val: images/val

nc: {len(NAMES_CN)}
names: {NAMES_CN}
"""
    with open(f"{OUT_DIR}/data.yaml", 'w') as f:
        f.write(yaml_content)
    print(f"data.yaml 已生成, 类别数: {len(NAMES_CN)}")
    print("数据准备完成！")

if __name__ == "__main__":
    main()
