#!/bin/bash
set -e
echo "=========================================="
echo " 麻将YOLO训练 - 一键部署脚本"
echo "=========================================="

echo "[1/4] 安装依赖..."
pip install ultralytics opencv-python-headless -q -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[2/4] 准备数据集..."
cd /root/mahjong_train
python3 prepare_data.py

echo "[3/4] 开始训练 (约30-60分钟)..."
python3 train.py

echo "[4/4] 完成！"
echo ""
echo "下载TFLite模型: /root/mahjong_train/mahjong_best.tflite"
echo "下载PT模型: /root/mahjong_train/mahjong_best.pt"
