# 麻将YOLOv8训练包

## AutoDL一键训练指南

### 第1步: 租用GPU实例
在AutoDL算力市场租一个最便宜的GPU（RTX 3060/4090D都行），选按量计费

### 第2步: 打开Terminal
在实例页面点"自定义服务" -> "JupyterLab" -> 打开Terminal

### 第3步: 一键运行
```bash
cd /root
git clone https://github.com/juhua458/mj-app.git mj_train_tmp
cp -r mj_train_tmp/yolo/train_package mahjong_train
cd mahjong_train

# 解压数据集图片
unzip -o images_raw.zip -d .

# 一键训练
bash setup_and_train.sh
```

### 第4步: 下载模型
训练完成后，从文件管理器下载:
- `/root/mahjong_train/mahjong_best.tflite` (Android端用)
- `/root/mahjong_train/mahjong_best.pt` (备用)

### 预计耗时
- 数据准备: 1-2分钟
- 模型训练: 30-60分钟
- 模型导出: 1-2分钟
