# 捉鸡麻将AI识别项目

手机端自动识别麻将牌面 + 策略提示（微乐麻将·贵州捉鸡）

## 项目概述

本项目使用YOLOv8模型进行麻将牌面检测，通过MediaProjection API截屏，在Android端进行ONNX本地推理，实时识别手牌数量。

## 技术方案

- **模型**: YOLOv8n → ONNX格式 (opset=17)
- **推理**: Android ONNX Runtime Mobile 1.17.0
- **截屏**: MediaProjection API
- **设备**: 一加13T，Android 15

## 项目结构

```
mj-project/
├── models/                          # ONNX模型文件
│   └── mahjong_detect.onnx         # 训练好的检测模型 (opset17, ~12MB)
├── scripts/                         # Python工具脚本
│   ├── benchmark.py                # 基准测试框架
│   ├── convert_opset.py            # ONNX模型opset降级工具
│   ├── android_test.py             # Android自动化测试
│   ├── label_mahjong.py            # 数据标注工具
│   ├── data_augmentation.py        # 数据增强
│   ├── train_model.py              # 模型训练脚本
│   └── requirements.txt            # Python依赖
├── android_app/                     # Android项目
│   ├── app/src/main/
│   │   ├── java/com/mahjong/detector/
│   │   │   ├── MainActivity.java        # 主界面
│   │   │   ├── MahjongDetector.java     # ONNX推理引擎
│   │   │   ├── ScreenCaptureService.java # 截屏服务
│   │   │   └── FloatingViewService.java  # 悬浮窗服务
│   │   ├── res/                         # 布局和资源
│   │   └── assets/mahjong_detect.onnx   # ONNX模型
│   ├── build.gradle
│   └── BUILD_GUIDE.md              # Android打包指南
├── test_images/                     # 测试截图
├── dataset/                         # 训练数据集
└── README.md                        # 本文件
```

## 快速开始

### 方式一：使用预训练模型（推荐）

模型已训练好并导出为ONNX格式，直接使用：

```bash
cd android_app
# 用 Android Studio 打开并打包APK（详见 BUILD_GUIDE.md）
```

### 方式二：重新训练模型

```bash
cd scripts
pip install -r requirements.txt

# 1. 标注数据
python label_mahjong.py

# 2. 数据增强
python data_augmentation.py --num-aug 50 --split

# 3. 训练模型
python train_model.py --train --epochs 100

# 4. 导出ONNX（opset=17）
python train_model.py --export --model-path ../runs/train_mahjong/weights/best.pt
```

### 方式三：模型opset降级（已有模型但opset不兼容）

```bash
python convert_opset.py \
    --model model.onnx \
    --opset 17 \
    --verify \
    --report convert_report.json
```

## Android APP 使用

1. **安装APK**：`adb install app-debug.apk`
2. **授予权限**：悬浮窗权限 + 截屏权限
3. **打开APP**：点击"开始检测"
4. **查看结果**：悬浮窗显示检测到的麻将牌数量

## 模型信息

| 属性 | 值 |
|------|-----|
| 模型 | YOLOv8n |
| 类别 | 1类 (mahjong_tile) |
| 输入 | 640x640 RGB |
| 输出 | 检测框 + 置信度 |
| opset | 17 |
| 大小 | ~12MB |
| 训练数据 | 4张截图（数据增强后） |

## 已知问题

- 当前模型基于4张截图训练，检测精度有限
- 建议收集更多截图重新训练以提高准确率
- 只检测手牌区域，不识别具体牌面类别

## 开发计划

- [x] 问题排查与根因定位
- [x] Python自动化测试框架
- [x] ONNX模型opset降级工具
- [x] 数据标注工具
- [x] 数据增强
- [x] 模型训练
- [x] Android推理模块
- [ ] 策略提示功能（下一步）
- [ ] 多类别牌面识别

## 项目仓库

https://github.com/juhua458/mj-app
