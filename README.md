# 捉鸡麻将AI识别项目

## 项目概述

手机端自动识别麻将牌面 + 策略提示（微乐麻将·贵州捉鸡）

## 技术方案

- **模型**: YOLOv8 → ONNX格式
- **推理**: Android本地推理（ONNX Runtime Mobile）
- **设备**: 一加13T，Android 15

## 项目结构

```
mj-project/
├── models/                 # ONNX模型文件
│   ├── mahjong-yolon-best.onnx      # 原始模型（opset20）
│   └── mahjong-yolon-best_opset17.onnx  # 降级模型（opset17）
├── test_images/            # 测试截图
├── scripts/                # Python工具脚本
│   ├── benchmark.py        # 基准测试框架
│   ├── convert_opset.py    # ONNX模型opset降级工具
│   ├── android_test.py     # Android自动化测试
│   └── requirements.txt    # Python依赖
├── android_app/            # Android项目代码
└── test_results/           # 测试结果
```

## 快速开始

### 1. 安装依赖

```bash
cd scripts
pip install -r requirements.txt
```

### 2. 模型opset降级（解决Android兼容性问题）

```bash
python convert_opset.py \
    --model ../models/mahjong-yolon-best.onnx \
    --opset 17 \
    --verify \
    --report ../test_results/convert_report.json
```

### 3. Python端基准测试

```bash
python benchmark.py \
    --model ../models/mahjong-yolon-best_opset17.onnx \
    --images ../test_images \
    --output ../test_results/python_benchmark
```

### 4. Android端自动化测试

```bash
# 确保手机连接并开启USB调试
python android_test.py \
    --model ../models/mahjong-yolon-best_opset17.onnx \
    --package com.example.mjapp \
    --tests 10 \
    --output ../test_results/android_benchmark
```

## 已知问题与解决方案

### 问题：Android端检测数只有2个（Python端有103个）

**根因**: ONNX模型opset=20 与 Android ONNX Runtime Mobile 1.17.0 不兼容

**解决方案**:
1. 模型opset降级 20→17（主方案）
2. 升级Android ORT 1.17.0→1.21+（辅助方案）

**建议**: 双保险，两个方案同时实施

## 测试结果

### Python端（基准）
- 检测数: 103~111（置信度>0.25）
- 推理时间: ~50ms

### Android端（修复前）
- 检测数: 2（置信度>0.25）
- 问题: opset不兼容导致Split算子解析错误

### Android端（修复后）
- 待验证...

## 开发计划

- [x] 问题排查与根因定位
- [x] Python自动化测试框架
- [x] ONNX模型opset降级工具
- [ ] Android推理模块修复
- [ ] Android自动化测试
- [ ] 策略提示模块
- [ ] 性能优化

## 项目仓库

https://github.com/juhua458/mj-app
