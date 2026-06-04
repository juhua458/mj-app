# Android APP 打包指南

## 项目概述

本项目是一个麻将牌面检测Android APP，基于ONNX Runtime Mobile进行本地推理。

## 环境要求

- Android Studio Hedgehog (2023.1.1) 或更高版本
- JDK 17+
- Android SDK (API 26-34)

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/juhua458/mj-app.git
cd mj-app/android_app
```

### 2. 用 Android Studio 打开

1. 打开 Android Studio
2. 选择 "Open an existing Android Studio project"
3. 选择 `mj-app/android_app` 目录
4. 等待 Gradle Sync 完成（首次需要下载依赖，约5-10分钟）

### 3. 构建 APK

**方式一：通过 Android Studio GUI**
1. 菜单栏 → Build → Build Bundle(s) / APK(s) → Build APK(s)
2. 等待构建完成
3. 右下角会弹出通知，点击 "locate" 找到 APK 文件
4. APK 路径：`app/build/outputs/apk/debug/app-debug.apk`

**方式二：通过命令行**
```bash
./gradlew assembleDebug
```

### 4. 安装到手机

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## APP 功能

1. **启动截屏检测**：点击"开始检测"按钮
2. **悬浮窗显示**：显示检测到的麻将牌数量
3. **实时更新**：每秒自动截屏并推理
4. **停止检测**：点击"停止检测"按钮

## 权限说明

APP 需要以下权限：
- **悬浮窗权限**：显示检测结果
- **截屏权限**：获取游戏画面
- **前台服务权限**：后台持续运行

## 项目结构

```
android_app/
├── app/
│   ├── src/main/
│   │   ├── java/com/mahjong/detector/
│   │   │   ├── MainActivity.java        # 主界面
│   │   │   ├── MahjongDetector.java     # ONNX推理引擎
│   │   │   ├── ScreenCaptureService.java # 截屏服务
│   │   │   └── FloatingViewService.java  # 悬浮窗服务
│   │   ├── res/                         # 布局和资源
│   │   └── assets/mahjong_detect.onnx   # ONNX模型
│   └── build.gradle                     # 模块配置
├── build.gradle                         # 项目配置
└── settings.gradle
```

## 模型信息

- **模型**：YOLOv8n 单类别检测（mahjong_tile）
- **输入**：640x640 RGB
- **输出**：检测框 + 置信度
- **opset**：17（兼容 Android ONNX Runtime 1.17.0）
- **大小**：约 12MB

## 常见问题

### Q: Gradle Sync 失败？
A: 检查网络连接，确保能访问 maven.google.com 和 repo1.maven.org

### Q: 构建失败，提示缺少 SDK？
A: 在 Android Studio 中打开 SDK Manager，安装 API 34 和 Build Tools 34.0.0

### Q: 安装后闪退？
A: 检查是否授予了悬浮窗权限和截屏权限

### Q: 检测不到牌？
A: 确保游戏画面在屏幕中央，手牌区域清晰可见

## 联系方式

项目仓库：https://github.com/juhua458/mj-app
