# 麻将截屏助手

配合「捉鸡麻将提示器v9」使用的Android App，提供自动截屏服务。

## 功能
- 后台每秒自动截取手机屏幕
- 通过HTTP服务提供最新截图（`http://127.0.0.1:8666/api/screenshot`）
- 浏览器v9自动拉取识别，全程零操作

## 使用方法
1. 安装APK
2. 打开App → 点「开始截屏」→ 允许录屏权限
3. 分屏或切到浏览器，打开v9的「🔄自动截屏」模式
4. 回到麻将App正常玩，v9自动识别

## 编译
推送到GitHub后，Actions自动编译APK。

## 技术栈
- Kotlin + Android SDK 34
- MediaProjection截屏
- NanoHTTPD提供HTTP服务
- 前台Service保活
