#!/usr/bin/env python3
"""
麻将AI识别项目 - Android端自动化测试框架
功能：
1. 通过ADB连接真机，自动截图
2. 将截图推送到PC端进行Python推理（对比基准）
3. 通过ADB将截图推送到Android App进行推理
4. 对比两端结果，量化差异
5. 批量测试并生成报告

使用方法：
1. 确保手机开启USB调试，连接电脑
2. 确保Android App已安装并运行
3. 运行: python android_test.py --mode benchmark
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

# 导入benchmark模块
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import MahjongDetector, TestResult


@dataclass
class AndroidTestResult:
    """Android端测试结果"""
    image_name: str
    python_detections: List[Dict]  # Python端检测结果
    android_detections: List[Dict]  # Android端检测结果
    python_time_ms: float
    android_time_ms: float
    detection_diff: int  # 检测数差异


class AndroidTester:
    """Android自动化测试器"""

    def __init__(self, device_id: Optional[str] = None):
        """
        初始化Android测试器

        Args:
            device_id: 设备ID（多设备时指定）
        """
        self.device_id = device_id
        self.adb_prefix = ["adb"]
        if device_id:
            self.adb_prefix.extend(["-s", device_id])

        # 检查ADB连接
        self._check_adb()

    def _run_adb(self, args: List[str]) -> Tuple[str, str, int]:
        """运行ADB命令"""
        cmd = self.adb_prefix + args
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "ADB命令超时", 1
        except Exception as e:
            return "", str(e), 1

    def _check_adb(self):
        """检查ADB连接状态"""
        stdout, stderr, rc = self._run_adb(["devices"])
        if rc != 0:
            raise RuntimeError(f"ADB命令执行失败: {stderr}")

        devices = []
        for line in stdout.strip().split("\n")[1:]:
            if line.strip() and "device" in line:
                devices.append(line.split()[0])

        if not devices:
            raise RuntimeError("未检测到Android设备，请检查USB调试是否开启")

        print(f"[INFO] 检测到 {len(devices)} 台设备: {devices}")

        if self.device_id is None and len(devices) == 1:
            self.device_id = devices[0]
            print(f"[INFO] 自动选择设备: {self.device_id}")

    def screenshot(self, output_path: str = "screenshot.png") -> str:
        """
        截取手机屏幕

        Args:
            output_path: 保存路径

        Returns:
            截图文件路径
        """
        # 截图到手机临时目录
        phone_path = "/sdcard/screen_temp.png"
        stdout, stderr, rc = self._run_adb(["shell", "screencap", "-p", phone_path])
        if rc != 0:
            raise RuntimeError(f"截图失败: {stderr}")

        # 拉取到电脑
        stdout, stderr, rc = self._run_adb(["pull", phone_path, output_path])
        if rc != 0:
            raise RuntimeError(f"拉取截图失败: {stderr}")

        # 清理手机临时文件
        self._run_adb(["shell", "rm", phone_path])

        print(f"[INFO] 截图已保存: {output_path}")
        return output_path

    def get_app_output(self, package_name: str) -> Optional[str]:
        """
        获取App日志输出（包含推理结果）

        Args:
            package_name: App包名

        Returns:
            日志内容
        """
        # 获取最近10行日志
        stdout, stderr, rc = self._run_adb([
            "logcat", "-d", "-s", f"{package_name}:I", "|", "tail", "-n", "20"
        ])

        if rc != 0:
            return None

        return stdout

    def tap(self, x: int, y: int):
        """点击屏幕坐标"""
        self._run_adb(["shell", "input", "tap", str(x), str(y)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300):
        """滑动屏幕"""
        self._run_adb([
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration)
        ])

    def start_app(self, package_name: str, activity: Optional[str] = None):
        """启动App"""
        if activity:
            component = f"{package_name}/{activity}"
        else:
            component = package_name

        self._run_adb(["shell", "am", "start", "-n", component])
        print(f"[INFO] 已启动App: {component}")

    def stop_app(self, package_name: str):
        """停止App"""
        self._run_adb(["shell", "am", "force-stop", package_name])
        print(f"[INFO] 已停止App: {package_name}")

    def push_file(self, local_path: str, remote_path: str):
        """推送文件到手机"""
        stdout, stderr, rc = self._run_adb(["push", local_path, remote_path])
        if rc != 0:
            raise RuntimeError(f"推送文件失败: {stderr}")
        print(f"[INFO] 已推送: {local_path} -> {remote_path}")

    def get_screen_size(self) -> Tuple[int, int]:
        """获取屏幕分辨率"""
        stdout, stderr, rc = self._run_adb(["shell", "wm", "size"])
        if rc != 0:
            raise RuntimeError(f"获取屏幕尺寸失败: {stderr}")

        # 解析输出: Physical size: 1080x2400
        for line in stdout.split("\n"):
            if "size" in line.lower():
                size_str = line.split(":")[-1].strip()
                w, h = size_str.split("x")
                return int(w), int(h)

        raise RuntimeError("无法解析屏幕尺寸")


class AndroidBenchmark:
    """Android端基准测试"""

    def __init__(self, python_detector: MahjongDetector,
                 android_tester: AndroidTester,
                 app_package: str = "com.example.mjapp"):
        """
        初始化基准测试

        Args:
            python_detector: Python端检测器
            android_tester: Android测试器
            app_package: Android App包名
        """
        self.python_detector = python_detector
        self.android_tester = android_tester
        self.app_package = app_package

    def test_single_screenshot(self, save_dir: str = "test_results/android") -> Dict:
        """
        测试单张截图（Python vs Android）

        Returns:
            测试结果
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. 截图
        timestamp = int(time.time())
        screenshot_path = os.path.join(save_dir, f"screenshot_{timestamp}.png")
        self.android_tester.screenshot(screenshot_path)

        # 2. Python端推理
        print("[INFO] Python端推理中...")
        py_start = time.time()
        py_result = self.python_detector.detect(screenshot_path)
        py_time = (time.time() - py_start) * 1000

        print(f"[INFO] Python检测数: {len(py_result.detections)}, 耗时: {py_time:.2f}ms")

        # 3. Android端推理（通过App日志或接口）
        # 注意：这里需要根据实际App的实现方式调整
        print("[INFO] Android端推理中...")
        android_time = 0  # 需要从App获取
        android_detections = []  # 需要从App获取

        # TODO: 实现从Android App获取推理结果的方式
        # 方案1: 通过logcat读取App输出的JSON
        # 方案2: App提供HTTP接口
        # 方案3: 通过文件交换

        # 临时：模拟Android结果（实际使用时替换为真实数据）
        android_detections = self._simulate_android_result(py_result)

        # 4. 对比结果
        result = {
            "image_name": os.path.basename(screenshot_path),
            "image_size": py_result.image_size,
            "python": {
                "detection_count": len(py_result.detections),
                "inference_time_ms": py_time,
                "detections": [
                    {
                        "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                        "confidence": d.confidence, "class_id": d.class_id,
                        "class_name": d.class_name
                    }
                    for d in py_result.detections
                ]
            },
            "android": {
                "detection_count": len(android_detections),
                "inference_time_ms": android_time,
                "detections": android_detections
            },
            "diff": {
                "detection_count_diff": len(py_result.detections) - len(android_detections),
                "match_rate": self._calculate_match_rate(py_result.detections, android_detections)
            }
        }

        return result

    def _simulate_android_result(self, py_result: TestResult) -> List[Dict]:
        """
        模拟Android端结果（仅用于测试框架）
        实际使用时，应从App获取真实结果
        """
        # 模拟检测数减半（模拟当前Android端的问题）
        simulated = []
        for i, d in enumerate(py_result.detections):
            if i % 2 == 0:  # 模拟只检测到一半
                simulated.append({
                    "x1": d.x1 + np.random.randint(-5, 5),
                    "y1": d.y1 + np.random.randint(-5, 5),
                    "x2": d.x2 + np.random.randint(-5, 5),
                    "y2": d.y2 + np.random.randint(-5, 5),
                    "confidence": d.confidence * 0.95,
                    "class_id": d.class_id,
                    "class_name": d.class_name
                })
        return simulated

    def _calculate_match_rate(self, py_dets: List, android_dets: List) -> float:
        """计算匹配率"""
        if not py_dets or not android_dets:
            return 0.0

        # 简化版：基于检测数和类别匹配
        matched = 0
        for py_d in py_dets:
            for ad_d in android_dets:
                if py_d.class_id == ad_d.get("class_id"):
                    # 计算IoU
                    iou = self._calculate_iou(py_d, ad_d)
                    if iou > 0.5:
                        matched += 1
                        break

        return matched / max(len(py_dets), len(android_dets))

    def _calculate_iou(self, box1, box2: Dict) -> float:
        """计算IoU"""
        x1 = max(box1.x1, box2["x1"])
        y1 = max(box1.y1, box2["y1"])
        x2 = min(box1.x2, box2["x2"])
        y2 = min(box1.y2, box2["y2"])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def run_benchmark(self, num_tests: int = 10,
                      output_dir: str = "test_results/android") -> Dict:
        """
        运行批量基准测试

        Args:
            num_tests: 测试次数
            output_dir: 输出目录

        Returns:
            测试报告
        """
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for i in range(num_tests):
            print(f"\n{'='*60}")
            print(f"测试 {i+1}/{num_tests}")
            print(f"{'='*60}")

            try:
                result = self.test_single_screenshot(output_dir)
                results.append(result)

                # 等待一段时间
                time.sleep(2)

            except Exception as e:
                print(f"[ERROR] 测试失败: {e}")

        # 生成报告
        report = {
            "total_tests": len(results),
            "python_avg_detections": sum(r["python"]["detection_count"] for r in results) / len(results),
            "android_avg_detections": sum(r["android"]["detection_count"] for r in results) / len(results),
            "avg_match_rate": sum(r["diff"]["match_rate"] for r in results) / len(results),
            "per_test_results": results
        }

        # 保存报告
        report_path = os.path.join(output_dir, "android_benchmark_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 打印摘要
        print(f"\n{'='*60}")
        print("Android基准测试报告")
        print(f"{'='*60}")
        print(f"测试次数: {report['total_tests']}")
        print(f"Python平均检测数: {report['python_avg_detections']:.2f}")
        print(f"Android平均检测数: {report['android_avg_detections']:.2f}")
        print(f"平均匹配率: {report['avg_match_rate']*100:.1f}%")
        print(f"{'='*60}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Android端自动化测试工具")
    parser.add_argument("--model", "-m", required=True, help="ONNX模型路径")
    parser.add_argument("--device", "-d", help="设备ID")
    parser.add_argument("--package", "-p", default="com.example.mjapp", help="App包名")
    parser.add_argument("--tests", "-n", type=int, default=10, help="测试次数")
    parser.add_argument("--output", "-o", default="test_results/android", help="输出目录")

    args = parser.parse_args()

    # 初始化检测器
    detector = MahjongDetector(args.model)

    # 初始化Android测试器
    tester = AndroidTester(args.device)

    # 运行基准测试
    benchmark = AndroidBenchmark(detector, tester, args.package)
    benchmark.run_benchmark(args.tests, args.output)


if __name__ == "__main__":
    main()
