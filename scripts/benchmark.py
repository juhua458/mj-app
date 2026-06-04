#!/usr/bin/env python3
"""
麻将AI识别项目 - Python端自动化测试框架
功能：
1. 批量推理测试图片
2. 统计检测指标（检测数、置信度分布、类别分布）
3. 对比不同模型/配置的输出差异
4. 生成测试报告
"""

import os
import sys
import time
import json
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

# ONNX Runtime
import onnxruntime as ort


@dataclass
class Detection:
    """单个检测结果"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""


@dataclass
class TestResult:
    """单张图片的测试结果"""
    image_name: str
    image_size: Tuple[int, int]
    detections: List[Detection]
    inference_time_ms: float
    preprocess_time_ms: float
    postprocess_time_ms: float


@dataclass
class BenchmarkReport:
    """批量测试报告"""
    model_path: str
    model_info: Dict
    total_images: int
    total_detections: int
    avg_detections_per_image: float
    avg_inference_time_ms: float
    confidence_distribution: Dict[str, int]
    class_distribution: Dict[str, int]
    per_image_results: List[Dict]


class MahjongDetector:
    """麻将牌面检测器"""

    # 麻将牌类别映射（根据实际模型调整）
    CLASS_NAMES = {
        0: "1wan", 1: "2wan", 2: "3wan", 3: "4wan", 4: "5wan",
        5: "6wan", 6: "7wan", 7: "8wan", 8: "9wan",
        9: "1tong", 10: "2tong", 11: "3tong", 12: "4tong", 13: "5tong",
        14: "6tong", 15: "7tong", 16: "8tong", 17: "9tong",
        18: "1tiao", 19: "2tiao", 20: "3tiao", 21: "4tiao", 22: "5tiao",
        23: "6tiao", 24: "7tiao", 25: "8tiao", 26: "9tiao",
        27: "dong", 28: "nan", 29: "xi", 30: "bei",
        31: "zhong", 32: "fa", 33: "bai",
        34: "chun", 35: "xia", 36: "qiu", 37: "dong_ji",
        38: "mei", 39: "lan", 40: "zhu", 41: "ju",
        # 根据实际模型类别数调整
    }

    def __init__(self, model_path: str, input_size: int = 640,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 providers: Optional[List[str]] = None):
        """
        初始化检测器

        Args:
            model_path: ONNX模型路径
            input_size: 模型输入尺寸
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            providers: ONNX Runtime执行提供器
        """
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 加载模型
        if providers is None:
            providers = ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 获取模型信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        print(f"[INFO] 模型加载成功: {model_path}")
        print(f"[INFO] 输入形状: {self.input_shape}")
        print(f"[INFO] 输出形状: {self.output_shape}")
        print(f"[INFO] 执行提供器: {providers}")

    def get_model_info(self) -> Dict:
        """获取模型元数据信息"""
        # 尝试读取opset和IR版本
        import onnx
        model = onnx.load(self.model_path)
        opset_version = model.opset_import[0].version if model.opset_import else "unknown"
        ir_version = model.ir_version

        return {
            "model_path": self.model_path,
            "input_shape": str(self.input_shape),
            "output_shape": str(self.output_shape),
            "opset_version": opset_version,
            "ir_version": ir_version,
            "input_size": self.input_size,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
        """
        预处理图像

        Returns:
            preprocessed_image: 预处理后的图像
            scale: 缩放比例
            pad_x, pad_y: 填充偏移
            orig_h, orig_w: 原始尺寸
        """
        start_time = time.time()

        orig_h, orig_w = image.shape[:2]

        # 计算缩放和填充（保持宽高比）
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建填充画布
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)

        # 计算填充偏移
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2

        # 放置缩放后的图像
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # 归一化并调整维度 (H,W,C) -> (C,H,W)
        preprocessed = padded.astype(np.float32) / 255.0
        preprocessed = np.transpose(preprocessed, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)

        preprocess_time = (time.time() - start_time) * 1000

        return preprocessed, scale, pad_x, pad_y, orig_h, orig_w, preprocess_time

    def postprocess(self, outputs: np.ndarray, scale: float,
                    pad_x: float, pad_y: float, orig_h: int, orig_w: int) -> List[Detection]:
        """
        后处理ONNX输出

        Args:
            outputs: 模型原始输出
            scale: 缩放比例
            pad_x, pad_y: 填充偏移
            orig_h, orig_w: 原始图像尺寸

        Returns:
            detections: 检测框列表
        """
        start_time = time.time()

        # YOLOv8输出格式: [batch, 4+num_classes, num_anchors] 或 [batch, num_anchors, 4+num_classes]
        # 需要根据实际模型调整

        # 假设输出是 [1, 39, 8400] 格式 (39 = 4 box + 35 classes)
        if outputs.shape[1] > outputs.shape[2]:
            # [batch, features, anchors] -> [batch, anchors, features]
            outputs = np.transpose(outputs, (0, 2, 1))

        # 解析输出
        batch_size, num_anchors, num_features = outputs.shape
        num_classes = num_features - 4

        detections = []

        for i in range(num_anchors):
            row = outputs[0, i, :]

            # 提取box和置信度
            x, y, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]

            # 获取最高置信度类别
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            # 转换为xyxy格式（在640x640空间）
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            # 反变换到原始图像坐标
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # 裁剪到图像边界
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            class_name = self.CLASS_NAMES.get(class_id, f"class_{class_id}")

            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            ))

        # 应用NMS（简化版）
        if len(detections) > 0:
            detections = self._nms(detections)

        postprocess_time = (time.time() - start_time) * 1000

        return detections, postprocess_time

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """非极大值抑制"""
        if not detections:
            return detections

        # 按置信度排序
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections[0]
            keep.append(best)

            # 计算与其他框的IoU
            remaining = []
            for det in detections[1:]:
                iou = self._iou(best, det)
                if iou < self.iou_threshold:
                    remaining.append(det)

            detections = remaining

        return keep

    def _iou(self, box1: Detection, box2: Detection) -> float:
        """计算两个框的IoU"""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def detect(self, image_path: str) -> TestResult:
        """对单张图片进行检测"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # 预处理
        preprocessed, scale, pad_x, pad_y, oh, ow, preprocess_time = self.preprocess(image)

        # 推理
        infer_start = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: preprocessed})
        inference_time = (time.time() - infer_start) * 1000

        # 后处理
        detections, postprocess_time = self.postprocess(
            outputs[0], scale, pad_x, pad_y, oh, ow
        )

        return TestResult(
            image_name=os.path.basename(image_path),
            image_size=(orig_w, orig_h),
            detections=detections,
            inference_time_ms=inference_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time
        )

    def benchmark(self, image_dir: str, output_dir: str = "test_results") -> BenchmarkReport:
        """批量测试并生成报告"""
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))

        if not image_paths:
            print(f"[WARNING] 在 {image_dir} 中未找到图片")
            return None

        print(f"[INFO] 找到 {len(image_paths)} 张测试图片")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        results = []
        all_detections = []
        total_inference_time = 0

        for i, img_path in enumerate(image_paths):
            print(f"\r[INFO] 处理中: {i+1}/{len(image_paths)} - {os.path.basename(img_path)}", end="")

            try:
                result = self.detect(img_path)
                results.append(result)
                all_detections.extend(result.detections)
                total_inference_time += result.inference_time_ms
            except Exception as e:
                print(f"\n[ERROR] 处理 {img_path} 失败: {e}")

        print(f"\n[INFO] 测试完成")

        # 统计置信度分布
        conf_ranges = {
            "0.90-1.00": 0, "0.80-0.90": 0, "0.70-0.80": 0,
            "0.60-0.70": 0, "0.50-0.60": 0, "0.40-0.50": 0,
            "0.30-0.40": 0, "0.25-0.30": 0, "<0.25": 0
        }

        for det in all_detections:
            conf = det.confidence
            if conf >= 0.90:
                conf_ranges["0.90-1.00"] += 1
            elif conf >= 0.80:
                conf_ranges["0.80-0.90"] += 1
            elif conf >= 0.70:
                conf_ranges["0.70-0.80"] += 1
            elif conf >= 0.60:
                conf_ranges["0.60-0.70"] += 1
            elif conf >= 0.50:
                conf_ranges["0.50-0.60"] += 1
            elif conf >= 0.40:
                conf_ranges["0.40-0.50"] += 1
            elif conf >= 0.30:
                conf_ranges["0.30-0.40"] += 1
            elif conf >= 0.25:
                conf_ranges["0.25-0.30"] += 1
            else:
                conf_ranges["<0.25"] += 1

        # 统计类别分布
        class_dist = {}
        for det in all_detections:
            name = det.class_name
            class_dist[name] = class_dist.get(name, 0) + 1

        # 生成报告
        report = BenchmarkReport(
            model_path=self.model_path,
            model_info=self.get_model_info(),
            total_images=len(results),
            total_detections=len(all_detections),
            avg_detections_per_image=len(all_detections) / len(results) if results else 0,
            avg_inference_time_ms=total_inference_time / len(results) if results else 0,
            confidence_distribution=conf_ranges,
            class_distribution=class_dist,
            per_image_results=[{
                "image_name": r.image_name,
                "image_size": r.image_size,
                "detection_count": len(r.detections),
                "inference_time_ms": round(r.inference_time_ms, 2),
                "preprocess_time_ms": round(r.preprocess_time_ms, 2),
                "postprocess_time_ms": round(r.postprocess_time_ms, 2),
                "detections": [
                    {
                        "x1": round(d.x1, 2), "y1": round(d.y1, 2),
                        "x2": round(d.x2, 2), "y2": round(d.y2, 2),
                        "confidence": round(d.confidence, 4),
                        "class_id": d.class_id,
                        "class_name": d.class_name
                    }
                    for d in r.detections
                ]
            } for r in results]
        )

        # 保存报告
        report_path = os.path.join(output_dir, "benchmark_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

        # 打印摘要
        self._print_summary(report)

        return report

    def _print_summary(self, report: BenchmarkReport):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("基准测试报告摘要")
        print("=" * 60)
        print(f"模型路径: {report.model_path}")
        print(f"测试图片数: {report.total_images}")
        print(f"总检测数: {report.total_detections}")
        print(f"平均每图检测数: {report.avg_detections_per_image:.2f}")
        print(f"平均推理时间: {report.avg_inference_time_ms:.2f} ms")
        print("\n置信度分布:")
        for range_name, count in report.confidence_distribution.items():
            if count > 0:
                print(f"  {range_name}: {count}")
        print("\n类别分布 (Top 10):")
        sorted_classes = sorted(report.class_distribution.items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_classes[:10]:
            print(f"  {name}: {count}")
        print("=" * 60)


def compare_models(model_a_path: str, model_b_path: str, image_dir: str,
                   output_dir: str = "test_results"):
    """对比两个模型的输出差异"""
    print(f"\n{'='*60}")
    print("模型对比测试")
    print(f"{'='*60}")

    detector_a = MahjongDetector(model_a_path)
    detector_b = MahjongDetector(model_b_path)

    report_a = detector_a.benchmark(image_dir, os.path.join(output_dir, "model_a"))
    report_b = detector_b.benchmark(image_dir, os.path.join(output_dir, "model_b"))

    # 对比摘要
    print(f"\n{'='*60}")
    print("对比结果")
    print(f"{'='*60}")
    print(f"{'指标':<30} {'模型A':>15} {'模型B':>15}")
    print("-" * 60)
    print(f"{'平均每图检测数':<30} {report_a.avg_detections_per_image:>15.2f} {report_b.avg_detections_per_image:>15.2f}")
    print(f"{'平均推理时间(ms)':<30} {report_a.avg_inference_time_ms:>15.2f} {report_b.avg_inference_time_ms:>15.2f}")
    print(f"{'总检测数':<30} {report_a.total_detections:>15} {report_b.total_detections:>15}")

    # 逐图对比
    diff_count = 0
    for ra, rb in zip(report_a.per_image_results, report_b.per_image_results):
        if ra["detection_count"] != rb["detection_count"]:
            diff_count += 1
            print(f"\n[DIFF] {ra['image_name']}: A={ra['detection_count']}, B={rb['detection_count']}")

    if diff_count == 0:
        print("\n[OK] 所有图片的检测数完全一致！")
    else:
        print(f"\n[WARNING] {diff_count}/{len(report_a.per_image_results)} 张图片检测数不一致")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="麻将AI识别基准测试工具")
    parser.add_argument("--model", "-m", required=True, help="ONNX模型路径")
    parser.add_argument("--images", "-i", required=True, help="测试图片目录")
    parser.add_argument("--output", "-o", default="test_results", help="输出目录")
    parser.add_argument("--compare", "-c", help="对比模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--input-size", type=int, default=640, help="输入尺寸")

    args = parser.parse_args()

    if args.compare:
        compare_models(args.model, args.compare, args.images, args.output)
    else:
        detector = MahjongDetector(
            args.model,
            input_size=args.input_size,
            conf_threshold=args.conf
        )
        detector.benchmark(args.images, args.output)


if __name__ == "__main__":
    # 导入cv2（放在这里避免全局导入问题）
    import cv2
    main()
