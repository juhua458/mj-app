#!/usr/bin/env python3
"""
ONNX模型opset降级工具
功能：
1. 将模型opset从20降级到17（解决Android ONNX Runtime兼容性问题）
2. 保持IR版本不变
3. 验证降级后的模型输出一致性
4. 生成模型信息对比报告
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import onnx
from onnx import version_converter, helper
import onnxruntime as ort


def get_model_info(model_path: str) -> Dict:
    """获取ONNX模型信息"""
    model = onnx.load(model_path)

    # 获取opset版本
    opset_version = "unknown"
    if model.opset_import:
        opset_version = model.opset_import[0].version

    # 获取IR版本
    ir_version = model.ir_version

    # 获取模型大小
    model_size = os.path.getsize(model_path)

    # 获取输入输出信息
    session = ort.InferenceSession(model_path)
    input_info = {
        "name": session.get_inputs()[0].name,
        "shape": str(session.get_inputs()[0].shape)
    }
    output_info = {
        "name": session.get_outputs()[0].name,
        "shape": str(session.get_outputs()[0].shape)
    }

    # 统计算子数量
    op_types = set()
    for node in model.graph.node:
        op_types.add(node.op_type)

    return {
        "model_path": model_path,
        "file_name": os.path.basename(model_path),
        "file_size_bytes": model_size,
        "file_size_mb": round(model_size / (1024 * 1024), 2),
        "opset_version": opset_version,
        "ir_version": ir_version,
        "input_info": input_info,
        "output_info": output_info,
        "operator_count": len(model.graph.node),
        "unique_operators": sorted(list(op_types)),
    }


def print_model_info(info: Dict, title: str = "模型信息"):
    """打印模型信息"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"文件路径: {info['model_path']}")
    print(f"文件大小: {info['file_size_bytes']:,} bytes ({info['file_size_mb']} MB)")
    print(f"opset版本: {info['opset_version']}")
    print(f"IR版本: {info['ir_version']}")
    print(f"算子总数: {info['operator_count']}")
    print(f"输入: {info['input_info']}")
    print(f"输出: {info['output_info']}")
    print(f"{'='*60}")


def convert_opset(model_path: str, target_opset: int = 17,
                  target_ir: Optional[int] = None) -> str:
    """
    转换模型opset版本

    Args:
        model_path: 输入模型路径
        target_opset: 目标opset版本（默认17）
        target_ir: 目标IR版本（默认保持原IR版本）

    Returns:
        输出模型路径
    """
    print(f"\n[INFO] 开始转换模型...")
    print(f"[INFO] 输入模型: {model_path}")
    print(f"[INFO] 目标opset: {target_opset}")

    # 加载模型
    model = onnx.load(model_path)
    original_opset = model.opset_import[0].version if model.opset_import else "unknown"
    original_ir = model.ir_version

    if target_ir is None:
        target_ir = original_ir

    print(f"[INFO] 原始opset: {original_opset}")
    print(f"[INFO] 原始IR版本: {original_ir}")
    print(f"[INFO] 目标IR版本: {target_ir}")

    # 使用onnx版本转换器
    try:
        converted_model = version_converter.convert_version(model, target_opset)
        print(f"[INFO] opset转换成功: {original_opset} -> {target_opset}")
    except Exception as e:
        print(f"[ERROR] opset转换失败: {e}")
        print(f"[INFO] 尝试手动修改opset_import...")

        # 手动修改opset_import（风险较高，仅作为fallback）
        converted_model = model
        if converted_model.opset_import:
            converted_model.opset_import[0].version = target_opset
        else:
            # 添加opset_import
            opset = helper.make_opsetid("", target_opset)
            converted_model.opset_import.append(opset)

    # 设置IR版本
    converted_model.ir_version = target_ir
    print(f"[INFO] IR版本设置: {target_ir}")

    # 验证模型
    try:
        onnx.checker.check_model(converted_model)
        print(f"[INFO] 模型验证通过 ✓")
    except Exception as e:
        print(f"[WARNING] 模型验证警告: {e}")

    # 生成输出路径
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.dirname(model_path)
    output_path = os.path.join(output_dir, f"{base_name}_opset{target_opset}.onnx")

    # 保存模型
    onnx.save(converted_model, output_path)
    print(f"[INFO] 模型已保存: {output_path}")

    return output_path


def verify_models(model_a_path: str, model_b_path: str,
                  test_input: Optional[np.ndarray] = None) -> Dict:
    """
    验证两个模型的输出一致性

    Args:
        model_a_path: 模型A路径（原始）
        model_b_path: 模型B路径（转换后）
        test_input: 测试输入（默认随机生成）

    Returns:
        验证结果字典
    """
    print(f"\n{'='*60}")
    print("模型输出一致性验证")
    print(f"{'='*60}")

    # 创建会话
    sess_a = ort.InferenceSession(model_a_path)
    sess_b = ort.InferenceSession(model_b_path)

    input_name_a = sess_a.get_inputs()[0].name
    input_name_b = sess_b.get_inputs()[0].name
    input_shape = sess_a.get_inputs()[0].shape

    # 生成测试输入
    if test_input is None:
        # 使用随机输入
        batch_size = 1
        channels = 3
        height = 640
        width = 640

        # 处理动态维度
        if isinstance(input_shape[0], str):
            batch_size = 1
        else:
            batch_size = input_shape[0]

        if isinstance(input_shape[1], str):
            channels = 3
        else:
            channels = input_shape[1]

        if isinstance(input_shape[2], str):
            height = 640
        else:
            height = input_shape[2]

        if isinstance(input_shape[3], str):
            width = 640
        else:
            width = input_shape[3]

        test_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    print(f"[INFO] 测试输入形状: {test_input.shape}")

    # 运行推理
    output_a = sess_a.run(None, {input_name_a: test_input})[0]
    output_b = sess_b.run(None, {input_name_b: test_input})[0]

    print(f"[INFO] 模型A输出形状: {output_a.shape}")
    print(f"[INFO] 模型B输出形状: {output_b.shape}")

    # 比较输出
    if output_a.shape != output_b.shape:
        print(f"[ERROR] 输出形状不一致!")
        return {
            "shape_match": False,
            "max_diff": None,
            "mean_diff": None,
            "is_identical": False
        }

    diff = np.abs(output_a - output_b)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    is_identical = max_diff < 1e-6

    print(f"[INFO] 最大绝对差异: {max_diff:.10f}")
    print(f"[INFO] 平均绝对差异: {mean_diff:.10f}")

    if is_identical:
        print(f"[INFO] 输出完全一致 ✓")
    else:
        print(f"[WARNING] 输出存在差异（可能是数值精度问题）")

    return {
        "shape_match": True,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "is_identical": is_identical,
        "output_a_shape": list(output_a.shape),
        "output_b_shape": list(output_b.shape),
    }


def compare_model_files(model_a_path: str, model_b_path: str) -> Dict:
    """对比两个模型文件的信息"""
    info_a = get_model_info(model_a_path)
    info_b = get_model_info(model_b_path)

    print(f"\n{'='*60}")
    print("模型对比")
    print(f"{'='*60}")
    print(f"{'属性':<20} {'原始模型':>20} {'转换后模型':>20}")
    print("-" * 60)
    print(f"{'文件名':<20} {info_a['file_name']:>20} {info_b['file_name']:>20}")
    print(f"{'文件大小(MB)':<20} {info_a['file_size_mb']:>20.2f} {info_b['file_size_mb']:>20.2f}")
    print(f"{'opset版本':<20} {str(info_a['opset_version']):>20} {str(info_b['opset_version']):>20}")
    print(f"{'IR版本':<20} {info_a['ir_version']:>20} {info_b['ir_version']:>20}")
    print(f"{'算子数':<20} {info_a['operator_count']:>20} {info_b['operator_count']:>20}")
    print(f"{'='*60}")

    return {
        "original": info_a,
        "converted": info_b,
        "size_diff_bytes": info_b["file_size_bytes"] - info_a["file_size_bytes"],
        "opset_changed": info_a["opset_version"] != info_b["opset_version"],
    }


def main():
    parser = argparse.ArgumentParser(description="ONNX模型opset降级工具")
    parser.add_argument("--model", "-m", required=True, help="输入ONNX模型路径")
    parser.add_argument("--output", "-o", help="输出模型路径（默认自动生成）")
    parser.add_argument("--opset", type=int, default=17, help="目标opset版本（默认17）")
    parser.add_argument("--ir", type=int, help="目标IR版本（默认保持原IR版本）")
    parser.add_argument("--verify", action="store_true", help="验证输出一致性")
    parser.add_argument("--report", "-r", help="输出对比报告JSON路径")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.model):
        print(f"[ERROR] 模型文件不存在: {args.model}")
        sys.exit(1)

    # 获取原始模型信息
    original_info = get_model_info(args.model)
    print_model_info(original_info, "原始模型信息")

    # 转换模型
    output_path = convert_opset(args.model, args.opset, args.ir)

    # 获取转换后模型信息
    converted_info = get_model_info(output_path)
    print_model_info(converted_info, "转换后模型信息")

    # 对比模型
    comparison = compare_model_files(args.model, output_path)

    # 验证输出一致性
    verification = None
    if args.verify:
        verification = verify_models(args.model, output_path)

    # 生成报告
    report = {
        "original_model": original_info,
        "converted_model": converted_info,
        "comparison": comparison,
        "verification": verification,
    }

    # 保存报告
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 报告已保存: {args.report}")

    print(f"\n[INFO] 转换完成!")
    print(f"[INFO] 输出模型: {output_path}")


if __name__ == "__main__":
    main()
