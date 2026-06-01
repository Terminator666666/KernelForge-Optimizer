#!/usr/bin/env python3
"""
Diagnosis Generator - 生成完整的性能诊断报告

整合所有分析结果，生成结构化的诊断报告，包括瓶颈、问题和优化建议。
"""

from typing import Dict, List
from datetime import datetime


def generate_diagnosis_report(
    metrics: Dict,
    memory_analysis: Dict,
    compute_analysis: Dict,
    roofline: Dict,
    bottleneck: str,
    confidence: float,
    gpu_name: str = 'Unknown'
) -> str:
    """
    生成完整的诊断报告

    参数：
        metrics: 原始 NCU 指标
        memory_analysis: 内存子系统分析结果
        compute_analysis: 计算子系统分析结果
        roofline: Roofline 模型分析结果
        bottleneck: 主要瓶颈类型
        confidence: 瓶颈置信度
        gpu_name: GPU 型号

    返回：
        str: 格式化的诊断报告
    """
    report = []

    # 报告头部
    report.append("=" * 80)
    report.append("CUDA Kernel Performance Diagnosis Report")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"GPU: {gpu_name}")
    report.append("")

    # 主要瓶颈
    report.append("PRIMARY BOTTLENECK")
    report.append("-" * 80)
    bottleneck_names = {
        'memory_bandwidth': 'Memory Bandwidth Bound',
        'memory_latency': 'Memory Latency Bound',
        'compute_bound': 'Compute Bound',
        'occupancy': 'Low Occupancy'
    }
    report.append(f"Type: {bottleneck_names.get(bottleneck, bottleneck)}")
    report.append(f"Confidence: {confidence:.1%}")
    report.append("")

    # 内存子系统分析
    report.append("MEMORY SUBSYSTEM ANALYSIS")
    report.append("-" * 80)
    report.append(f"Bandwidth Utilization: {memory_analysis.get('bandwidth_util', 0):.1f}%")
    report.append(f"Achieved Bandwidth: {memory_analysis.get('achieved_bandwidth', 0):.1f} GB/s")
    report.append(f"Access Pattern: {memory_analysis.get('access_pattern', 'unknown')}")

    if 'l1_hit_rate' in memory_analysis:
        report.append(f"L1 Cache Hit Rate: {memory_analysis['l1_hit_rate']:.1f}%")
    if 'l2_hit_rate' in memory_analysis:
        report.append(f"L2 Cache Hit Rate: {memory_analysis['l2_hit_rate']:.1f}%")
    if 'load_efficiency' in memory_analysis:
        report.append(f"Load Efficiency: {memory_analysis['load_efficiency']:.1f}%")
    if 'store_efficiency' in memory_analysis:
        report.append(f"Store Efficiency: {memory_analysis['store_efficiency']:.1f}%")
    report.append("")

    # 计算子系统分析
    report.append("COMPUTE SUBSYSTEM ANALYSIS")
    report.append("-" * 80)
    report.append(f"Achieved Occupancy: {compute_analysis.get('achieved_occupancy', 0):.1f}%")
    report.append(f"Theoretical Occupancy: {compute_analysis.get('theoretical_occupancy', 0):.1f}%")
    report.append(f"Limiting Factor: {compute_analysis.get('limiting_factor', 'unknown')}")
    report.append(f"SM Efficiency: {compute_analysis.get('sm_efficiency', 0):.1f}%")

    if 'warp_efficiency' in compute_analysis:
        report.append(f"Warp Efficiency: {compute_analysis['warp_efficiency']:.1f}%")
    if 'compute_util' in compute_analysis:
        report.append(f"Compute Utilization: {compute_analysis['compute_util']:.1f}%")
    report.append("")

    # Roofline 模型分析
    report.append("ROOFLINE MODEL ANALYSIS")
    report.append("-" * 80)
    report.append(f"Arithmetic Intensity: {roofline.get('arithmetic_intensity', 0):.2f} FLOPs/Byte")
    report.append(f"Ridge Point: {roofline.get('ridge_point', 0):.2f} FLOPs/Byte")
    report.append(f"Region: {roofline.get('region', 'unknown')}")
    report.append(f"Efficiency: {roofline.get('distance_to_peak', 0):.1%} of theoretical peak")
    report.append("")

    # 识别的问题
    report.append("IDENTIFIED ISSUES")
    report.append("-" * 80)

    issues = []
    issues.extend(memory_analysis.get('issues', []))
    issues.extend(compute_analysis.get('issues', []))

    if issues:
        for i, issue in enumerate(issues, 1):
            report.append(f"{i}. {issue}")
    else:
        report.append("No major issues identified")
    report.append("")

    # 优化建议
    report.append("OPTIMIZATION RECOMMENDATIONS")
    report.append("-" * 80)

    suggestions = get_optimization_suggestions(
        bottleneck, memory_analysis, compute_analysis, roofline
    )

    for i, suggestion in enumerate(suggestions, 1):
        report.append(f"{i}. {suggestion}")
    report.append("")

    # 报告尾部
    report.append("=" * 80)
    report.append("End of Report")
    report.append("=" * 80)

    return "\n".join(report)


def get_optimization_suggestions(
    bottleneck: str,
    memory_analysis: Dict,
    compute_analysis: Dict,
    roofline: Dict
) -> List[str]:
    """
    根据瓶颈类型和分析结果生成优化建议

    参数：
        bottleneck: 主要瓶颈类型
        memory_analysis: 内存分析结果
        compute_analysis: 计算分析结果
        roofline: Roofline 分析结果

    返回：
        List[str]: 优化建议列表
    """
    suggestions = []

    if bottleneck == 'memory_bandwidth':
        suggestions.append("使用共享内存缓存频繁访问的数据，减少全局内存访问")
        suggestions.append("提高算术强度：增加计算量，减少内存访问次数")
        suggestions.append("Kernel fusion：合并多个 kernel，减少中间数据传输")

        if memory_analysis.get('access_pattern') == 'strided':
            suggestions.append("优化数据布局：考虑从 AoS 转换为 SoA")

    elif bottleneck == 'memory_latency':
        suggestions.append("改善内存合并访问（coalescing），提高访问效率")
        suggestions.append("提高占用率以隐藏内存延迟")
        suggestions.append("使用共享内存缓存频繁访问的数据")

        if memory_analysis.get('l1_hit_rate', 100) < 50:
            suggestions.append("优化数据访问模式以提高 L1 缓存命中率")

    elif bottleneck == 'compute_bound':
        suggestions.append("使用 Tensor Cores（FP16/BF16/INT8）提升计算性能")
        suggestions.append("优化指令混合，使用 FMA 指令")
        suggestions.append("考虑算法优化以减少计算量")

        if roofline.get('distance_to_peak', 1.0) < 0.7:
            suggestions.append("提高占用率以更好地利用计算资源")

    elif bottleneck == 'occupancy':
        limiting_factor = compute_analysis.get('limiting_factor', 'unknown')

        if limiting_factor == 'registers':
            suggestions.append("减少每线程寄存器使用：减少局部变量、降低循环展开")
            suggestions.append("使用 __launch_bounds__ 限制寄存器分配")
        elif limiting_factor == 'shared_memory':
            suggestions.append("减少共享内存使用或调整 tile size")
            suggestions.append("考虑分阶段处理数据以减少共享内存需求")
        elif limiting_factor == 'block_size':
            suggestions.append("调整 block size（推荐 128-512）")

        suggestions.append("简化 kernel 逻辑以减少资源压力")

    # 通用建议
    if roofline.get('distance_to_peak', 1.0) < 0.5:
        suggestions.append(f"当前效率仅为理论峰值的 {roofline.get('distance_to_peak', 0):.1%}，有较大优化空间")

    return suggestions


def generate_summary(
    bottleneck: str,
    memory_analysis: Dict,
    compute_analysis: Dict,
    roofline: Dict
) -> str:
    """
    生成简短的诊断摘要

    参数：
        bottleneck: 主要瓶颈类型
        memory_analysis: 内存分析结果
        compute_analysis: 计算分析结果
        roofline: Roofline 分析结果

    返回：
        str: 诊断摘要
    """
    bottleneck_names = {
        'memory_bandwidth': 'Memory Bandwidth',
        'memory_latency': 'Memory Latency',
        'compute_bound': 'Compute',
        'occupancy': 'Low Occupancy'
    }

    summary_parts = [
        f"Bottleneck: {bottleneck_names.get(bottleneck, bottleneck)}",
        f"Memory: {memory_analysis.get('bandwidth_util', 0):.1f}% BW, {memory_analysis.get('access_pattern', 'unknown')} access",
        f"Compute: {compute_analysis.get('achieved_occupancy', 0):.1f}% occupancy, {compute_analysis.get('sm_efficiency', 0):.1f}% SM efficiency",
        f"Roofline: {roofline.get('region', 'unknown')}, AI={roofline.get('arithmetic_intensity', 0):.2f}, {roofline.get('distance_to_peak', 0):.1%} efficient"
    ]

    return " | ".join(summary_parts)


def main():
    """命令行接口示例"""
    # 示例数据
    metrics = {'duration': 1000000}

    memory_analysis = {
        'bandwidth_util': 85.0,
        'achieved_bandwidth': 856.8,
        'access_pattern': 'coalesced',
        'l1_hit_rate': 75.0,
        'l2_hit_rate': 68.5,
        'load_efficiency': 92.0,
        'store_efficiency': 88.0,
        'issues': ['High bandwidth utilization - memory bandwidth is likely the bottleneck']
    }

    compute_analysis = {
        'achieved_occupancy': 60.0,
        'theoretical_occupancy': 75.0,
        'limiting_factor': 'none',
        'sm_efficiency': 68.5,
        'warp_efficiency': 95.0,
        'compute_util': 30.0,
        'issues': []
    }

    roofline = {
        'arithmetic_intensity': 2.5,
        'ridge_point': 82.0,
        'region': 'memory_bound',
        'distance_to_peak': 0.65
    }

    bottleneck = 'memory_bandwidth'
    confidence = 0.9

    # 生成报告
    report = generate_diagnosis_report(
        metrics, memory_analysis, compute_analysis, roofline,
        bottleneck, confidence, 'RTX 4090'
    )

    print(report)

    print("\n\nSUMMARY:")
    print(generate_summary(bottleneck, memory_analysis, compute_analysis, roofline))


if __name__ == '__main__':
    main()
