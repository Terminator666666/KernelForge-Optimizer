#!/usr/bin/env python3
"""
Bottleneck Identifier - 识别 CUDA kernel 的主要性能瓶颈

基于内存、计算和 Roofline 分析结果，自动识别主要瓶颈类型并给出置信度。
"""

from typing import Dict, Tuple


def identify_bottleneck(
    memory_analysis: Dict,
    compute_analysis: Dict,
    roofline: Dict
) -> Tuple[str, float]:
    """
    识别主要瓶颈及置信度

    参数：
        memory_analysis: 内存子系统分析结果
        compute_analysis: 计算子系统分析结果
        roofline: Roofline 模型分析结果

    返回：
        Tuple[str, float]: (瓶颈类型, 置信度)
            瓶颈类型: 'memory_bandwidth', 'memory_latency', 'compute_bound', 'occupancy'
            置信度: 0.0 到 1.0
    """
    bw_util = memory_analysis.get('bandwidth_util', 0)
    compute_util = compute_analysis.get('compute_util', 0)
    occupancy = compute_analysis.get('achieved_occupancy', 0)
    region = roofline.get('region', 'balanced')

    # 初始化各瓶颈得分
    scores = {
        'memory_bandwidth': 0.0,
        'memory_latency': 0.0,
        'compute_bound': 0.0,
        'occupancy': 0.0
    }

    # === Memory Bandwidth 瓶颈指标 ===
    if bw_util > 70:
        scores['memory_bandwidth'] += 0.4  # 高带宽利用率
    if region == 'memory_bound':
        scores['memory_bandwidth'] += 0.3  # Roofline 显示 memory-bound
    if memory_analysis.get('access_pattern') in ['coalesced', 'strided']:
        scores['memory_bandwidth'] += 0.2  # 访问模式良好

    # === Memory Latency 瓶颈指标 ===
    if bw_util < 40 and occupancy < 50:
        scores['memory_latency'] += 0.4  # 低带宽 + 低占用率
    if memory_analysis.get('access_pattern') in ['random', 'mixed']:
        scores['memory_latency'] += 0.3  # 访问模式差
    l1_hit = memory_analysis.get('l1_hit_rate', 100)
    if l1_hit < 50:
        scores['memory_latency'] += 0.2  # 低缓存命中率

    # === Compute Bound 瓶颈指标 ===
    if compute_util > 70:
        scores['compute_bound'] += 0.4  # 高计算利用率
    if region == 'compute_bound':
        scores['compute_bound'] += 0.3  # Roofline 显示 compute-bound
    if roofline.get('arithmetic_intensity', 0) > roofline.get('ridge_point', 100):
        scores['compute_bound'] += 0.2  # 算术强度超过 ridge point

    # === Occupancy 瓶颈指标 ===
    if occupancy < 30:
        scores['occupancy'] += 0.5  # 极低占用率
    if compute_analysis.get('limiting_factor') in ['registers', 'shared_memory']:
        scores['occupancy'] += 0.3  # 明确的资源限制

    # 找出得分最高的瓶颈
    bottleneck = max(scores, key=scores.get)
    confidence = scores[bottleneck]

    # 归一化置信度到 0-1 范围
    confidence = min(confidence, 1.0)

    return bottleneck, confidence


def get_bottleneck_description(bottleneck: str, confidence: float) -> str:
    """
    获取瓶颈的描述信息

    参数：
        bottleneck: 瓶颈类型
        confidence: 置信度

    返回：
        str: 瓶颈描述
    """
    descriptions = {
        'memory_bandwidth': 'Memory Bandwidth Bound - 内存带宽饱和，数据传输速度限制性能',
        'memory_latency': 'Memory Latency Bound - 内存访问延迟高，等待数据时间长',
        'compute_bound': 'Compute Bound - 计算单元饱和，计算能力限制性能',
        'occupancy': 'Low Occupancy - SM 占用率过低，并行度不足限制性能'
    }

    desc = descriptions.get(bottleneck, 'Unknown bottleneck')
    confidence_level = 'high' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'low'

    return f"{desc} (confidence: {confidence:.1%}, level: {confidence_level})"


def get_optimization_suggestions(bottleneck: str) -> list:
    """
    根据瓶颈类型获取优化建议

    参数：
        bottleneck: 瓶颈类型

    返回：
        list: 优化建议列表
    """
    suggestions = {
        'memory_bandwidth': [
            '使用共享内存缓存频繁访问的数据',
            '提高算术强度，增加计算量减少内存访问',
            'Kernel fusion，合并多个 kernel 减少中间数据传输',
            '使用低精度数据类型（FP16, INT8）减少带宽需求'
        ],
        'memory_latency': [
            '改善内存合并访问（coalescing）',
            '提高占用率以隐藏内存延迟',
            '使用共享内存缓存频繁访问的数据',
            '重组数据布局提高局部性'
        ],
        'compute_bound': [
            '使用 Tensor Cores（FP16/BF16/INT8）',
            '优化指令混合，使用 FMA 指令',
            '考虑算法优化减少计算量',
            '减少寄存器压力提高占用率'
        ],
        'occupancy': [
            '减少每线程寄存器使用',
            '使用 __launch_bounds__ 限制寄存器分配',
            '减少共享内存使用或调整 tile size',
            '调整 block size（通常 128-512）'
        ]
    }

    return suggestions.get(bottleneck, ['No specific suggestions available'])


def main():
    """命令行接口示例"""
    # 示例数据
    memory_analysis = {
        'bandwidth_util': 85.0,
        'access_pattern': 'coalesced',
        'l1_hit_rate': 75.0
    }

    compute_analysis = {
        'achieved_occupancy': 60.0,
        'compute_util': 30.0,
        'limiting_factor': 'none'
    }

    roofline = {
        'region': 'memory_bound',
        'arithmetic_intensity': 2.5,
        'ridge_point': 82.0
    }

    # 识别瓶颈
    bottleneck, confidence = identify_bottleneck(
        memory_analysis, compute_analysis, roofline
    )

    # 输出结果
    print("Bottleneck Identification Result:")
    print(f"  Type: {bottleneck}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Description: {get_bottleneck_description(bottleneck, confidence)}")
    print("\nOptimization Suggestions:")
    for i, suggestion in enumerate(get_optimization_suggestions(bottleneck), 1):
        print(f"  {i}. {suggestion}")


if __name__ == '__main__':
    main()
