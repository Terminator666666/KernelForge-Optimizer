#!/usr/bin/env python3
"""
策略推荐工具
基于历史数据和当前诊断推荐下一步优化策略
"""

from typing import Dict, List, Optional
from analyze_trends import get_recent_trend, detect_bottleneck_shift, analyze_strategy_effectiveness


# 策略优先级（基于瓶颈类型）
STRATEGY_PRIORITY = {
    'memory_bandwidth': ['vectorized_memory', 'kernel_fusion', 'matmul_tiling'],
    'memory_latency': ['matmul_tiling', 'reduction_warp_primitives', 'bank_conflict_free'],
    'compute_bound': ['tensor_core', 'occupancy_tuning'],
    'occupancy': ['occupancy_tuning', 'reduction_warp_primitives', 'cooperative_groups']
}


def recommend_next_strategy(rounds: List, current_diagnosis: Dict) -> Dict:
    """
    推荐下一步优化策略

    参数:
        rounds: 历史轮次列表
        current_diagnosis: 当前诊断结果

    返回:
        {
            'strategy': 'kernel_fusion',
            'reason': '性能停滞，尝试不同类型的策略',
            'expected_speedup': '2-4x',
            'priority': 'high'
        }
    """
    bottleneck = current_diagnosis.get('bottleneck', 'unknown')

    # 如果没有历史，直接推荐基于瓶颈的策略
    if not rounds:
        return _recommend_by_bottleneck(bottleneck)

    # 获取最近使用的策略
    recent_strategies = [r.strategy for r in rounds[-3:]]

    # 分析趋势
    trend = get_recent_trend(rounds, window=3)

    # 检测瓶颈转移
    shift = detect_bottleneck_shift(rounds)

    # 根据趋势推荐
    if trend == 'improving':
        # 性能持续提升，继续当前方向
        return _recommend_similar_strategy(recent_strategies[-1], recent_strategies, bottleneck)

    elif trend == 'stagnant':
        # 性能停滞，尝试不同类型的策略
        return _recommend_different_approach(bottleneck, recent_strategies)

    elif trend == 'degrading':
        # 性能下降，回退到之前有效的策略
        return _recommend_previous_best(rounds, bottleneck)

    else:  # unstable
        # 波动大，尝试更稳定的策略
        return _recommend_stable_strategy(bottleneck, recent_strategies)


def _recommend_by_bottleneck(bottleneck: str) -> Dict:
    """基于瓶颈类型推荐"""
    strategies = STRATEGY_PRIORITY.get(bottleneck, ['occupancy_tuning'])

    return {
        'strategy': strategies[0],
        'reason': f'基于瓶颈类型 {bottleneck} 的首选策略',
        'expected_speedup': _get_expected_speedup(strategies[0]),
        'priority': 'high'
    }


def _recommend_similar_strategy(last_strategy: str, recent_strategies: List, bottleneck: str) -> Dict:
    """推荐类似的策略（性能提升时）"""
    # 策略组合建议
    combinations = {
        'matmul_tiling': ['vectorized_memory', 'tensor_core'],
        'vectorized_memory': ['kernel_fusion'],
        'kernel_fusion': ['vectorized_memory'],
        'reduction_warp_primitives': ['occupancy_tuning']
    }

    next_strategies = combinations.get(last_strategy, [])

    # 过滤已使用的策略
    for strategy in next_strategies:
        if strategy not in recent_strategies:
            return {
                'strategy': strategy,
                'reason': f'与 {last_strategy} 组合使用，进一步提升性能',
                'expected_speedup': _get_expected_speedup(strategy),
                'priority': 'high'
            }

    # 如果没有组合策略，推荐基于瓶颈的策略
    return _recommend_by_bottleneck(bottleneck)


def _recommend_different_approach(bottleneck: str, recent_strategies: List) -> Dict:
    """推荐不同类型的策略（性能停滞时）"""
    strategies = STRATEGY_PRIORITY.get(bottleneck, ['occupancy_tuning'])

    # 选择未使用过的策略
    for strategy in strategies:
        if strategy not in recent_strategies:
            return {
                'strategy': strategy,
                'reason': '性能停滞，尝试不同类型的优化策略',
                'expected_speedup': _get_expected_speedup(strategy),
                'priority': 'medium'
            }

    # 如果所有策略都用过，推荐效果最好的
    return {
        'strategy': strategies[0],
        'reason': '所有策略已尝试，重新尝试最有效的策略',
        'expected_speedup': _get_expected_speedup(strategies[0]),
        'priority': 'low'
    }


def _recommend_previous_best(rounds: List, bottleneck: str) -> Dict:
    """推荐之前效果最好的策略（性能下降时）"""
    # 找到加速比最高的轮次
    best_round = max(rounds, key=lambda r: r.performance.get('speedup', 0))

    return {
        'strategy': best_round.strategy,
        'reason': f'性能下降，回退到 Round {best_round.round} 的有效策略',
        'expected_speedup': f"{best_round.performance.get('speedup', 1.0):.1f}x",
        'priority': 'high'
    }


def _recommend_stable_strategy(bottleneck: str, recent_strategies: List) -> Dict:
    """推荐更稳定的策略（性能波动时）"""
    # 稳定的策略（通常效果可预测）
    stable_strategies = ['matmul_tiling', 'vectorized_memory', 'occupancy_tuning']

    for strategy in stable_strategies:
        if strategy not in recent_strategies:
            return {
                'strategy': strategy,
                'reason': '性能波动大，尝试更稳定的优化策略',
                'expected_speedup': _get_expected_speedup(strategy),
                'priority': 'medium'
            }

    return _recommend_by_bottleneck(bottleneck)


def _get_expected_speedup(strategy: str) -> str:
    """获取策略的预期加速比"""
    speedup_ranges = {
        'matmul_tiling': '2-5x',
        'reduction_warp_primitives': '3-8x',
        'vectorized_memory': '1.5-3x',
        'kernel_fusion': '2-4x',
        'tensor_core': '2-10x',
        'bank_conflict_free': '1.5-3x',
        'occupancy_tuning': '1.5-4x',
        'cooperative_groups': '1.2-2x',
        'persistent_threads': '2-5x'
    }
    return speedup_ranges.get(strategy, '1.5-3x')


def main():
    """示例用法"""
    from track_optimization import OptimizationHistory

    # 加载历史
    history = OptimizationHistory('example_history.json')
    rounds = history.get_all_rounds()

    # 当前诊断
    current_diagnosis = {
        'bottleneck': 'memory_bandwidth',
        'bandwidth_util': 70.0,
        'occupancy': 65.0
    }

    # 推荐策略
    recommendation = recommend_next_strategy(rounds, current_diagnosis)

    print("策略推荐:")
    print(f"  策略: {recommendation['strategy']}")
    print(f"  理由: {recommendation['reason']}")
    print(f"  预期加速: {recommendation['expected_speedup']}")
    print(f"  优先级: {recommendation['priority']}")


if __name__ == '__main__':
    main()
