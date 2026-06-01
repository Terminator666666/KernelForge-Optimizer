#!/usr/bin/env python3
"""
趋势分析工具
分析优化趋势、检测瓶颈转移、评估策略有效性
"""

from typing import Dict, List, Optional
import statistics


def get_recent_trend(rounds: List, window: int = 3) -> str:
    """
    分析最近 N 轮的优化趋势

    返回:
        'improving': 性能持续提升
        'stagnant': 性能停滞不前
        'degrading': 性能下降
        'unstable': 性能波动大
    """
    if len(rounds) < 2:
        return 'insufficient_data'

    recent = rounds[-window:] if len(rounds) >= window else rounds

    # 计算性能变化率
    changes = []
    for i in range(len(recent) - 1):
        time_prev = recent[i].performance.get('time_ms', 0)
        time_curr = recent[i + 1].performance.get('time_ms', 0)

        if time_prev > 0:
            # 时间减少 = 性能提升（负变化率）
            change = (time_curr - time_prev) / time_prev
            changes.append(change)

    if not changes:
        return 'insufficient_data'

    avg_change = sum(changes) / len(changes)
    std_change = statistics.stdev(changes) if len(changes) > 1 else 0

    # 判断趋势
    if std_change > 0.15:  # 波动大于 15%
        return 'unstable'
    elif avg_change < -0.05:  # 平均性能提升 > 5%
        return 'improving'
    elif avg_change > 0.05:  # 平均性能下降 > 5%
        return 'degrading'
    else:
        return 'stagnant'


def detect_bottleneck_shift(rounds: List, window: int = 3) -> Optional[Dict]:
    """
    检测瓶颈是否发生转移

    返回:
        {'from': 'memory_bandwidth', 'to': 'compute_bound', 'round': 3}
        或 None（无转移）
    """
    if len(rounds) < 2:
        return None

    recent = rounds[-window:] if len(rounds) >= window else rounds

    bottlenecks = [r.diagnosis.get('bottleneck', 'unknown') for r in recent]

    # 检查是否有变化
    if len(set(bottlenecks)) > 1:
        return {
            'from': bottlenecks[0],
            'to': bottlenecks[-1],
            'round': recent[-1].round
        }

    return None


def analyze_strategy_effectiveness(rounds: List) -> Dict:
    """
    分析每个策略的有效性

    返回:
        {
            'matmul_tiling': {
                'count': 2,
                'avg_speedup': 3.5,
                'max_speedup': 4.2,
                'success_rate': 1.0
            }
        }
    """
    strategy_stats = {}

    for round_obj in rounds:
        strategy = round_obj.strategy
        speedup = round_obj.performance.get('speedup', 1.0)

        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                'count': 0,
                'speedups': [],
                'success_count': 0
            }

        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['speedups'].append(speedup)

        # 加速比 > 1.1 视为成功
        if speedup > 1.1:
            strategy_stats[strategy]['success_count'] += 1

    # 计算统计数据
    result = {}
    for strategy, stats in strategy_stats.items():
        speedups = stats['speedups']
        result[strategy] = {
            'count': stats['count'],
            'avg_speedup': sum(speedups) / len(speedups),
            'max_speedup': max(speedups),
            'min_speedup': min(speedups),
            'success_rate': stats['success_count'] / stats['count']
        }

    return result


def calculate_improvement_rate(rounds: List) -> float:
    """
    计算总体改进率

    返回:
        改进率（例如 0.75 表示性能提升了 75%）
    """
    if len(rounds) < 2:
        return 0.0

    first_time = rounds[0].performance.get('time_ms', 0)
    last_time = rounds[-1].performance.get('time_ms', 0)

    if first_time > 0:
        return (first_time - last_time) / first_time

    return 0.0


def identify_best_round(rounds: List) -> Optional[int]:
    """
    识别性能最好的轮次

    返回:
        轮次编号
    """
    if not rounds:
        return None

    best_round = min(rounds, key=lambda r: r.performance.get('time_ms', float('inf')))
    return best_round.round


def main():
    """示例用法"""
    from track_optimization import OptimizationHistory

    # 加载历史
    history = OptimizationHistory('example_history.json')
    rounds = history.get_all_rounds()

    if not rounds:
        print("无历史数据")
        return

    # 分析趋势
    trend = get_recent_trend(rounds, window=3)
    print(f"优化趋势: {trend}")

    # 检测瓶颈转移
    shift = detect_bottleneck_shift(rounds)
    if shift:
        print(f"瓶颈转移: {shift['from']} → {shift['to']}")
    else:
        print("瓶颈未转移")

    # 策略有效性
    effectiveness = analyze_strategy_effectiveness(rounds)
    print("\n策略有效性:")
    for strategy, stats in effectiveness.items():
        print(f"  {strategy}: 平均加速 {stats['avg_speedup']:.2f}x, "
              f"成功率 {stats['success_rate']:.1%}")

    # 总体改进
    improvement = calculate_improvement_rate(rounds)
    print(f"\n总体改进率: {improvement:.1%}")

    # 最佳轮次
    best = identify_best_round(rounds)
    print(f"最佳轮次: Round {best}")


if __name__ == '__main__':
    main()
