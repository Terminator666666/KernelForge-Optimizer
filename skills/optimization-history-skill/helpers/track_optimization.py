#!/usr/bin/env python3
"""
优化历史记录工具
记录每轮优化的诊断、策略、参数和性能结果
"""

import json
from datetime import datetime
from typing import Dict, List, Optional


class OptimizationRound:
    """单轮优化记录"""

    def __init__(self, round_number: int, diagnosis: Dict, strategy: str,
                 parameters: Dict, performance: Dict, timestamp: str = None):
        self.round = round_number
        self.timestamp = timestamp or datetime.now().isoformat()
        self.diagnosis = diagnosis
        self.strategy = strategy
        self.parameters = parameters
        self.performance = performance

    def to_dict(self) -> Dict:
        return {
            'round': self.round,
            'timestamp': self.timestamp,
            'diagnosis': self.diagnosis,
            'strategy': self.strategy,
            'parameters': self.parameters,
            'performance': self.performance
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            round_number=data['round'],
            diagnosis=data['diagnosis'],
            strategy=data['strategy'],
            parameters=data['parameters'],
            performance=data['performance'],
            timestamp=data.get('timestamp')
        )


class OptimizationHistory:
    """优化历史管理"""

    def __init__(self, history_file: str = 'optimization_history.json',
                 kernel_name: str = 'unknown_kernel'):
        self.history_file = history_file
        self.kernel_name = kernel_name
        self.rounds: List[OptimizationRound] = []
        self.load()

    def add_round(self, round_number: int, diagnosis: Dict, strategy: str,
                  parameters: Dict, performance: Dict):
        """添加一轮优化记录"""
        round_obj = OptimizationRound(
            round_number, diagnosis, strategy, parameters, performance
        )
        self.rounds.append(round_obj)

    def get_round(self, round_number: int) -> Optional[OptimizationRound]:
        """获取指定轮次"""
        for round_obj in self.rounds:
            if round_obj.round == round_number:
                return round_obj
        return None

    def get_recent_rounds(self, window: int = 3) -> List[OptimizationRound]:
        """获取最近 N 轮"""
        return self.rounds[-window:] if len(self.rounds) >= window else self.rounds

    def get_all_rounds(self) -> List[OptimizationRound]:
        """获取所有轮次"""
        return self.rounds

    def save(self):
        """保存到文件"""
        data = {
            'kernel_name': self.kernel_name,
            'total_rounds': len(self.rounds),
            'rounds': [r.to_dict() for r in self.rounds]
        }

        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """从文件加载"""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.kernel_name = data.get('kernel_name', 'unknown_kernel')
                self.rounds = [OptimizationRound.from_dict(r) for r in data.get('rounds', [])]
        except FileNotFoundError:
            # 文件不存在，使用空历史
            self.rounds = []

    def clear(self):
        """清空历史"""
        self.rounds = []

    def generate_summary(self) -> str:
        """生成摘要"""
        if not self.rounds:
            return "无优化历史"

        first_round = self.rounds[0]
        last_round = self.rounds[-1]

        total_speedup = last_round.performance.get('speedup', 1.0)

        summary = f"""
优化历史摘要
============
Kernel: {self.kernel_name}
总轮次: {len(self.rounds)}
总加速比: {total_speedup:.2f}x

第一轮:
  策略: {first_round.strategy}
  瓶颈: {first_round.diagnosis.get('bottleneck', 'unknown')}
  时间: {first_round.performance.get('time_ms', 0):.2f} ms

最后一轮:
  策略: {last_round.strategy}
  瓶颈: {last_round.diagnosis.get('bottleneck', 'unknown')}
  时间: {last_round.performance.get('time_ms', 0):.2f} ms
"""
        return summary


def main():
    """示例用法"""
    history = OptimizationHistory('example_history.json', 'matmul_kernel')

    # 添加几轮优化
    history.add_round(
        1,
        {'bottleneck': 'memory_bandwidth', 'bandwidth_util': 85.0},
        'matmul_tiling',
        {'TILE_SIZE': 32},
        {'time_ms': 12.5, 'speedup': 3.6}
    )

    history.add_round(
        2,
        {'bottleneck': 'memory_bandwidth', 'bandwidth_util': 75.0},
        'vectorized_memory',
        {'VECTOR_SIZE': 4},
        {'time_ms': 9.8, 'speedup': 4.6}
    )

    # 保存
    history.save()

    # 打印摘要
    print(history.generate_summary())


if __name__ == '__main__':
    main()
