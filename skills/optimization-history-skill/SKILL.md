---
name: optimization-history-skill
description: CUDA 优化历史管理，记录优化轮次、分析趋势、推荐下一步策略
version: 1.0.0
author: KernelForge-Optimizer
tags: [cuda, optimization, history, trend-analysis, recommendation]
---

# Optimization History Skill

## 概述

Optimization History Skill 提供优化历史记录和趋势分析功能，帮助 LLM 和开发者：
- 记录每轮优化的诊断、策略、参数、性能结果
- 分析优化趋势（improving/stagnant/degrading/unstable）
- 检测瓶颈转移（从内存瓶颈转为计算瓶颈等）
- 推荐下一步优化策略（基于历史数据和当前状态）

## 核心功能

### 1. 优化历史记录

记录每轮优化的完整信息：

```python
from optimization_history import OptimizationHistory

history = OptimizationHistory()

# 记录一轮优化
history.add_round(
    round_number=1,
    diagnosis={'bottleneck': 'memory_bandwidth', 'bandwidth_util': 85.0},
    strategy='matmul_tiling',
    parameters={'TILE_SIZE': 32},
    performance={'time_ms': 12.5, 'speedup': 3.6}
)
```

### 2. 趋势分析

```python
trend = history.get_recent_trend(window=3)
# 返回: 'improving', 'stagnant', 'degrading', 'unstable'
```

### 3. 瓶颈转移检测

```python
shift = history.detect_bottleneck_shift()
# 返回: {'from': 'memory_bandwidth', 'to': 'compute_bound'}
```

### 4. 智能推荐

```python
recommendation = history.recommend_next_strategy(
    current_diagnosis={'bottleneck': 'memory_bandwidth'}
)
# 返回: {'strategy': 'kernel_fusion', 'reason': '...'}
```

## 使用方法

```python
# 初始化
history = OptimizationHistory('optimization_history.json')

# 记录优化
history.add_round(1, diagnosis, strategy, parameters, performance)

# 分析趋势
trend = history.get_recent_trend()

# 推荐下一步
recommendation = history.recommend_next_strategy(current_diagnosis)

# 保存
history.save()
```

## 历史记录格式

```json
{
  "kernel_name": "matmul_kernel",
  "rounds": [
    {
      "round": 1,
      "diagnosis": {"bottleneck": "memory_bandwidth"},
      "strategy": "matmul_tiling",
      "parameters": {"TILE_SIZE": 32},
      "performance": {"time_ms": 12.5, "speedup": 3.6}
    }
  ]
}
```

## 参考文档

- `reference/00-overview.md` - 优化历史概述
- `reference/01-history-tracking.md` - 历史记录详解
- `reference/02-trend-analysis.md` - 趋势分析算法

## Python 工具

- `helpers/track_optimization.py` - 优化历史记录
- `helpers/analyze_trends.py` - 趋势分析
- `helpers/recommend_strategy.py` - 策略推荐
