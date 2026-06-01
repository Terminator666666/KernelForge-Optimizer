# Optimization History Skill

CUDA 优化历史管理和趋势分析工具。

## 功能

- ✅ 记录每轮优化的完整信息
- ✅ 分析优化趋势（improving/stagnant/degrading/unstable）
- ✅ 检测瓶颈转移
- ✅ 分析策略有效性
- ✅ 智能推荐下一步策略

## 快速开始

```python
from optimization_history import OptimizationHistory

# 初始化
history = OptimizationHistory('history.json')

# 记录优化
history.add_round(
    round_number=1,
    diagnosis={'bottleneck': 'memory_bandwidth', 'bandwidth_util': 85.0},
    strategy='matmul_tiling',
    parameters={'TILE_SIZE': 32},
    performance={'time_ms': 12.5, 'speedup': 3.6}
)

# 分析趋势
trend = history.get_recent_trend(window=3)
print(f"趋势: {trend}")

# 推荐下一步
rec = history.recommend_next_strategy({'bottleneck': 'memory_bandwidth'})
print(f"推荐: {rec['strategy']}")

# 保存
history.save()
```

## 目录结构

```
optimization-history-skill/
├── SKILL.md                    # Skill 定义
├── README.md                   # 本文件
├── reference/                  # 参考文档
│   ├── 00-overview.md         # 概述
│   ├── 01-history-tracking.md # 历史记录
│   └── 02-trend-analysis.md   # 趋势分析
├── helpers/                    # Python 工具
│   ├── track_optimization.py  # 历史记录
│   ├── analyze_trends.py      # 趋势分析
│   └── recommend_strategy.py  # 策略推荐
└── data/                       # 数据文件
    └── history_schema.json    # 历史记录模式
```

## 核心概念

### 优化轮次 (Round)

每轮优化包含：
- **诊断 (Diagnosis)**：瓶颈类型、性能指标
- **策略 (Strategy)**：使用的优化策略
- **参数 (Parameters)**：策略参数
- **性能 (Performance)**：优化后的性能

### 趋势类型 (Trend)

- **improving**：性能持续提升
- **stagnant**：性能停滞不前
- **degrading**：性能下降
- **unstable**：性能波动大

### 瓶颈转移 (Bottleneck Shift)

优化过程中瓶颈可能发生变化：
- memory_bandwidth → compute_bound
- memory_latency → memory_bandwidth
- occupancy → compute_bound

## 使用示例

### 示例 1：记录优化历史

```python
history = OptimizationHistory()

# 第 1 轮：使用分块优化
history.add_round(1, 
    {'bottleneck': 'memory_bandwidth', 'bandwidth_util': 85.0},
    'matmul_tiling',
    {'TILE_SIZE': 32},
    {'time_ms': 12.5, 'speedup': 3.6}
)

# 第 2 轮：使用向量化
history.add_round(2,
    {'bottleneck': 'memory_bandwidth', 'bandwidth_util': 75.0},
    'vectorized_memory',
    {'VECTOR_SIZE': 4},
    {'time_ms': 9.8, 'speedup': 4.6}
)
```

### 示例 2：分析趋势

```python
# 获取最近 3 轮的趋势
trend = history.get_recent_trend(window=3)

if trend == 'improving':
    print("性能持续提升，继续当前方向")
elif trend == 'stagnant':
    print("性能停滞，需要尝试不同策略")
elif trend == 'degrading':
    print("性能下降，考虑回退")
```

### 示例 3：检测瓶颈转移

```python
shift = history.detect_bottleneck_shift()

if shift:
    print(f"瓶颈从 {shift['from']} 转移到 {shift['to']}")
    print("需要调整优化策略")
```

### 示例 4：推荐下一步策略

```python
recommendation = history.recommend_next_strategy(
    current_diagnosis={'bottleneck': 'memory_bandwidth', 'bandwidth_util': 70.0}
)

print(f"推荐策略: {recommendation['strategy']}")
print(f"推荐理由: {recommendation['reason']}")
print(f"预期效果: {recommendation['expected_speedup']}")
```

## 与其他 Skills 的协作

### 完整优化流程

1. **NCU Interpreter** → 分析性能，识别瓶颈
2. **Optimization History** → 检查历史，分析趋势
3. **Strategy Library** → 根据瓶颈和历史推荐策略
4. **Agent** → 实现优化代码
5. **Verification** → 验证性能提升
6. **Optimization History** → 记录结果，更新历史

## API 参考

### OptimizationHistory

```python
class OptimizationHistory:
    def __init__(self, history_file: str)
    def add_round(self, round_number, diagnosis, strategy, parameters, performance)
    def get_recent_trend(self, window: int = 3) -> str
    def detect_bottleneck_shift() -> dict
    def analyze_strategy_effectiveness() -> dict
    def recommend_next_strategy(self, current_diagnosis) -> dict
    def save()
    def load()
    def generate_report() -> str
```

## 常见问题

### Q: 历史记录应该保留多久？

**A**: 推荐保留最近 10-20 轮，足够分析趋势。更早的记录可以归档。

### Q: 如何处理性能波动？

**A**: 使用趋势分析的 `window` 参数，增大窗口可以平滑波动。

### Q: 推荐算法的准确性如何？

**A**: 基于历史数据和规则，准确率约 70-80%。需要结合人工判断。

## 参考资料

- Kernel Design Agents (KDA)
- Auto-tuning 技术
- Bayesian Optimization

## 许可证

MIT License
