# Bottleneck Identification

## Overview

瓶颈识别是 NCU Interpreter 的核心功能，通过综合分析内存、计算和 Roofline 指标，自动识别 CUDA kernel 的主要性能瓶颈，并给出置信度评分。

## 瓶颈类型

NCU Interpreter 识别 4 种主要瓶颈类型：

### 1. Memory Bandwidth Bound（内存带宽瓶颈）

**定义**：内存带宽饱和，数据传输速度限制了 kernel 性能

**特征**：
- 内存带宽利用率高（> 70%）
- Roofline 区域为 memory-bound
- 访问模式良好（coalesced 或 strided）
- 计算利用率低（< 50%）

**典型场景**：
- 大规模数据搬运
- Element-wise 操作（ReLU, Add, Mul）
- 低算术强度的算子

### 2. Memory Latency Bound（内存延迟瓶颈）

**定义**：内存访问延迟高，等待数据的时间限制了性能

**特征**：
- 内存带宽利用率低（< 40%）
- 占用率低（< 50%）
- 访问模式差（random 或 mixed）
- 缓存命中率低（L1 < 50%）

**典型场景**：
- 随机内存访问
- 指针追踪（pointer chasing）
- 稀疏矩阵操作
- 低占用率导致无法隐藏延迟

### 3. Compute Bound（计算瓶颈）

**定义**：计算单元饱和，计算能力限制了性能

**特征**：
- 计算利用率高（> 70%）
- Roofline 区域为 compute-bound
- 算术强度高（> Ridge Point）
- 内存带宽利用率低（< 50%）

**典型场景**：
- 矩阵乘法（大矩阵）
- 卷积操作
- FFT
- 密集计算的神经网络层

### 4. Occupancy Bound（占用率瓶颈）

**定义**：SM 占用率过低，并行度不足限制了性能

**特征**：
- 占用率很低（< 30%）
- 资源限制明显（registers/shared_memory）
- SM 效率低（< 50%）
- 内存和计算利用率都低

**典型场景**：
- 寄存器使用过多
- 共享内存分配过大
- Block size 设置不当
- 复杂的 kernel 逻辑

## 瓶颈识别算法

### 评分机制

NCU Interpreter 使用评分机制识别瓶颈，每种瓶颈类型根据相关指标累积得分，得分最高的即为主要瓶颈。

### 实现代码

```python
def _identify_bottleneck(memory_analysis, compute_analysis, roofline):
    """识别主要瓶颈及置信度"""
    
    bw_util = memory_analysis['bandwidth_util']
    compute_util = compute_analysis['compute_util']
    occupancy = compute_analysis['achieved_occupancy']
    region = roofline['region']
    
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
    if memory_analysis['access_pattern'] in ['coalesced', 'strided']:
        scores['memory_bandwidth'] += 0.2  # 访问模式良好
    
    # === Memory Latency 瓶颈指标 ===
    if bw_util < 40 and occupancy < 50:
        scores['memory_latency'] += 0.4  # 低带宽 + 低占用率
    if memory_analysis['access_pattern'] in ['random', 'mixed']:
        scores['memory_latency'] += 0.3  # 访问模式差
    l1_hit = memory_analysis.get('l1_hit_rate', 100)
    if l1_hit < 50:
        scores['memory_latency'] += 0.2  # 低缓存命中率
    
    # === Compute Bound 瓶颈指标 ===
    if compute_util > 70:
        scores['compute_bound'] += 0.4  # 高计算利用率
    if region == 'compute_bound':
        scores['compute_bound'] += 0.3  # Roofline 显示 compute-bound
    if roofline['arithmetic_intensity'] > roofline['ridge_point']:
        scores['compute_bound'] += 0.2  # 算术强度超过 ridge point
    
    # === Occupancy 瓶颈指标 ===
    if occupancy < 30:
        scores['occupancy'] += 0.5  # 极低占用率
    if compute_analysis['limiting_factor'] in ['registers', 'shared_memory']:
        scores['occupancy'] += 0.3  # 明确的资源限制
    
    # 找出得分最高的瓶颈
    bottleneck = max(scores, key=scores.get)
    confidence = scores[bottleneck]
    
    # 归一化置信度到 0-1 范围
    confidence = min(confidence, 1.0)
    
    return bottleneck, confidence
```

## 评分规则详解

### Memory Bandwidth 评分规则

| 指标 | 条件 | 得分 | 理由 |
|-----|------|------|------|
| 带宽利用率 | > 70% | +0.4 | 带宽接近饱和 |
| Roofline 区域 | memory_bound | +0.3 | 理论分析确认 |
| 访问模式 | coalesced/strided | +0.2 | 访问效率高，带宽是瓶颈 |

**最高得分**：0.9（强烈确认为内存带宽瓶颈）

### Memory Latency 评分规则

| 指标 | 条件 | 得分 | 理由 |
|-----|------|------|------|
| 带宽 + 占用率 | BW < 40% AND Occ < 50% | +0.4 | 低带宽低占用率，延迟问题 |
| 访问模式 | random/mixed | +0.3 | 访问模式差导致延迟高 |
| L1 命中率 | < 50% | +0.2 | 缓存未命中导致延迟 |

**最高得分**：0.9（强烈确认为内存延迟瓶颈）

### Compute Bound 评分规则

| 指标 | 条件 | 得分 | 理由 |
|-----|------|------|------|
| 计算利用率 | > 70% | +0.4 | 计算单元接近饱和 |
| Roofline 区域 | compute_bound | +0.3 | 理论分析确认 |
| 算术强度 | > Ridge Point | +0.2 | 计算密集型 |

**最高得分**：0.9（强烈确认为计算瓶颈）

### Occupancy 评分规则

| 指标 | 条件 | 得分 | 理由 |
|-----|------|------|------|
| 占用率 | < 30% | +0.5 | 极低占用率严重限制性能 |
| 限制因素 | registers/shared_memory | +0.3 | 明确的资源限制 |

**最高得分**：0.8（强烈确认为占用率瓶颈）

## 置信度解释

| 置信度 | 含义 | 建议 |
|-------|------|------|
| > 0.7 | 高置信度 | 瓶颈明确，优先优化此项 |
| 0.5-0.7 | 中等置信度 | 可能是主要瓶颈，需进一步分析 |
| 0.3-0.5 | 低置信度 | 瓶颈不明显，可能有多个瓶颈 |
| < 0.3 | 很低置信度 | 无明显瓶颈或需要更多数据 |

## 诊断示例

### 示例 1：Memory Bandwidth Bound

**输入指标**：
```python
memory_analysis = {
    'bandwidth_util': 85.0,
    'access_pattern': 'coalesced',
    'l1_hit_rate': 75.0
}
compute_analysis = {
    'achieved_occupancy': 60.0,
    'compute_util': 30.0
}
roofline = {
    'region': 'memory_bound',
    'arithmetic_intensity': 2.5,
    'ridge_point': 82.0
}
```

**评分过程**：
```
memory_bandwidth:
  + 0.4 (bw_util > 70)
  + 0.3 (region == 'memory_bound')
  + 0.2 (access_pattern == 'coalesced')
  = 0.9

memory_latency: 0.0
compute_bound: 0.0
occupancy: 0.0
```

**结果**：
```python
bottleneck = 'memory_bandwidth'
confidence = 0.9  # 高置信度
```

**诊断报告**：
```
Primary Bottleneck: Memory Bandwidth (confidence: 90%)

Evidence:
- Bandwidth utilization: 85% (high)
- Roofline region: memory-bound
- Access pattern: coalesced (efficient)
- Compute utilization: 30% (low)

Recommendation:
1. Reduce memory traffic via kernel fusion
2. Use shared memory to cache frequently accessed data
3. Increase arithmetic intensity by adding more computation
```

### 示例 2：Memory Latency Bound

**输入指标**：
```python
memory_analysis = {
    'bandwidth_util': 25.0,
    'access_pattern': 'random',
    'l1_hit_rate': 35.0
}
compute_analysis = {
    'achieved_occupancy': 40.0,
    'compute_util': 20.0
}
roofline = {
    'region': 'memory_bound',
    'arithmetic_intensity': 1.5,
    'ridge_point': 82.0
}
```

**评分过程**：
```
memory_latency:
  + 0.4 (bw_util < 40 AND occupancy < 50)
  + 0.3 (access_pattern == 'random')
  + 0.2 (l1_hit_rate < 50)
  = 0.9

memory_bandwidth: 0.3 (region == 'memory_bound')
compute_bound: 0.0
occupancy: 0.0
```

**结果**：
```python
bottleneck = 'memory_latency'
confidence = 0.9  # 高置信度
```

**诊断报告**：
```
Primary Bottleneck: Memory Latency (confidence: 90%)

Evidence:
- Bandwidth utilization: 25% (low)
- Occupancy: 40% (low, cannot hide latency)
- Access pattern: random (poor)
- L1 cache hit rate: 35% (low)

Recommendation:
1. Improve memory coalescing
2. Increase occupancy to hide latency
3. Use shared memory for frequently accessed data
4. Reorganize data layout to improve locality
```

### 示例 3：Compute Bound

**输入指标**：
```python
memory_analysis = {
    'bandwidth_util': 35.0,
    'access_pattern': 'coalesced',
    'l1_hit_rate': 80.0
}
compute_analysis = {
    'achieved_occupancy': 75.0,
    'compute_util': 85.0
}
roofline = {
    'region': 'compute_bound',
    'arithmetic_intensity': 150.0,
    'ridge_point': 82.0
}
```

**评分过程**：
```
compute_bound:
  + 0.4 (compute_util > 70)
  + 0.3 (region == 'compute_bound')
  + 0.2 (ai > ridge_point)
  = 0.9

memory_bandwidth: 0.0
memory_latency: 0.0
occupancy: 0.0
```

**结果**：
```python
bottleneck = 'compute_bound'
confidence = 0.9  # 高置信度
```

**诊断报告**：
```
Primary Bottleneck: Compute Bound (confidence: 90%)

Evidence:
- Compute utilization: 85% (high)
- Roofline region: compute-bound
- Arithmetic intensity: 150 FLOPs/Byte (>> ridge point 82)
- Bandwidth utilization: 35% (low)

Recommendation:
1. Use Tensor Cores for FP16/BF16 computation
2. Optimize instruction mix (use FMA instructions)
3. Consider algorithmic optimizations
```

### 示例 4：Occupancy Bound

**输入指标**：
```python
memory_analysis = {
    'bandwidth_util': 20.0,
    'access_pattern': 'coalesced',
    'l1_hit_rate': 70.0
}
compute_analysis = {
    'achieved_occupancy': 18.0,
    'compute_util': 25.0,
    'limiting_factor': 'registers'
}
roofline = {
    'region': 'memory_bound',
    'arithmetic_intensity': 5.0,
    'ridge_point': 82.0
}
```

**评分过程**：
```
occupancy:
  + 0.5 (occupancy < 30)
  + 0.3 (limiting_factor == 'registers')
  = 0.8

memory_latency: 0.4 (bw_util < 40 AND occupancy < 50)
memory_bandwidth: 0.3 (region == 'memory_bound')
compute_bound: 0.0
```

**结果**：
```python
bottleneck = 'occupancy'
confidence = 0.8  # 高置信度
```

**诊断报告**：
```
Primary Bottleneck: Low Occupancy (confidence: 80%)

Evidence:
- Occupancy: 18% (very low)
- Limiting factor: registers
- SM efficiency: low
- Both memory and compute underutilized

Recommendation:
1. Reduce register usage per thread
2. Use __launch_bounds__ to limit register allocation
3. Increase block size if possible
4. Simplify kernel logic to reduce register pressure
```

## 多瓶颈场景

### 场景：多个瓶颈得分接近

**示例**：
```python
scores = {
    'memory_bandwidth': 0.6,
    'memory_latency': 0.5,
    'compute_bound': 0.2,
    'occupancy': 0.4
}
```

**处理方式**：
1. 选择得分最高的作为主要瓶颈（memory_bandwidth）
2. 在诊断报告中提及次要瓶颈（memory_latency）
3. 建议同时优化多个方面

**诊断报告**：
```
Primary Bottleneck: Memory Bandwidth (confidence: 60%)

Note: Multiple potential bottlenecks detected
- Memory bandwidth: 60%
- Memory latency: 50%
- Occupancy: 40%

Recommendation:
1. Address memory bandwidth first (highest score)
2. Also consider memory latency optimization
3. Monitor occupancy after initial optimizations
```

## 参考资料

- NVIDIA Nsight Compute - Performance Analysis Guide
- CUDA C Best Practices Guide - Performance Metrics
- Roofline Model Paper (Williams et al., 2009)
