# Roofline Model Analysis

## Overview

Roofline 模型是一个直观的性能分析框架，用于判断 CUDA kernel 是 memory-bound 还是 compute-bound，并计算距离理论峰值性能的差距。

## Roofline 模型基础

### 核心概念

**Roofline 模型**由两条"屋顶线"组成：
1. **Memory Roof（内存屋顶）**：由内存带宽限制的性能上限
2. **Compute Roof（计算屋顶）**：由计算能力限制的性能上限

**Ridge Point（脊点）**：两条屋顶线的交点，是算术强度的临界值

### 数学定义

**性能上限**：
```
Performance = min(
    Memory_Bandwidth × Arithmetic_Intensity,  # Memory-bound
    Peak_Compute_Throughput                    # Compute-bound
)
```

**Ridge Point**：
```
Ridge_Point = Peak_Compute_Throughput / Memory_Bandwidth
```

**示例**（RTX 4090）：
```
Peak FP32 = 82.6 TFLOPS = 82.6 × 10^12 FLOPs/s
Peak Bandwidth = 1008 GB/s = 1008 × 10^9 Bytes/s
Ridge Point = 82.6 × 10^12 / (1008 × 10^9) ≈ 82 FLOPs/Byte
```

### Roofline 图示

```
Performance (TFLOPS)
    ^
    |                    _____________________ Compute Roof (82.6 TFLOPS)
    |                   /
    |                  /
    |                 /
    |                / Ridge Point (82 FLOPs/Byte)
    |               /
    |              /
    |             /  Memory Roof (Bandwidth × AI)
    |            /
    |           /
    |          /
    |_________/____________________________________________> Arithmetic Intensity (FLOPs/Byte)
    0         10        20        50       100
```

## Roofline 分析实现

### 1. 计算算术强度 (Arithmetic Intensity)

**定义**：每字节内存访问执行的浮点运算数量

**计算公式**：
```python
arithmetic_intensity = total_flops / total_bytes_transferred
```

**实现**：
```python
# 统计浮点运算数
flops = (
    metrics.get('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum', 0) +
    metrics.get('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum', 0) +
    metrics.get('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum', 0) * 2
)

# 统计内存传输字节数
bytes_transferred = metrics.get('dram__bytes.sum', 1)

# 计算算术强度
arithmetic_intensity = flops / max(bytes_transferred, 1)
```

### 2. 确定 Roofline 区域

**区域划分**：
- **Memory-bound**：AI < Ridge_Point × 0.5
- **Balanced**：Ridge_Point × 0.5 ≤ AI ≤ Ridge_Point × 2.0
- **Compute-bound**：AI > Ridge_Point × 2.0

**实现**：
```python
def _analyze_roofline(derived, memory_analysis, compute_analysis):
    """执行 Roofline 模型分析"""
    ai = derived['arithmetic_intensity']
    achieved_bw = derived['achieved_bandwidth']
    achieved_flops = derived['achieved_tflops'] * 1e12
    
    # 计算 ridge point
    ridge_point = peak_flops_fp32 / (peak_bandwidth * 1e9)
    
    # 确定区域
    if ai < ridge_point * 0.5:
        region = 'memory_bound'
        theoretical_peak = achieved_bw * 1e9 * ai  # 带宽限制
    elif ai > ridge_point * 2.0:
        region = 'compute_bound'
        theoretical_peak = peak_flops_fp32  # 计算限制
    else:
        region = 'balanced'
        theoretical_peak = min(achieved_bw * 1e9 * ai, peak_flops_fp32)
    
    # 计算距离理论峰值的差距
    distance = achieved_flops / max(theoretical_peak, 1)
    
    return {
        'arithmetic_intensity': ai,
        'region': region,
        'ridge_point': ridge_point,
        'distance_to_peak': distance,
        'theoretical_peak_flops': theoretical_peak,
        'achieved_flops': achieved_flops
    }
```

### 3. 计算效率 (Distance to Roofline)

**定义**：实际性能占理论峰值性能的百分比

**计算公式**：
```python
efficiency = achieved_performance / theoretical_peak_performance
```

**解释指南**：

| 效率 | 含义 | 优化建议 |
|-----|------|---------|
| > 80% | 接近理论峰值 | 性能已优化良好 |
| 50-80% | 中等效率 | 有优化空间 |
| 30-50% | 较低效率 | 需要优化 |
| < 30% | 很低效率 | 严重性能问题 |

## Roofline 区域特征

### Memory-Bound Region

**特征**：
- 算术强度低（< Ridge_Point × 0.5）
- 内存带宽利用率高（> 70%）
- 计算利用率低（< 50%）

**典型算子**：
- Element-wise operations（ReLU, Sigmoid, Add）
- Transpose
- Copy
- Reduction（部分）

**优化策略**：
1. **减少内存访问**：
   - Kernel fusion（合并多个 kernel）
   - 使用共享内存缓存数据
   - 向量化访问（float4, int4）

2. **提高算术强度**：
   - 增加计算量（如果可能）
   - 重用数据

3. **优化内存访问模式**：
   - 确保内存合并访问
   - 提高缓存命中率

**代码示例**：
```cuda
// Memory-bound: Element-wise operation
__global__ void relu_kernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = max(0.0f, data[tid]);  // AI ≈ 1 FLOPs/4 Bytes = 0.25
    }
}

// 优化：Kernel fusion 提高 AI
__global__ void fused_relu_add_kernel(float* a, float* b, float* c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = a[tid] + b[tid];  // 1 FLOP
        c[tid] = max(0.0f, val);      // 1 FLOP
        // AI ≈ 2 FLOPs / 12 Bytes = 0.17 → 仍然 memory-bound，但减少了 kernel 启动开销
    }
}
```

### Compute-Bound Region

**特征**：
- 算术强度高（> Ridge_Point × 2.0）
- 计算利用率高（> 70%）
- 内存带宽利用率低（< 50%）

**典型算子**：
- Matrix Multiplication
- Convolution
- FFT
- Dense Neural Network Layers

**优化策略**：
1. **使用专用硬件**：
   - Tensor Cores（FP16/BF16/INT8）
   - CUDA Cores 优化

2. **优化指令混合**：
   - 使用 FMA 指令
   - 减少低效指令

3. **算法优化**：
   - 使用更高效的算法（如 Strassen 矩阵乘法）
   - 减少不必要的计算

**代码示例**：
```cuda
// Compute-bound: Matrix Multiplication
__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];  // 2N FLOPs
    }
    C[row * N + col] = sum;
    // AI ≈ 2N FLOPs / (2N + 1) × 4 Bytes ≈ N/4 (对于大 N)
    // 例如 N=1024，AI ≈ 256 FLOPs/Byte >> Ridge Point
}

// 优化：使用 Tensor Cores (FP16)
// AI 不变，但峰值计算能力提高 2-4 倍
```

### Balanced Region

**特征**：
- 算术强度接近 Ridge Point
- 内存和计算都有一定利用率

**典型算子**：
- Softmax
- LayerNorm
- Reduction（部分）
- Pooling

**优化策略**：
- 同时优化内存和计算
- 根据具体瓶颈调整策略

## 典型算子的 Roofline 位置

### 算子分类表

| 算子类型 | 算术强度 (AI) | Roofline 区域 | 主要瓶颈 |
|---------|--------------|--------------|---------|
| Element-wise (ReLU, Add) | 0.25 - 1 | Memory-bound | 内存带宽 |
| Transpose | 0 | Memory-bound | 内存带宽 |
| Reduction (Sum, Max) | 1 - 5 | Memory-bound / Balanced | 内存带宽 |
| Softmax | 5 - 10 | Balanced | 内存 + 计算 |
| LayerNorm | 10 - 20 | Balanced / Compute-bound | 内存 + 计算 |
| MatMul (N=64) | 16 | Balanced | 内存 + 计算 |
| MatMul (N=256) | 64 | Compute-bound | 计算 |
| MatMul (N=1024) | 256 | Compute-bound | 计算 |
| Convolution | 50 - 500 | Compute-bound | 计算 |

### RTX 4090 示例

**GPU 规格**：
- Peak FP32: 82.6 TFLOPS
- Peak Bandwidth: 1008 GB/s
- Ridge Point: 82 FLOPs/Byte

**算子分析**：

```
AI < 41 FLOPs/Byte  → Memory-bound
41 ≤ AI ≤ 164       → Balanced
AI > 164            → Compute-bound
```

**示例**：
- ReLU (AI ≈ 0.25): 严重 memory-bound
- Softmax (AI ≈ 8): Memory-bound
- MatMul N=256 (AI ≈ 64): Balanced
- MatMul N=1024 (AI ≈ 256): Compute-bound

## Roofline 分析输出

### 输出格式

```python
{
    'arithmetic_intensity': 64.5,        # 算术强度 64.5 FLOPs/Byte
    'region': 'balanced',                # Roofline 区域
    'ridge_point': 82.0,                 # Ridge point 82 FLOPs/Byte
    'distance_to_peak': 0.65,            # 效率 65%
    'theoretical_peak_flops': 5.2e13,    # 理论峰值 52 TFLOPS
    'achieved_flops': 3.4e13             # 实际性能 34 TFLOPS
}
```

### 诊断报告示例

**场景 1：Memory-bound Kernel**
```
Roofline Analysis:
- Arithmetic Intensity: 2.5 FLOPs/Byte
- Region: Memory-bound (AI < 41 FLOPs/Byte)
- Efficiency: 45% of theoretical peak
- Bottleneck: Memory bandwidth (850 GB/s / 1008 GB/s = 84%)

Recommendation:
1. Reduce memory traffic via kernel fusion
2. Use shared memory to cache frequently accessed data
3. Increase arithmetic intensity by adding more computation
```

**场景 2：Compute-bound Kernel**
```
Roofline Analysis:
- Arithmetic Intensity: 180 FLOPs/Byte
- Region: Compute-bound (AI > 164 FLOPs/Byte)
- Efficiency: 55% of theoretical peak
- Bottleneck: Compute throughput (45 TFLOPS / 82.6 TFLOPS = 55%)

Recommendation:
1. Use Tensor Cores for FP16/BF16 computation
2. Optimize instruction mix (use FMA instructions)
3. Improve occupancy to hide latency
```

**场景 3：Balanced Kernel**
```
Roofline Analysis:
- Arithmetic Intensity: 75 FLOPs/Byte
- Region: Balanced (41 ≤ AI ≤ 164)
- Efficiency: 60% of theoretical peak
- Bottleneck: Both memory and compute

Recommendation:
1. Optimize memory access patterns (coalescing)
2. Improve cache hit rates via tiling
3. Increase occupancy
4. Consider using Tensor Cores if applicable
```

## 使用 Roofline 指导优化

### 优化决策树

```
1. 确定 Roofline 区域
   ├─ Memory-bound → 优化内存访问
   ├─ Compute-bound → 优化计算效率
   └─ Balanced → 同时优化内存和计算

2. 检查效率
   ├─ > 80% → 性能已优化良好
   ├─ 50-80% → 有优化空间
   └─ < 50% → 需要深入优化

3. 识别具体瓶颈
   ├─ 带宽利用率
   ├─ 访问模式
   ├─ 缓存命中率
   ├─ 占用率
   └─ 计算利用率

4. 应用优化策略
   └─ 根据瓶颈选择合适的优化技术
```

## 参考资料

- Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
- NVIDIA Nsight Compute - Roofline Analysis
- CUDA C Best Practices Guide - Performance Metrics
