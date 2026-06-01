# Derived Metrics Calculation

## Overview

派生指标（Derived Metrics）是从原始 NCU 指标计算得出的高层次性能指标，更容易理解和解释。NCU Interpreter 计算以下关键派生指标：

## 1. 内存带宽利用率 (Memory Bandwidth Utilization)

### 定义

内存带宽利用率表示 kernel 实际使用的内存带宽占 GPU 理论峰值带宽的百分比。

### 计算公式

```
bandwidth_util = (achieved_bandwidth / peak_bandwidth) × 100%

其中：
achieved_bandwidth = total_bytes_transferred / kernel_duration (GB/s)
peak_bandwidth = GPU 理论峰值带宽 (GB/s)
```

### 所需 NCU 指标

- `dram__bytes.sum` - DRAM 传输的总字节数
- `duration` - Kernel 执行时间（纳秒）

### 实现示例

```python
bytes_transferred = metrics['dram__bytes.sum']
duration_ns = metrics['duration']
achieved_bw_gbps = bytes_transferred / duration_ns  # GB/s
bandwidth_util = (achieved_bw_gbps / peak_bandwidth) * 100
```

### 解释指南

| 带宽利用率 | 含义 | 优化建议 |
|-----------|------|---------|
| > 80% | 内存带宽饱和，可能是瓶颈 | 减少内存访问、使用共享内存、提高算术强度 |
| 50-80% | 中等带宽使用 | 可能有优化空间，检查访问模式 |
| 30-50% | 较低带宽使用 | Kernel 可能是 compute-bound 或占用率低 |
| < 30% | 很低带宽使用 | 检查并行度、占用率、计算瓶颈 |

## 2. 算术强度 (Arithmetic Intensity)

### 定义

算术强度表示每字节内存访问执行的浮点运算数量，是 Roofline 模型的核心指标。

### 计算公式

```
arithmetic_intensity = total_flops / total_bytes_transferred (FLOPs/Byte)

其中：
total_flops = FADD + FMUL + FFMA×2
```

### 所需 NCU 指标

- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` - FP32 加法指令数
- `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` - FP32 乘法指令数
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` - FP32 融合乘加指令数
- `dram__bytes.sum` - DRAM 传输的总字节数

### 实现示例

```python
flops = (
    metrics.get('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum', 0) +
    metrics.get('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum', 0) +
    metrics.get('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum', 0) * 2
)
bytes_transferred = metrics.get('dram__bytes.sum', 1)
arithmetic_intensity = flops / max(bytes_transferred, 1)
```

### 解释指南

| 算术强度 | 特征 | 典型算子 |
|---------|------|---------|
| < 1 FLOPs/Byte | Memory-bound，内存密集型 | Element-wise ops, Transpose, Copy |
| 1-10 FLOPs/Byte | Balanced，平衡型 | Reduction, Softmax, LayerNorm |
| > 10 FLOPs/Byte | Compute-bound，计算密集型 | MatMul, Convolution, FFT |

**Ridge Point（脊点）**：算术强度的临界值，由 GPU 架构决定
```
ridge_point = peak_flops / peak_bandwidth
```
- AI < ridge_point：memory-bound
- AI > ridge_point：compute-bound

## 3. 占用率 (Occupancy)

### 定义

占用率表示 SM 上实际活跃的 warps 数量占理论最大 warps 数量的百分比。

### 计算公式

```
achieved_occupancy = active_warps / max_warps × 100%

theoretical_occupancy = (max_blocks × block_size) / max_threads_per_sm × 100%

其中：
max_blocks = min(
    max_threads_per_sm / block_size,
    max_registers / (registers_per_thread × block_size),
    max_shared_memory / shared_mem_per_block,
    16  # 硬件限制
)
```

### 所需 NCU 指标

- `sm__warps_active.avg.pct_of_peak_sustained_active` - 实际占用率百分比

### 所需 Kernel 信息

- `block_size` - 线程块大小
- `registers_per_thread` - 每线程寄存器数
- `shared_mem_per_block` - 每块共享内存大小（字节）

### 实现示例

```python
# 实际占用率（从 NCU 直接获取）
achieved_occupancy = metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0)

# 理论占用率（基于资源限制计算）
block_size = kernel_info['block_size']
registers_per_thread = kernel_info['registers_per_thread']
shared_mem_per_block = kernel_info['shared_mem_per_block']

max_blocks_by_threads = max_threads_per_sm // block_size
max_blocks_by_regs = 65536 // (registers_per_thread * block_size)
max_blocks_by_smem = 49152 // shared_mem_per_block if shared_mem_per_block > 0 else 999

max_blocks = min(max_blocks_by_threads, max_blocks_by_regs, max_blocks_by_smem, 16)
theoretical_occupancy = (max_blocks * block_size / max_threads_per_sm) * 100
```

### 占用率限制因素

| 限制因素 | 说明 | 优化建议 |
|---------|------|---------|
| `registers` | 寄存器使用过多 | 减少局部变量、使用 `__launch_bounds__` |
| `shared_memory` | 共享内存使用过多 | 减少共享内存分配、分块处理 |
| `block_size` | 线程块大小不合适 | 调整 block_size（通常 128-512） |
| `none` | 无明显限制 | 占用率已接近理论最大值 |

### 解释指南

| 占用率 | 含义 | 优化建议 |
|-------|------|---------|
| > 70% | 高占用率，并行度充足 | 关注其他瓶颈（内存、计算） |
| 50-70% | 中等占用率 | 可能有优化空间，检查限制因素 |
| 30-50% | 较低占用率 | 优化资源使用，提高并行度 |
| < 30% | 很低占用率，严重限制性能 | 必须优化，检查限制因素 |

**注意**：高占用率不一定意味着高性能，但低占用率通常会限制性能（尤其是 memory-bound kernel）。

## 4. 计算吞吐量 (Compute Throughput)

### 定义

计算吞吐量表示 kernel 实际达到的浮点运算速度。

### 计算公式

```
achieved_tflops = total_flops / kernel_duration (TFLOPS)
compute_util = (achieved_tflops / peak_tflops) × 100%
```

### 实现示例

```python
duration_s = metrics['duration'] * 1e-9  # 转换为秒
achieved_tflops = (total_flops / duration_s) / 1e12
compute_util = (achieved_tflops * 1e12 / peak_flops_fp32) * 100
```

### 解释指南

| 计算利用率 | 含义 | 优化建议 |
|-----------|------|---------|
| > 80% | 计算单元饱和，可能是瓶颈 | 使用 Tensor Cores、优化指令混合 |
| 50-80% | 中等计算利用率 | 可能有优化空间 |
| < 50% | 较低计算利用率 | Kernel 可能是 memory-bound |

## 5. SM 效率 (SM Efficiency)

### 定义

SM 效率表示 SM 处于活跃状态的时间百分比。

### 所需 NCU 指标

- `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` - SM 活跃周期百分比

### 实现示例

```python
sm_efficiency = metrics.get('smsp__cycles_active.avg.pct_of_peak_sustained_elapsed', 0)
```

### 解释指南

| SM 效率 | 含义 | 可能原因 |
|--------|------|---------|
| > 80% | SM 持续工作 | 良好的并行度和资源利用 |
| 50-80% | SM 有空闲时间 | 可能有同步开销或负载不均衡 |
| < 50% | SM 大量空闲 | 并行度不足、占用率低、或频繁同步 |

## 6. Warp 执行效率 (Warp Execution Efficiency)

### 定义

Warp 执行效率表示 warp 中实际执行的线程数占总线程数的比例，反映线程分支（divergence）情况。

### 所需 NCU 指标

- `smsp__thread_inst_executed_per_inst_executed.ratio` - 每条指令执行的线程数比率

### 实现示例

```python
warp_efficiency = metrics.get('smsp__thread_inst_executed_per_inst_executed.ratio', 1.0) * 100
```

### 解释指南

| Warp 效率 | 含义 | 优化建议 |
|----------|------|---------|
| > 95% | 几乎无分支 | 良好的控制流 |
| 80-95% | 轻微分支 | 可接受，检查是否可优化 |
| < 80% | 严重分支 | 重构代码减少分支、使用 predication |

## 派生指标的综合使用

### 诊断流程

1. **检查带宽利用率和计算利用率**
   - 高带宽利用率 → memory-bound
   - 高计算利用率 → compute-bound
   - 两者都低 → 检查占用率

2. **检查占用率**
   - 低占用率 → 识别限制因素（寄存器/共享内存/block size）
   - 高占用率但性能差 → 检查其他指标

3. **检查算术强度**
   - 与 ridge point 比较，确定 memory/compute bound
   - 指导优化方向

4. **检查 SM 效率和 Warp 效率**
   - 低 SM 效率 → 并行度或同步问题
   - 低 Warp 效率 → 线程分支问题

### 示例诊断

**场景 1：Memory Bandwidth Bound**
```
bandwidth_util = 85%
compute_util = 30%
occupancy = 60%
arithmetic_intensity = 0.5 FLOPs/Byte
→ 诊断：内存带宽瓶颈
→ 建议：使用共享内存、提高算术强度、优化访问模式
```

**场景 2：Low Occupancy**
```
bandwidth_util = 25%
compute_util = 20%
occupancy = 15%
limiting_factor = registers
→ 诊断：占用率过低限制性能
→ 建议：减少寄存器使用、调整 block size
```

**场景 3：Compute Bound**
```
bandwidth_util = 40%
compute_util = 90%
occupancy = 75%
arithmetic_intensity = 15 FLOPs/Byte
→ 诊断：计算瓶颈
→ 建议：使用 Tensor Cores、优化指令混合
```

## 实现注意事项

1. **处理缺失指标**：某些 NCU 指标可能不可用，需要提供默认值
2. **单位转换**：注意时间单位（纳秒）、带宽单位（GB/s）、计算单位（TFLOPS）
3. **除零保护**：在计算比率时使用 `max(value, 1)` 避免除零
4. **GPU 架构差异**：不同 GPU 的资源限制不同（寄存器数、共享内存大小等）

## 参考资料

- NVIDIA Nsight Compute Documentation
- CUDA C Programming Guide - Occupancy Calculator
- Roofline Model Paper (Williams et al., 2009)
