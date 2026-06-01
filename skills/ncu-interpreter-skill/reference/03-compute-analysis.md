# Compute Subsystem Analysis

## Overview

计算子系统分析评估 CUDA kernel 的计算资源利用情况，包括占用率、SM 效率、Warp 执行效率和计算吞吐量。

## 分析维度

### 1. 占用率分析 (Occupancy Analysis)

#### 实际占用率 (Achieved Occupancy)

**NCU 指标**：
- `sm__warps_active.avg.pct_of_peak_sustained_active` - 实际活跃 warps 百分比

**含义**：SM 上实际活跃的 warps 占理论最大值的百分比

**获取方式**：
```python
achieved_occupancy = metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0)
```

#### 理论占用率 (Theoretical Occupancy)

**计算依据**：基于资源限制计算理论最大占用率

**资源限制**：
1. **线程数限制**：`max_threads_per_sm / block_size`
2. **寄存器限制**：`max_registers / (registers_per_thread × block_size)`
3. **共享内存限制**：`max_shared_memory / shared_mem_per_block`
4. **硬件限制**：最多 16 个 blocks per SM（Ampere 架构）

**计算公式**：
```python
max_blocks = min(
    max_threads_per_sm // block_size,
    65536 // (registers_per_thread * block_size),
    49152 // shared_mem_per_block if shared_mem_per_block > 0 else 999,
    16  # 硬件限制
)
theoretical_occupancy = (max_blocks * block_size / max_threads_per_sm) * 100
```

#### 占用率限制因素识别

**逻辑**：
```python
if max_blocks == max_blocks_by_regs:
    limiting_factor = 'registers'
elif max_blocks == max_blocks_by_smem:
    limiting_factor = 'shared_memory'
elif max_blocks == max_blocks_by_threads:
    limiting_factor = 'block_size'
else:
    limiting_factor = 'none'
```

**优化建议**：

| 限制因素 | 原因 | 优化策略 |
|---------|------|---------|
| `registers` | 每线程寄存器使用过多 | 1. 减少局部变量<br>2. 使用 `__launch_bounds__`<br>3. 降低循环展开程度 |
| `shared_memory` | 共享内存分配过多 | 1. 减少共享内存使用<br>2. 使用更小的 tile size<br>3. 分阶段处理数据 |
| `block_size` | 线程块大小不合适 | 1. 增加 block_size（128→256→512）<br>2. 平衡 block_size 和资源使用 |
| `none` | 无明显限制 | 占用率已接近最优 |

### 2. SM 效率 (SM Efficiency)

**NCU 指标**：
- `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` - SM 活跃周期百分比

**含义**：SM 处于活跃状态（执行指令）的时间占总时间的百分比

**获取方式**：
```python
sm_efficiency = metrics.get('smsp__cycles_active.avg.pct_of_peak_sustained_elapsed', 0)
```

**解释指南**：

| SM 效率 | 含义 | 可能原因 | 优化建议 |
|--------|------|---------|---------|
| > 80% | SM 持续工作 | 良好的并行度 | 保持当前状态 |
| 50-80% | SM 有空闲时间 | 同步开销、负载不均衡 | 减少 `__syncthreads()`、优化负载分配 |
| < 50% | SM 大量空闲 | 并行度不足、占用率低 | 提高占用率、增加并行度 |

### 3. Warp 执行效率 (Warp Execution Efficiency)

**NCU 指标**：
- `smsp__thread_inst_executed_per_inst_executed.ratio` - 每条指令执行的线程数比率

**含义**：Warp 中实际执行的线程占 32 个线程的比例，反映线程分支（divergence）情况

**获取方式**：
```python
warp_efficiency = metrics.get('smsp__thread_inst_executed_per_inst_executed.ratio', 1.0) * 100
```

**解释指南**：

| Warp 效率 | 含义 | 优化建议 |
|----------|------|---------|
| > 95% | 几乎无分支 | 良好的控制流 |
| 80-95% | 轻微分支 | 可接受，检查是否可优化 |
| 50-80% | 中等分支 | 考虑重构代码 |
| < 50% | 严重分支 | 必须优化，使用 predication 或重构 |

**线程分支示例**：
```cuda
// 导致分支的代码
__global__ void divergent_kernel(int* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        if (data[tid] > 0) {  // 分支：部分线程执行 if，部分执行 else
            data[tid] = data[tid] * 2;
        } else {
            data[tid] = data[tid] + 1;
        }
    }
}

// 优化：使用 predication 避免分支
__global__ void predicated_kernel(int* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int val = data[tid];
        int mask = val > 0;  // 0 或 1
        data[tid] = mask * (val * 2) + (1 - mask) * (val + 1);
    }
}
```

### 4. 计算吞吐量 (Compute Throughput)

**派生指标**：
- `achieved_tflops` - 实际达到的 TFLOPS
- `compute_util` - 计算利用率百分比

**计算公式**：
```python
duration_s = metrics['duration'] * 1e-9
achieved_tflops = (total_flops / duration_s) / 1e12
compute_util = (achieved_tflops * 1e12 / peak_flops_fp32) * 100
```

**解释指南**：

| 计算利用率 | 含义 | 优化建议 |
|-----------|------|---------|
| > 80% | 计算单元饱和 | 使用 Tensor Cores、优化指令混合 |
| 50-80% | 中等计算利用率 | 可能有优化空间 |
| < 50% | 较低计算利用率 | Kernel 可能是 memory-bound |

## 计算问题诊断

### 诊断规则实现

```python
def _analyze_compute(metrics, derived, kernel_info):
    """分析计算子系统性能"""
    analysis = {}
    
    # 实际占用率
    achieved_occ = metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0)
    analysis['achieved_occupancy'] = achieved_occ
    
    # 理论占用率
    if kernel_info:
        block_size = kernel_info.get('block_size', 256)
        registers_per_thread = kernel_info.get('registers_per_thread', 32)
        shared_mem_per_block = kernel_info.get('shared_mem_per_block', 0)
        
        max_blocks_by_threads = max_threads_per_sm // block_size
        max_blocks_by_regs = (65536 // (registers_per_thread * block_size)) if registers_per_thread > 0 else 999
        max_blocks_by_smem = (49152 // shared_mem_per_block) if shared_mem_per_block > 0 else 999
        
        max_blocks = min(max_blocks_by_threads, max_blocks_by_regs, max_blocks_by_smem, 16)
        theoretical_occ = (max_blocks * block_size / max_threads_per_sm) * 100
        analysis['theoretical_occupancy'] = min(theoretical_occ, 100)
        
        # 识别限制因素
        if max_blocks == max_blocks_by_regs:
            analysis['limiting_factor'] = 'registers'
        elif max_blocks == max_blocks_by_smem:
            analysis['limiting_factor'] = 'shared_memory'
        elif max_blocks == max_blocks_by_threads:
            analysis['limiting_factor'] = 'block_size'
        else:
            analysis['limiting_factor'] = 'none'
    else:
        analysis['theoretical_occupancy'] = 100
        analysis['limiting_factor'] = 'unknown'
    
    # SM 效率
    sm_active = metrics.get('smsp__cycles_active.avg.pct_of_peak_sustained_elapsed', 0)
    analysis['sm_efficiency'] = sm_active
    
    # Warp 执行效率
    if 'smsp__thread_inst_executed_per_inst_executed.ratio' in metrics:
        analysis['warp_efficiency'] = metrics['smsp__thread_inst_executed_per_inst_executed.ratio'] * 100
    
    # 计算利用率
    analysis['compute_util'] = derived['compute_util']
    
    # 诊断计算问题
    issues = []
    
    if achieved_occ < 30:
        issues.append(f'Low occupancy ({achieved_occ:.1f}%) - limited by {analysis["limiting_factor"]}')
    
    if sm_active < 50:
        issues.append('Low SM efficiency - SMs are idle for significant time')
    
    warp_eff = analysis.get('warp_efficiency', 100)
    if warp_eff < 80:
        issues.append('Low warp efficiency - thread divergence detected')
    
    if derived['compute_util'] > 80:
        issues.append('High compute utilization - kernel is compute-bound')
    
    analysis['issues'] = issues
    
    return analysis
```

## 常见计算问题及优化策略

### 问题 1：低占用率（< 30%）

**症状**：
- 实际占用率 < 30%
- 限制因素：registers/shared_memory/block_size

**优化策略**：

#### 寄存器限制
```cuda
// 优化前：寄存器使用过多
__global__ void high_register_kernel() {
    float a, b, c, d, e, f, g, h;  // 大量局部变量
    // ... 复杂计算
}

// 优化后：使用 __launch_bounds__ 限制寄存器
__global__ void __launch_bounds__(256, 4)  // 256 threads, 4 blocks per SM
optimized_kernel() {
    // 编译器会限制寄存器使用
}
```

#### 共享内存限制
```cuda
// 优化前：共享内存使用过多
__global__ void large_smem_kernel() {
    __shared__ float data[2048];  // 8KB 共享内存
    // ...
}

// 优化后：减少共享内存或分阶段处理
__global__ void small_smem_kernel() {
    __shared__ float data[512];  // 2KB 共享内存
    // 分多次处理
}
```

#### Block Size 不合适
```cuda
// 优化前：block size 太小
kernel<<<grid, 64>>>();  // 占用率低

// 优化后：增加 block size
kernel<<<grid, 256>>>();  // 更好的占用率
```

### 问题 2：低 SM 效率（< 50%）

**症状**：
- SM 效率 < 50%
- SM 大量空闲时间

**可能原因**：
1. 频繁的 `__syncthreads()`
2. 负载不均衡
3. 占用率过低

**优化策略**：
```cuda
// 优化前：频繁同步
__global__ void frequent_sync_kernel() {
    for (int i = 0; i < 100; i++) {
        // 少量计算
        __syncthreads();  // 频繁同步
    }
}

// 优化后：减少同步次数
__global__ void less_sync_kernel() {
    // 批量处理，减少同步
    for (int i = 0; i < 100; i += 10) {
        // 处理 10 次迭代
        __syncthreads();  // 同步次数减少 10 倍
    }
}
```

### 问题 3：线程分支（Warp 效率 < 80%）

**症状**：
- Warp 执行效率 < 80%
- 存在条件分支

**优化策略**：

#### 使用 Predication
```cuda
// 优化前：条件分支
if (condition) {
    result = compute_a();
} else {
    result = compute_b();
}

// 优化后：predication
int mask = condition;
result = mask * compute_a() + (1 - mask) * compute_b();
```

#### 重组数据
```cuda
// 优化前：混合数据导致分支
__global__ void mixed_data_kernel(int* types, float* data) {
    int tid = threadIdx.x;
    if (types[tid] == TYPE_A) {
        process_type_a(data[tid]);
    } else {
        process_type_b(data[tid]);
    }
}

// 优化后：分离数据，避免分支
__global__ void separated_kernel_a(float* data_a) {
    // 只处理 TYPE_A
}
__global__ void separated_kernel_b(float* data_b) {
    // 只处理 TYPE_B
}
```

### 问题 4：高计算利用率（> 80%）

**症状**：
- 计算利用率 > 80%
- Kernel 是 compute-bound

**优化策略**：
1. **使用 Tensor Cores**（FP16/BF16/INT8）
2. **优化指令混合**（减少低效指令）
3. **使用 CUDA Graphs**（减少启动开销）
4. **算法优化**（减少计算量）

## 计算分析输出示例

```python
{
    'achieved_occupancy': 45.2,       # 实际占用率 45.2%
    'theoretical_occupancy': 75.0,    # 理论占用率 75.0%
    'limiting_factor': 'registers',   # 限制因素：寄存器
    'sm_efficiency': 68.5,            # SM 效率 68.5%
    'warp_efficiency': 72.3,          # Warp 效率 72.3%
    'compute_util': 35.6,             # 计算利用率 35.6%
    'issues': [
        'Low occupancy (45.2%) - limited by registers',
        'Low warp efficiency - thread divergence detected'
    ]
}
```

## 参考资料

- CUDA C Programming Guide - Occupancy
- NVIDIA Nsight Compute - Compute Workload Analysis
- CUDA Best Practices Guide - Execution Configuration Optimization
