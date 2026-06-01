# Memory Subsystem Analysis

## Overview

内存子系统分析评估 CUDA kernel 的内存访问性能，包括带宽利用率、缓存命中率、访问效率和访问模式。这是性能优化的关键环节，因为大多数 CUDA kernel 都是 memory-bound。

## 分析维度

### 1. 带宽利用率 (Bandwidth Utilization)

**目的**：判断内存带宽是否成为瓶颈

**关键指标**：
- 实际带宽 vs 理论峰值带宽
- 带宽利用率百分比

**诊断规则**：
```python
if bandwidth_util > 80:
    # 内存带宽饱和，是主要瓶颈
    suggestion = "减少内存访问、使用共享内存、提高算术强度"
elif bandwidth_util < 30:
    # 带宽利用率低，可能是其他瓶颈
    suggestion = "检查计算瓶颈或占用率问题"
```

### 2. 缓存命中率 (Cache Hit Rates)

#### L1 Cache Hit Rate

**NCU 指标**：
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` - L1 load 请求总数
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit.sum` - L1 load 命中数

**计算公式**：
```python
l1_requests = metrics['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']
l1_hits = metrics.get('l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit.sum', 0)
l1_hit_rate = (l1_hits / max(l1_requests, 1)) * 100
```

**解释指南**：

| L1 命中率 | 含义 | 优化建议 |
|----------|------|---------|
| > 80% | 良好的数据局部性 | 保持当前访问模式 |
| 50-80% | 中等局部性 | 考虑优化访问模式 |
| < 50% | 差的数据局部性 | 重组数据访问、使用共享内存 |

#### L2 Cache Hit Rate

**NCU 指标**：
- `lts__t_sectors_op_read.sum` - L2 读请求总数
- `lts__t_sectors_op_read_hit.sum` - L2 读命中数

**计算公式**：
```python
l2_requests = metrics['lts__t_sectors_op_read.sum']
l2_hits = metrics.get('lts__t_sectors_op_read_hit.sum', 0)
l2_hit_rate = (l2_hits / max(l2_requests, 1)) * 100
```

**解释指南**：

| L2 命中率 | 含义 | 优化建议 |
|----------|------|---------|
| > 70% | 良好的全局数据重用 | 保持当前模式 |
| 40-70% | 中等数据重用 | 考虑数据分块、提高重用 |
| < 40% | 差的数据重用 | 重组算法、增加数据重用 |

### 3. 内存访问效率 (Memory Access Efficiency)

#### Global Load Efficiency

**NCU 指标**：
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` - 全局内存 load 效率

**含义**：实际请求的数据占传输数据的百分比

**计算示例**：
```python
load_efficiency = metrics.get('smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct', 100)
```

**解释指南**：

| Load 效率 | 访问模式 | 优化建议 |
|----------|---------|---------|
| > 80% | Coalesced（合并访问） | 良好，保持 |
| 50-80% | Strided（跨步访问） | 考虑重组数据布局 |
| 25-50% | Mixed（混合模式） | 分析访问模式，局部优化 |
| < 25% | Random（随机访问） | 严重问题，必须重构 |

#### Global Store Efficiency

**NCU 指标**：
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` - 全局内存 store 效率

**计算示例**：
```python
store_efficiency = metrics.get('smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct', 100)
```

**解释指南**：与 load efficiency 类似，但 store 操作通常更容易优化（因为是顺序写入）。

### 4. 内存访问模式识别 (Access Pattern Detection)

**目的**：自动识别 kernel 的内存访问模式

**识别逻辑**：
```python
load_eff = analysis.get('load_efficiency', 100)
store_eff = analysis.get('store_efficiency', 100)
avg_eff = (load_eff + store_eff) / 2

if avg_eff > 80:
    access_pattern = 'coalesced'  # 合并访问
elif avg_eff > 50:
    access_pattern = 'strided'    # 跨步访问
elif avg_eff > 25:
    access_pattern = 'mixed'      # 混合模式
else:
    access_pattern = 'random'     # 随机访问
```

**访问模式特征**：

#### Coalesced Access（合并访问）
- **特征**：连续线程访问连续内存地址
- **效率**：> 80%
- **示例**：`data[tid]`，其中 `tid = threadIdx.x + blockIdx.x * blockDim.x`
- **性能**：最优，单次内存事务

#### Strided Access（跨步访问）
- **特征**：连续线程访问固定间隔的内存地址
- **效率**：50-80%
- **示例**：`data[tid * stride]`，stride > 1
- **性能**：中等，多次内存事务

#### Mixed Access（混合模式）
- **特征**：部分合并、部分跨步
- **效率**：25-50%
- **示例**：矩阵转置、某些归约操作
- **性能**：较差，需要优化

#### Random Access（随机访问）
- **特征**：线程访问不规则的内存地址
- **效率**：< 25%
- **示例**：`data[indices[tid]]`，indices 无序
- **性能**：很差，必须优化

## 内存问题诊断

### 诊断规则实现

```python
def _analyze_memory(metrics, derived):
    """分析内存子系统性能"""
    analysis = {
        'bandwidth_util': derived['bandwidth_util'],
        'achieved_bandwidth': derived['achieved_bandwidth']
    }
    
    # 计算缓存命中率
    if 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum' in metrics:
        l1_requests = metrics['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']
        l1_hits = metrics.get('l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit.sum', 0)
        analysis['l1_hit_rate'] = (l1_hits / max(l1_requests, 1)) * 100
    
    if 'lts__t_sectors_op_read.sum' in metrics:
        l2_requests = metrics['lts__t_sectors_op_read.sum']
        l2_hits = metrics.get('lts__t_sectors_op_read_hit.sum', 0)
        analysis['l2_hit_rate'] = (l2_hits / max(l2_requests, 1)) * 100
    
    # 计算访问效率
    if 'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct' in metrics:
        analysis['load_efficiency'] = metrics['smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct']
    
    if 'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct' in metrics:
        analysis['store_efficiency'] = metrics['smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct']
    
    # 识别访问模式
    load_eff = analysis.get('load_efficiency', 100)
    store_eff = analysis.get('store_efficiency', 100)
    avg_eff = (load_eff + store_eff) / 2
    
    if avg_eff > 80:
        analysis['access_pattern'] = 'coalesced'
    elif avg_eff > 50:
        analysis['access_pattern'] = 'strided'
    elif avg_eff > 25:
        analysis['access_pattern'] = 'mixed'
    else:
        analysis['access_pattern'] = 'random'
    
    # 诊断内存问题
    issues = []
    
    if analysis['bandwidth_util'] < 30:
        issues.append('Low bandwidth utilization - kernel may be compute-bound or have insufficient parallelism')
    elif analysis['bandwidth_util'] > 80:
        issues.append('High bandwidth utilization - memory bandwidth is likely the bottleneck')
    
    if load_eff < 50:
        issues.append('Poor load efficiency - memory accesses are not coalesced')
    if store_eff < 50:
        issues.append('Poor store efficiency - memory writes are not coalesced')
    
    l1_hit = analysis.get('l1_hit_rate', 100)
    if l1_hit < 50:
        issues.append('Low L1 cache hit rate - poor data locality')
    
    analysis['issues'] = issues
    
    return analysis
```

## 常见内存问题及优化策略

### 问题 1：高带宽利用率（> 80%）

**症状**：
- 带宽利用率 > 80%
- 性能受内存带宽限制

**优化策略**：
1. **使用共享内存**：缓存频繁访问的数据
2. **提高算术强度**：增加计算量，减少内存访问
3. **Kernel Fusion**：合并多个 kernel，减少中间数据传输
4. **数据压缩**：使用低精度数据类型（FP16, INT8）

**代码示例**：
```cuda
// 优化前：直接访问全局内存
__global__ void naive_kernel(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid];
        // 多次访问 data[tid]
        result = val * val + val;
    }
}

// 优化后：使用共享内存
__global__ void optimized_kernel(float* data, int n) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 加载到共享内存
    if (tid < n) {
        shared_data[threadIdx.x] = data[tid];
    }
    __syncthreads();
    
    // 从共享内存读取
    if (tid < n) {
        float val = shared_data[threadIdx.x];
        result = val * val + val;
    }
}
```

### 问题 2：低访问效率（< 50%）

**症状**：
- Load/Store 效率 < 50%
- 访问模式为 strided 或 random

**优化策略**：
1. **重组数据布局**：从 AoS 转换为 SoA
2. **调整访问模式**：确保连续线程访问连续内存
3. **使用 Padding**：避免 bank conflicts
4. **向量化访问**：使用 float4, int4 等

**代码示例**：
```cuda
// 优化前：跨步访问
__global__ void strided_access(float* data, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid * stride];  // 跨步访问
    }
}

// 优化后：合并访问
__global__ void coalesced_access(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid];  // 连续访问
    }
}
```

### 问题 3：低缓存命中率（< 50%）

**症状**：
- L1/L2 命中率 < 50%
- 数据局部性差

**优化策略**：
1. **数据分块（Tiling）**：提高数据重用
2. **重组计算顺序**：增加时间局部性
3. **使用共享内存**：显式管理数据缓存
4. **预取数据**：提前加载即将使用的数据

**代码示例**：
```cuda
// 优化前：无数据重用
__global__ void no_reuse(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];  // 每次从全局内存读取
    }
    C[row * N + col] = sum;
}

// 优化后：使用共享内存分块
__global__ void tiled_matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 加载 tile 到共享内存
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // 从共享内存计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

### 问题 4：内存延迟瓶颈

**症状**：
- 带宽利用率低（< 40%）
- 占用率低（< 50%）
- 访问模式为 random 或 mixed

**优化策略**：
1. **提高占用率**：增加并行度以隐藏延迟
2. **优化访问模式**：改善内存合并
3. **使用纹理内存**：利用硬件缓存
4. **异步内存拷贝**：重叠计算和内存传输

## 内存分析输出示例

```python
{
    'bandwidth_util': 75.3,           # 带宽利用率 75.3%
    'achieved_bandwidth': 756.8,      # 实际带宽 756.8 GB/s
    'l1_hit_rate': 45.2,              # L1 命中率 45.2%
    'l2_hit_rate': 68.5,              # L2 命中率 68.5%
    'load_efficiency': 62.3,          # Load 效率 62.3%
    'store_efficiency': 78.9,         # Store 效率 78.9%
    'access_pattern': 'strided',      # 跨步访问模式
    'issues': [
        'High bandwidth utilization - memory bandwidth is likely the bottleneck',
        'Low L1 cache hit rate - poor data locality'
    ]
}
```

## 参考资料

- NVIDIA Nsight Compute Documentation - Memory Workload Analysis
- CUDA C Best Practices Guide - Memory Optimization
- GPU Performance Analysis and Optimization (NVIDIA GTC)
