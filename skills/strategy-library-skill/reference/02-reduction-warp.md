# Warp 级归约优化（Reduction with Warp Primitives）

## 概述

归约（Reduction）是将数组中的所有元素通过某种操作（如求和、求最大值）合并为单个值的过程。传统的归约实现依赖共享内存和原子操作，而 Warp 级归约使用 warp shuffle 指令，可以显著减少同步开销和内存访问。

**核心思想**：利用 warp 内线程可以直接交换寄存器数据的特性（`__shfl_down_sync`），在 warp 内完成归约，避免使用共享内存。

## 适用场景

### 算子类型
- ✅ 求和（Sum）
- ✅ 求最大值（Max）
- ✅ 求最小值（Min）
- ✅ 求平均值（Mean）
- ✅ 范数计算（Norm）
- ✅ 点积（Dot Product）

### 瓶颈类型
- ✅ 内存延迟瓶颈（共享内存原子操作慢）
- ✅ 低占用率（同步开销大）
- ⚠️ 内存带宽瓶颈（可以结合向量化）

### GPU 架构要求
- **最低要求**：Compute Capability 3.0+（支持 warp shuffle）
- **推荐**：Compute Capability 7.0+（更高效的 shuffle 指令）

## 优化原理

### 传统归约的问题

```cuda
// 传统归约：使用共享内存
__global__ void reduce_naive(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // 树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 每次迭代都需要同步
    }
    
    // 写回结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

**问题分析**：
- 需要共享内存存储中间结果
- 每次迭代都需要 `__syncthreads()`（开销大）
- 共享内存访问可能有 bank conflict
- 占用率受共享内存限制

### Warp Shuffle 优化

```cuda
// Warp 内归约：使用 shuffle 指令
__device__ float warp_reduce_sum(float val) {
    // warp size = 32
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_warp(float *input, float *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到寄存器
    float val = (tid < n) ? input[tid] : 0.0f;
    
    // Warp 内归约（无需同步）
    val = warp_reduce_sum(val);
    
    // 每个 warp 的第一个线程写回结果
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, val);  // 或使用共享内存进一步归约
    }
}
```

**优化效果**：
- ✅ 无需共享内存（节省资源）
- ✅ 无需 `__syncthreads()`（减少同步开销）
- ✅ 直接在寄存器中操作（更快）
- ✅ 无 bank conflict
- ✅ 更高的占用率

## 实现细节

### 基本 Warp 归约

```cuda
// Warp 内求和
__device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;  // 所有线程参与
    
    // 树形归约：16 -> 8 -> 4 -> 2 -> 1
    val += __shfl_down_sync(mask, val, 16);  // 0+16, 1+17, ..., 15+31
    val += __shfl_down_sync(mask, val, 8);   // 0+8, 1+9, ..., 7+15
    val += __shfl_down_sync(mask, val, 4);   // 0+4, 1+5, 2+6, 3+7
    val += __shfl_down_sync(mask, val, 2);   // 0+2, 1+3
    val += __shfl_down_sync(mask, val, 1);   // 0+1
    
    return val;  // 线程 0 持有最终结果
}

// Warp 内求最大值
__device__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;
    
    val = fmaxf(val, __shfl_down_sync(mask, val, 16));
    val = fmaxf(val, __shfl_down_sync(mask, val, 8));
    val = fmaxf(val, __shfl_down_sync(mask, val, 4));
    val = fmaxf(val, __shfl_down_sync(mask, val, 2));
    val = fmaxf(val, __shfl_down_sync(mask, val, 1));
    
    return val;
}

// Warp 内求最小值
__device__ float warp_reduce_min(float val) {
    unsigned mask = 0xffffffff;
    
    val = fminf(val, __shfl_down_sync(mask, val, 16));
    val = fminf(val, __shfl_down_sync(mask, val, 8));
    val = fminf(val, __shfl_down_sync(mask, val, 4));
    val = fminf(val, __shfl_down_sync(mask, val, 2));
    val = fminf(val, __shfl_down_sync(mask, val, 1));
    
    return val;
}
```

### Block 级归约

```cuda
// Block 级归约：多个 warp 的结果需要进一步归约
__global__ void reduce_block(float *input, float *output, int n) {
    __shared__ float warp_results[32];  // 最多 32 个 warp（1024 线程）
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;        // warp 内线程 ID
    int warp_id = threadIdx.x / 32;     // warp ID
    
    // 1. 加载数据
    float val = (tid < n) ? input[tid] : 0.0f;
    
    // 2. Warp 内归约
    val = warp_reduce_sum(val);
    
    // 3. 每个 warp 的第一个线程写入共享内存
    if (lane == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();
    
    // 4. 第一个 warp 归约所有 warp 的结果
    if (warp_id == 0) {
        val = (lane < (blockDim.x / 32)) ? warp_results[lane] : 0.0f;
        val = warp_reduce_sum(val);
        
        // 5. 线程 0 写回结果
        if (lane == 0) {
            output[blockIdx.x] = val;
        }
    }
}
```

### 多元素归约

```cuda
// 每个线程处理多个元素，减少 kernel 启动次数
#define ELEMENTS_PER_THREAD 4

__global__ void reduce_multi_element(float *input, float *output, int n) {
    __shared__ float warp_results[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // 1. 每个线程处理多个元素
    float sum = 0.0f;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * blockDim.x * gridDim.x;
        if (idx < n) {
            sum += input[idx];
        }
    }
    
    // 2. Warp 内归约
    sum = warp_reduce_sum(sum);
    
    // 3. Block 级归约
    if (lane == 0) {
        warp_results[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_results[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (lane == 0) {
            atomicAdd(output, sum);  // 多个 block 的结果累加
        }
    }
}
```

## 参数选择

### BLOCK_SIZE 选择规则

#### 规则 1：必须是 32 的倍数

```
BLOCK_SIZE = 32 × num_warps
```

**推荐值**：
- 128（4 个 warp）
- 256（8 个 warp）
- 512（16 个 warp）
- 1024（32 个 warp）

#### 规则 2：平衡占用率和效率

更大的 BLOCK_SIZE：
- ✅ 更高的占用率
- ✅ 更好地隐藏延迟
- ❌ 更多的 warp 间归约开销

**推荐**：256 或 512（平衡点）

### ELEMENTS_PER_THREAD 选择规则

**推荐值**：4 或 8

### 推荐配置

| 数组大小 | BLOCK_SIZE | ELEMENTS_PER_THREAD | 预期加速 |
|---------|------------|---------------------|---------|
| 小（<1M） | 256 | 4 | 3-5× |
| 中（1M-10M） | 512 | 4 | 5-7× |
| 大（>10M） | 512 | 8 | 6-8× |

## 性能分析

### 实际性能

**测试配置**：
- GPU: RTX 4090
- 数组大小: 16M 元素

**结果**：

| 实现 | 时间 (ms) | 带宽 (GB/s) | 加速比 |
|-----|----------|------------|--------|
| 传统归约 | 0.85 | 75 | 1.0× |
| Warp 归约 | 0.12 | 533 | 7.1× |
| Warp + 多元素 | 0.08 | 800 | 10.6× |

## 常见问题

### Q1: 为什么 warp shuffle 比共享内存快？

**A**: 
1. **无需内存访问**：shuffle 直接在寄存器间传输数据
2. **无需同步**：warp 内线程自动同步
3. **无 bank conflict**
4. **更低延迟**：寄存器访问 < 1 cycle

### Q2: __shfl_down_sync 的 mask 参数是什么？

**A**: 
- `mask` 指定哪些线程参与 shuffle
- `0xffffffff` 表示所有 32 个线程都参与

## 实战示例

### 完整代码

参见 `templates/reduction_warp.cu`

### 编译和运行

```bash
nvcc -O3 -arch=sm_89 reduction_warp.cu -o reduction_warp
./reduction_warp 16777216
```

## 参考资料

- **CUDA C Programming Guide - Warp Shuffle Functions**
- **Optimizing Parallel Reduction in CUDA**
- **CUB Library - Device-wide Reductions**

## 下一步

- 查看 CUDA 代码模板：`templates/reduction_warp.cu`
- 学习向量化优化：`03-vectorized-memory.md`
