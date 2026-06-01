# Memory Access Patterns

## Overview

内存访问模式（Memory Access Pattern）描述了 CUDA kernel 中线程访问全局内存的方式。访问模式直接影响内存效率和性能，是优化的关键因素。

## 访问模式类型

### 1. Coalesced Access（合并访问）

**定义**：连续的线程访问连续的内存地址，GPU 可以将多个访问合并为少量内存事务。

**特征**：
- Load/Store 效率 > 80%
- 内存带宽利用率高
- 最优的访问模式

**代码示例**：
```cuda
__global__ void coalesced_access(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid];  // 连续线程访问连续地址
        // Thread 0 访问 data[0]
        // Thread 1 访问 data[1]
        // Thread 2 访问 data[2]
        // ...
    }
}
```

**内存事务**：
- 32 个线程（1 个 warp）访问连续的 128 字节
- GPU 合并为 1 次 128-byte 内存事务
- 效率：100%

**适用场景**：
- 向量加法、乘法
- 矩阵的行访问
- 顺序数据处理

### 2. Strided Access（跨步访问）

**定义**：连续的线程访问固定间隔（stride）的内存地址。

**特征**：
- Load/Store 效率 50-80%
- 需要多次内存事务
- 性能中等

**代码示例**：
```cuda
__global__ void strided_access(float* data, int stride, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[tid * stride];  // 跨步访问
        // Thread 0 访问 data[0]
        // Thread 1 访问 data[stride]
        // Thread 2 访问 data[2*stride]
        // ...
    }
}
```

**内存事务**（stride = 2）：
- 32 个线程访问 256 字节（每个线程 4 字节，间隔 4 字节）
- GPU 需要 2 次 128-byte 内存事务
- 效率：50%

**常见场景**：
- 矩阵转置
- 列主序访问
- AoS（Array of Structures）数据布局

### 3. Mixed Access（混合模式）

**定义**：部分线程合并访问，部分跨步访问，访问模式不规则。

**特征**：
- Load/Store 效率 25-50%
- 内存事务数量多
- 性能较差

**代码示例**：
```cuda
__global__ void mixed_access(float* A, float* B, int* indices, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        // 部分合并访问
        float a = A[tid];
        // 部分跨步访问
        float b = B[indices[tid] * 2];
    }
}
```

**适用场景**：
- 复杂的数据结构
- 部分规则的访问模式
- 某些归约操作

### 4. Random Access（随机访问）

**定义**：线程访问不规则、不连续的内存地址，无法预测访问模式。

**特征**：
- Load/Store 效率 < 25%
- 每个线程可能需要单独的内存事务
- 性能很差

**代码示例**：
```cuda
__global__ void random_access(float* data, int* indices, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int idx = indices[tid];  // 随机索引
        float val = data[idx];   // 随机访问
        // Thread 0 访问 data[1523]
        // Thread 1 访问 data[42]
        // Thread 2 访问 data[9876]
        // ...
    }
}
```

**内存事务**：
- 32 个线程可能需要 32 次独立的内存事务
- 效率：< 10%

**常见场景**：
- 稀疏矩阵操作
- 图算法（邻接表）
- 哈希表查找
- 间接寻址

## 访问模式识别

### 自动识别算法

NCU Interpreter 通过分析 Load/Store 效率自动识别访问模式：

```python
def detect_access_pattern(memory_analysis):
    """识别内存访问模式"""
    load_eff = memory_analysis.get('load_efficiency', 100)
    store_eff = memory_analysis.get('store_efficiency', 100)
    avg_eff = (load_eff + store_eff) / 2
    
    if avg_eff > 80:
        return 'coalesced'
    elif avg_eff > 50:
        return 'strided'
    elif avg_eff > 25:
        return 'mixed'
    else:
        return 'random'
```

### 识别阈值

| 平均效率 | 访问模式 | 内存事务效率 |
|---------|---------|-------------|
| > 80% | Coalesced | 优秀（1-2 次事务） |
| 50-80% | Strided | 良好（2-4 次事务） |
| 25-50% | Mixed | 较差（4-16 次事务） |
| < 25% | Random | 很差（16-32 次事务） |

## 访问模式优化

### 优化 1：Coalesced → 保持

**当前状态**：已经是最优访问模式

**建议**：
- 保持当前访问模式
- 关注其他瓶颈（带宽、计算、占用率）

### 优化 2：Strided → Coalesced

**策略 1：数据布局转换（AoS → SoA）**

```cuda
// 优化前：AoS (Array of Structures)
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
Particle particles[N];

__global__ void aos_kernel(Particle* particles, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float x = particles[tid].x;  // Strided access (stride = 24 bytes)
    }
}

// 优化后：SoA (Structure of Arrays)
struct ParticlesSoA {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
};

__global__ void soa_kernel(ParticlesSoA particles, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float x = particles.x[tid];  // Coalesced access
    }
}
```

**策略 2：转置数据**

```cuda
// 优化前：列主序访问（Strided）
__global__ void column_access(float* matrix, int rows, int cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int col = tid % cols;
    int row = tid / cols;
    if (row < rows && col < cols) {
        float val = matrix[col * rows + row];  // Strided
    }
}

// 优化后：转置矩阵，改为行主序访问（Coalesced）
__global__ void row_access(float* matrix_transposed, int rows, int cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row = tid / cols;
    int col = tid % cols;
    if (row < rows && col < cols) {
        float val = matrix_transposed[row * cols + col];  // Coalesced
    }
}
```

**策略 3：使用共享内存**

```cuda
// 矩阵转置：使用共享内存优化
__global__ void transpose_shared(float* input, float* output, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Coalesced read from input
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();
    
    // Coalesced write to output (transposed)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    if (x < N && y < N) {
        output[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### 优化 3：Mixed → Strided/Coalesced

**策略**：分析访问模式，分离规则和不规则访问

```cuda
// 优化前：混合访问
__global__ void mixed_kernel(float* A, float* B, int* indices, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float a = A[tid];              // Coalesced
        float b = B[indices[tid]];     // Random
        float c = C[tid * 2];          // Strided
    }
}

// 优化后：分离访问，使用共享内存
__global__ void optimized_kernel(float* A, float* B, int* indices, int n) {
    __shared__ float shared_B[BLOCK_SIZE];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 1. Coalesced load A
    float a = A[tid];
    
    // 2. Load B indices to shared memory
    int idx = indices[tid];
    shared_B[threadIdx.x] = B[idx];
    __syncthreads();
    
    // 3. Access from shared memory
    float b = shared_B[threadIdx.x];
}
```

### 优化 4：Random → Mixed/Strided

**策略 1：数据重组**

```cuda
// 优化前：随机访问
__global__ void random_kernel(float* data, int* indices, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data[indices[tid]];  // Random
    }
}

// 优化后：预先排序索引，改善局部性
// 1. 在 CPU 或 GPU 上对 indices 排序
// 2. 重组数据以匹配访问模式
__global__ void sorted_kernel(float* data_sorted, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        float val = data_sorted[tid];  // Coalesced
    }
}
```

**策略 2：使用共享内存缓存**

```cuda
// 稀疏矩阵乘法：使用共享内存缓存随机访问
__global__ void sparse_matmul(float* values, int* col_indices, 
                               float* x, float* y, int nnz) {
    __shared__ float shared_x[BLOCK_SIZE];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 预加载 x 到共享内存
    if (threadIdx.x < BLOCK_SIZE) {
        shared_x[threadIdx.x] = x[threadIdx.x];
    }
    __syncthreads();
    
    if (tid < nnz) {
        int col = col_indices[tid];
        float val = values[tid];
        // 从共享内存读取（如果在范围内）
        float x_val = (col < BLOCK_SIZE) ? shared_x[col] : x[col];
        atomicAdd(&y[tid], val * x_val);
    }
}
```

**策略 3：使用纹理内存**

```cuda
// 使用纹理内存优化随机访问（只读数据）
texture<float, 1, cudaReadModeElementType> tex_data;

__global__ void texture_kernel(int* indices, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int idx = indices[tid];
        float val = tex1Dfetch(tex_data, idx);  // 纹理缓存优化随机访问
        output[tid] = val;
    }
}
```

## 访问模式诊断

### 诊断输出示例

**Coalesced Access**：
```
Memory Access Pattern: Coalesced
- Load efficiency: 95%
- Store efficiency: 92%
- Average efficiency: 93.5%

Analysis: Excellent memory access pattern
Recommendation: Maintain current access pattern, focus on other bottlenecks
```

**Strided Access**：
```
Memory Access Pattern: Strided
- Load efficiency: 65%
- Store efficiency: 58%
- Average efficiency: 61.5%

Analysis: Strided access detected, moderate efficiency
Recommendation:
1. Consider data layout transformation (AoS → SoA)
2. Use shared memory for data reuse
3. Transpose data if accessing columns
```

**Mixed Access**：
```
Memory Access Pattern: Mixed
- Load efficiency: 42%
- Store efficiency: 38%
- Average efficiency: 40%

Analysis: Mixed access pattern, poor efficiency
Recommendation:
1. Analyze and separate regular vs irregular accesses
2. Use shared memory to cache irregular accesses
3. Reorganize computation to improve locality
```

**Random Access**：
```
Memory Access Pattern: Random
- Load efficiency: 18%
- Store efficiency: 15%
- Average efficiency: 16.5%

Analysis: Random access detected, very poor efficiency
Recommendation:
1. Sort or reorganize data to improve locality
2. Use shared memory to cache frequently accessed data
3. Consider using texture memory for read-only data
4. Batch similar accesses together
```

## 访问模式与性能

### 性能影响

| 访问模式 | 内存事务数 | 相对性能 | 优化优先级 |
|---------|-----------|---------|-----------|
| Coalesced | 1-2× | 100% | 低（已优化） |
| Strided | 2-4× | 50-80% | 中 |
| Mixed | 4-16× | 25-50% | 高 |
| Random | 16-32× | 10-25% | 很高 |

### 优化收益估算

**Strided → Coalesced**：
- 性能提升：1.5-2×
- 实现难度：中等
- 推荐优先级：高

**Mixed → Strided**：
- 性能提升：1.2-1.5×
- 实现难度：中等到高
- 推荐优先级：中

**Random → Mixed**：
- 性能提升：1.5-3×
- 实现难度：高
- 推荐优先级：很高

## 参考资料

- CUDA C Best Practices Guide - Coalesced Access to Global Memory
- NVIDIA Nsight Compute - Memory Workload Analysis
- GPU Performance Analysis and Optimization (NVIDIA GTC)
