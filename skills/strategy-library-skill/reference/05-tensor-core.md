# Tensor Core 优化

## 概述

Tensor Core 是 NVIDIA GPU 中专门用于加速矩阵运算的硬件单元，支持混合精度计算（FP16/BF16/INT8/FP8），可以显著提升矩阵乘法和卷积的性能。

**核心思想**：使用 Tensor Core 专用 API（WMMA）执行 16×16×16 或 8×8×4 的小矩阵乘法，性能比标准 CUDA Core 高 5-10 倍。

## 适用场景

### 算子类型
- ✅ 矩阵乘法（MatMul、GEMM）
- ✅ 卷积（Conv2D、Conv3D）
- ✅ 注意力机制（Attention）
- ✅ 全连接层（Linear）

### 瓶颈类型
- ✅ 计算能力瓶颈（compute-bound）
- ⚠️ 内存带宽瓶颈（效果有限，但仍有提升）

### GPU 架构要求
- **最低要求**：Compute Capability 7.0+（Volta）
- **推荐**：
  - Volta (7.0): FP16 Tensor Cores
  - Ampere (8.0): FP16/BF16/TF32/INT8
  - Ada Lovelace (8.9): 第 4 代 Tensor Cores
  - Hopper (9.0): FP8 支持

## 优化原理

### 标准 CUDA Core 计算

```cuda
// 标准 FP32 矩阵乘法
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i][k] * B[k][j];  // FP32 乘加
        }
        C[i][j] = sum;
    }
}
```

**性能**：
- RTX 4090: 82.6 TFLOPS (FP32)
- 每个 SM 每周期执行 128 个 FP32 FMA

### Tensor Core 计算

```cuda
// Tensor Core FP16 矩阵乘法
// C (FP32) = A (FP16) × B (FP16) + C (FP32)
// 一次计算 16×16×16 矩阵块
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**性能**：
- RTX 4090: 660.6 TFLOPS (FP16 Tensor Core)
- **加速比**：8× vs FP32
- 每个 Tensor Core 每周期执行 256 个 FP16 FMA

## 实现细节

### WMMA API 基础

```cuda
#include <mma.h>
using namespace nvcuda;

// 定义 fragment（寄存器中的矩阵块）
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// 加载数据到 fragment
wmma::load_matrix_sync(a_frag, A_ptr, lda);
wmma::load_matrix_sync(b_frag, B_ptr, ldb);
wmma::fill_fragment(c_frag, 0.0f);

// 执行矩阵乘法（使用 Tensor Core）
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储结果
wmma::store_matrix_sync(C_ptr, c_frag, ldc, wmma::mem_row_major);
```

### 完整矩阵乘法实现

```cuda
__global__ void matmul_wmma(
    half *A, half *B, float *C,
    int M, int N, int K
) {
    // Warp 和 lane ID
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // 定义 fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 遍历 K 维度
    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * 16;
        
        // 边界检查
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // 加载 A 和 B 的块
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Tensor Core 计算
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // 存储结果
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}
```

### 结合共享内存分块

```cuda
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define TILE_SIZE 128

__global__ void matmul_wmma_tiled(half *A, half *B, float *C, int M, int N, int K) {
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];
    
    // Fragment 定义
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 分块遍历
    for (int t = 0; t < K; t += TILE_SIZE) {
        // 协作加载 tile 到共享内存
        // ... (类似标准分块)
        
        __syncthreads();
        
        // 在 tile 内使用 WMMA
        for (int k = 0; k < TILE_SIZE; k += WMMA_K) {
            wmma::load_matrix_sync(a_frag, &As[warpM * WMMA_M][k], TILE_SIZE);
            wmma::load_matrix_sync(b_frag, &Bs[k][warpN * WMMA_N], TILE_SIZE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // 存储结果
    wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}
```

## 参数选择

### WMMA 尺寸选择

| GPU 架构 | 支持的尺寸 | 精度 | 推荐 |
|---------|-----------|------|------|
| Volta (7.0) | 16×16×16 | FP16 | 16×16×16 |
| Turing (7.5) | 16×16×16, 8×8×4 | FP16, INT8 | 16×16×16 |
| Ampere (8.0) | 16×16×16, 8×8×4 | FP16, BF16, TF32, INT8 | 16×16×16 |
| Ada (8.9) | 16×16×16 | FP16, BF16, INT8, FP8 | 16×16×16 |
| Hopper (9.0) | 16×16×16 | FP16, BF16, INT8, FP8 | 16×16×16 |

**推荐**：16×16×16（最通用）

### 精度选择

| 精度 | 加速比 | 精度损失 | 适用场景 |
|-----|--------|---------|---------|
| FP32 | 1× | 无 | 基准 |
| TF32 | 4-5× | 极小 | 训练（Ampere+） |
| FP16 | 8-10× | 小 | 训练+推理 |
| BF16 | 8-10× | 小 | 训练（更稳定） |
| INT8 | 16-20× | 中 | 推理 |
| FP8 | 20-30× | 中 | 推理（Hopper+） |

**推荐**：
- 训练：FP16 或 BF16
- 推理：INT8 或 FP8

### 矩阵维度要求

**约束**：
- M, N, K 必须是 WMMA 尺寸的倍数
- 例如：16×16×16 要求 M, N, K 都是 16 的倍数

**处理非对齐**：
- Padding：填充到对齐尺寸
- 混合：对齐部分用 WMMA，剩余用标准 CUDA

## 性能分析

### 理论性能

**RTX 4090**：
- FP32: 82.6 TFLOPS
- FP16 Tensor Core: 660.6 TFLOPS
- **理论加速**：8×

**H100**：
- FP32: 67 TFLOPS
- FP16 Tensor Core: 1979 TFLOPS
- FP8 Tensor Core: 3958 TFLOPS
- **理论加速**：30× (FP16), 59× (FP8)

### 实际性能

**测试配置**：
- GPU: RTX 4090
- 矩阵: 4096×4096 × 4096×4096

**结果**：

| 实现 | 时间 (ms) | TFLOPS | 加速比 |
|-----|----------|--------|--------|
| FP32 朴素 | 285 | 0.48 | 1.0× |
| FP32 分块 | 42 | 3.26 | 6.8× |
| FP16 WMMA | 8.5 | 16.1 | 33.5× |
| FP16 WMMA + 分块 | 5.2 | 26.3 | 54.8× |
| cuBLAS FP16 | 4.8 | 28.5 | 59.4× |

## 混合精度训练

### 基本流程

```cuda
// 1. 权重以 FP32 存储
float *weights_fp32;

// 2. 转换为 FP16 用于前向传播
half *weights_fp16;
convert_fp32_to_fp16<<<grid, block>>>(weights_fp32, weights_fp16, n);

// 3. 使用 Tensor Core 计算
matmul_wmma<<<grid, block>>>(input_fp16, weights_fp16, output_fp32, M, N, K);

// 4. 梯度累加使用 FP32
// 5. 权重更新使用 FP32
```

### Loss Scaling

```cuda
// 防止梯度下溢
float loss_scale = 1024.0f;

// 前向传播
forward(input, output);

// 损失缩放
loss = loss * loss_scale;

// 反向传播
backward(loss);

// 梯度缩放回来
scale_gradients<<<grid, block>>>(gradients, 1.0f / loss_scale, n);

// 更新权重
update_weights(weights, gradients, lr);
```

## 常见问题

### Q1: Tensor Core 适用于所有矩阵乘法吗？

**A**: 不一定
- ✅ 大矩阵（>512）：效果显著
- ⚠️ 中等矩阵（128-512）：有提升
- ❌ 小矩阵（<128）：可能不如 FP32

### Q2: FP16 会损失精度吗？

**A**: 
- 训练：使用混合精度 + loss scaling，精度损失极小
- 推理：大多数模型可以直接使用 FP16，精度损失 <1%

### Q3: 如何选择 FP16 还是 BF16？

**A**: 
- **FP16**：更高精度（10-bit 尾数），适合大多数场景
- **BF16**：更大范围（8-bit 指数），数值更稳定，适合训练

### Q4: 矩阵维度不是 16 的倍数怎么办？

**A**: 
```cuda
// 方法 1：Padding
int M_padded = ((M + 15) / 16) * 16;

// 方法 2：混合实现
// 对齐部分用 WMMA，剩余用标准 CUDA
```

## 实战示例

### 完整代码

参见 `templates/tensor_core.cu`

### 编译和运行

```bash
# 需要 Volta+ GPU
nvcc -O3 -arch=sm_89 tensor_core.cu -o tensor_core
./tensor_core 4096 4096 4096
```

### 预期结果

```
Matrix size: 4096 x 4096 x 4096
FP32 Tiled:  42.0 ms, 3.26 TFLOPS
FP16 WMMA:   5.2 ms, 26.3 TFLOPS
Speedup: 8.1x
```

## 参考资料

- **CUDA C Programming Guide - Warp Matrix Functions**
- **NVIDIA Tensor Core Programming Guide**
- **Mixed Precision Training**: https://arxiv.org/abs/1710.03740
- **cuBLAS Documentation**

## 下一步

- 查看 CUDA 代码模板：`templates/tensor_core.cu`
- 学习混合精度训练技术
- 了解 cuBLAS 和 cuDNN 的 Tensor Core 支持
