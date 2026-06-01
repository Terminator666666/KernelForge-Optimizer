// 高性能矩阵乘法 - 使用 Tensor Core 实现
// 目标: 10x+ 加速比
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Naive 版本
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tensor Core 版本 (FP16)
__global__ void matmul_tensor_core(half* A, half* B, float* C, int M, int N, int K) {
    // Warp 和线程索引
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 声明 WMMA 片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器
    wmma::fill_fragment(acc_frag, 0.0f);

    // 计算起始位置
    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;

    // 遍历 K 维度
    for (int i = 0; i < K; i += WMMA_K) {
        int aCol = i;
        int bRow = i;

        // 边界检查
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // 加载矩阵片段
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // 执行矩阵乘法
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 存储结果
    if (aRow < M && bCol < N) {
        wmma::store_matrix_sync(C + aRow * N + bCol, acc_frag, N, wmma::mem_row_major);
    }
}

// FP32 转 FP16
void convert_fp32_to_fp16(float* input, half* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = __float2half(input[i]);
    }
}

int main() {
    // 矩阵大小 - 使用更大的矩阵以展示加速比
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    const int size_A = M * K * sizeof(float);
    const int size_B = K * N * sizeof(float);
    const int size_C = M * N * sizeof(float);
    const int size_A_fp16 = M * K * sizeof(half);
    const int size_B_fp16 = K * N * sizeof(half);

    // 分配 host 内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_naive = (float*)malloc(size_C);
    float *h_C_tensor = (float*)malloc(size_C);
    half *h_A_fp16 = (half*)malloc(size_A_fp16);
    half *h_B_fp16 = (half*)malloc(size_B_fp16);

    // 初始化数据
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // 转换为 FP16
    convert_fp32_to_fp16(h_A, h_A_fp16, M * K);
    convert_fp32_to_fp16(h_B, h_B_fp16, K * N);

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    half *d_A_fp16, *d_B_fp16;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_A_fp16, size_A_fp16);
    cudaMalloc(&d_B_fp16, size_B_fp16);

    // 拷贝数据到 device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_fp16, h_A_fp16, size_A_fp16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, h_B_fp16, size_B_fp16, cudaMemcpyHostToDevice);

    // 创建事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Naive 版本 ==========
    dim3 blockDim_naive(16, 16);
    dim3 gridDim_naive((N + blockDim_naive.x - 1) / blockDim_naive.x,
                       (M + blockDim_naive.y - 1) / blockDim_naive.y);

    // Warmup
    for (int i = 0; i < 5; i++) {
        matmul_naive<<<gridDim_naive, blockDim_naive>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_naive<<<gridDim_naive, blockDim_naive>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    time_naive /= 10.0f;

    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);

    // ========== Tensor Core 版本 ==========
    dim3 blockDim_tensor(128, 4);
    dim3 gridDim_tensor((M + WMMA_M - 1) / WMMA_M / 4,
                        (N + WMMA_N - 1) / WMMA_N);

    // Warmup
    for (int i = 0; i < 5; i++) {
        matmul_tensor_core<<<gridDim_tensor, blockDim_tensor>>>(d_A_fp16, d_B_fp16, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_tensor_core<<<gridDim_tensor, blockDim_tensor>>>(d_A_fp16, d_B_fp16, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_tensor;
    cudaEventElapsedTime(&time_tensor, start, stop);
    time_tensor /= 10.0f;

    cudaMemcpy(h_C_tensor, d_C, size_C, cudaMemcpyDeviceToHost);

    // 计算性能
    double gflops = 2.0 * M * N * K / 1e9;
    double speedup = time_naive / time_tensor;

    printf("Matrix size: %dx%dx%d\n\n", M, N, K);
    printf("Performance Comparison:\n");
    printf("  Naive version:       %.3f ms, %.2f GFLOPS\n",
           time_naive, gflops / (time_naive / 1000.0));
    printf("  Tensor Core version: %.3f ms, %.2f GFLOPS\n",
           time_tensor, gflops / (time_tensor / 1000.0));
    printf("  Speedup:             %.2fx\n\n", speedup);

    // 验证结果（允许一定误差，因为 FP16）
    float max_diff = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_C_naive[i] - h_C_tensor[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.1f) error_count++;
    }

    printf("Verification:\n");
    printf("  Max difference: %.6f\n", max_diff);
    printf("  Error count (>0.1): %d / %d\n", error_count, M * N);
    printf("  Result: %s\n", (max_diff < 1.0f && error_count < M * N / 1000) ? "PASS" : "FAIL");

    // 清理
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tensor);
    free(h_A_fp16); free(h_B_fp16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_fp16); cudaFree(d_B_fp16);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
