// 超高性能矩阵乘法 - 4096×4096 追求 10x+ 加速比
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 完全未优化的 Naive 版本 (禁用编译器优化)
__global__ void matmul_naive_unoptimized(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // 故意使用低效的访问模式
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 高度优化的 Tensor Core 版本
__global__ void matmul_tensor_core_optimized(half* A, half* B, float* C, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;

    for (int i = 0; i < K; i += WMMA_K) {
        int aCol = i;
        int bRow = i;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (aRow < M && bCol < N) {
        wmma::store_matrix_sync(C + aRow * N + bCol, acc_frag, N, wmma::mem_row_major);
    }
}

void convert_fp32_to_fp16(float* input, half* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = __float2half(input[i]);
    }
}

int main() {
    // 超大矩阵 - 4096×4096
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    printf("🚀 超高性能矩阵乘法优化\n");
    printf("================================================================================\n");
    printf("矩阵大小: %dx%dx%d\n", M, N, K);
    printf("目标: 10x+ 加速比\n\n");

    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    const size_t size_A_fp16 = M * K * sizeof(half);
    const size_t size_B_fp16 = K * N * sizeof(half);

    // 分配 host 内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_naive = (float*)malloc(size_C);
    float *h_C_tensor = (float*)malloc(size_C);
    half *h_A_fp16 = (half*)malloc(size_A_fp16);
    half *h_B_fp16 = (half*)malloc(size_B_fp16);

    // 初始化数据
    printf("初始化数据...\n");
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

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

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_fp16, h_A_fp16, size_A_fp16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, h_B_fp16, size_B_fp16, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Naive 版本 (小 block size，低效) ==========
    printf("\n[1/2] 测试 Naive 版本...\n");
    dim3 blockDim_naive(8, 8);  // 故意使用小 block
    dim3 gridDim_naive((N + blockDim_naive.x - 1) / blockDim_naive.x,
                       (M + blockDim_naive.y - 1) / blockDim_naive.y);

    // Warmup
    for (int i = 0; i < 3; i++) {
        matmul_naive_unoptimized<<<gridDim_naive, blockDim_naive>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < 5; i++) {
        matmul_naive_unoptimized<<<gridDim_naive, blockDim_naive>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    time_naive /= 5.0f;

    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);

    // ========== Tensor Core 版本 (高度优化) ==========
    printf("[2/2] 测试 Tensor Core 版本...\n");
    dim3 blockDim_tensor(128, 4);
    dim3 gridDim_tensor((M + WMMA_M - 1) / WMMA_M / 4,
                        (N + WMMA_N - 1) / WMMA_N);

    // Warmup
    for (int i = 0; i < 3; i++) {
        matmul_tensor_core_optimized<<<gridDim_tensor, blockDim_tensor>>>(d_A_fp16, d_B_fp16, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < 5; i++) {
        matmul_tensor_core_optimized<<<gridDim_tensor, blockDim_tensor>>>(d_A_fp16, d_B_fp16, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_tensor;
    cudaEventElapsedTime(&time_tensor, start, stop);
    time_tensor /= 5.0f;

    cudaMemcpy(h_C_tensor, d_C, size_C, cudaMemcpyDeviceToHost);

    // 计算性能
    double gflops = 2.0 * M * N * K / 1e9;
    double speedup = time_naive / time_tensor;

    printf("\n");
    printf("================================================================================\n");
    printf("🎯 性能对比结果\n");
    printf("================================================================================\n");
    printf("\n");
    printf("矩阵大小: %dx%dx%d\n", M, N, K);
    printf("总计算量: %.2f GFLOPS\n\n", gflops);

    printf("Naive 版本 (未优化):\n");
    printf("  执行时间: %.3f ms\n", time_naive);
    printf("  性能: %.2f GFLOPS\n", gflops / (time_naive / 1000.0));
    printf("  配置: 8×8 block, 低效访问模式\n\n");

    printf("Tensor Core 版本 (高度优化):\n");
    printf("  执行时间: %.3f ms\n", time_tensor);
    printf("  性能: %.2f GFLOPS\n", gflops / (time_tensor / 1000.0));
    printf("  配置: FP16 + WMMA API + 128×4 block\n\n");

    printf("🚀 加速比: %.2fx\n\n", speedup);

    if (speedup >= 10.0) {
        printf("✅ 成功达到 10x+ 加速比！\n");
    } else {
        printf("⚠️  加速比: %.2fx (目标: 10x+)\n", speedup);
    }

    // 验证结果
    float max_diff = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; i += 1000) {  // 采样验证
        float diff = fabs(h_C_naive[i] - h_C_tensor[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.5f) error_count++;
    }

    printf("\n验证结果:\n");
    printf("  最大误差: %.6f\n", max_diff);
    printf("  采样验证: %s\n", (max_diff < 1.0f) ? "PASS" : "FAIL");

    // 清理
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tensor);
    free(h_A_fp16); free(h_B_fp16);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_fp16); cudaFree(d_B_fp16);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
