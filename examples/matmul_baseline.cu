// 完整的矩阵乘法 kernel - Baseline 版本
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Naive 矩阵乘法 kernel
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

int main() {
    // 矩阵大小
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const int size_A = M * K * sizeof(float);
    const int size_B = K * N * sizeof(float);
    const int size_C = M * N * sizeof(float);

    // 分配 host 内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // 初始化数据
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 拷贝数据到 device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 配置 kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    // Warmup
    for (int i = 0; i < 5; i++) {
        matmul_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Matrix size: %dx%dx%d\n", M, N, K);
    printf("Average time: %.3f ms\n", milliseconds / 10.0f);
    printf("GFLOPS: %.2f\n", (2.0 * M * N * K * 10) / (milliseconds * 1e6));

    // 拷贝结果回 host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 简单验证（检查第一个元素）
    float expected = 0.0f;
    for (int k = 0; k < K; k++) {
        expected += h_A[k] * h_B[k * N];
    }

    bool correct = fabs(h_C[0] - expected) < 1e-3;
    printf("Result: %s (h_C[0]=%.6f, expected=%.6f)\n",
           correct ? "PASS" : "FAIL", h_C[0], expected);

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
