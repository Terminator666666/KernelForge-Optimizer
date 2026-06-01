// Reduction 优化 - 真实实现
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Naive Reduction
__global__ void reduce_naive(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += input[i];
        }
        *output = sum;
    }
}

// Optimized Reduction with Warp Shuffle
__global__ void reduce_optimized(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load and add two elements per thread
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}

int main() {
    const int N = 16 * 1024 * 1024; // 16M elements
    const int size = N * sizeof(float);

    printf("🚀 Reduction 优化测试\n");
    printf("================================================================================\n");
    printf("数据大小: %d 个元素 (%.2f MB)\n\n", N, size / 1024.0 / 1024.0);

    // Allocate memory
    float *h_input = (float*)malloc(size);
    float *h_output_naive = (float*)malloc(sizeof(float));
    float *h_output_opt = (float*)malloc(sizeof(float));

    // Initialize
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Naive Version ==========
    printf("[1/2] 测试 Naive Reduction...\n");

    cudaMemset(d_output, 0, sizeof(float));

    // Warmup
    for (int i = 0; i < 3; i++) {
        reduce_naive<<<1, 1>>>(d_input, d_output, N);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_output, 0, sizeof(float));
        reduce_naive<<<1, 1>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    time_naive /= 10.0f;

    cudaMemcpy(h_output_naive, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // ========== Optimized Version ==========
    printf("[2/2] 测试 Optimized Reduction...\n");

    int blockSize = 256;
    int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);

    cudaMemset(d_output, 0, sizeof(float));

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemset(d_output, 0, sizeof(float));
        reduce_optimized<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_output, 0, sizeof(float));
        reduce_optimized<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_opt;
    cudaEventElapsedTime(&time_opt, start, stop);
    time_opt /= 10.0f;

    cudaMemcpy(h_output_opt, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Results
    float speedup = time_naive / time_opt;
    float bandwidth_naive = (N * sizeof(float) / 1e9) / (time_naive / 1000.0);
    float bandwidth_opt = (N * sizeof(float) / 1e9) / (time_opt / 1000.0);

    printf("\n");
    printf("================================================================================\n");
    printf("🎯 性能对比结果\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Naive Reduction:\n");
    printf("  执行时间: %.3f ms\n", time_naive);
    printf("  带宽: %.2f GB/s\n", bandwidth_naive);
    printf("  结果: %.0f\n\n", *h_output_naive);

    printf("Optimized Reduction:\n");
    printf("  执行时间: %.3f ms\n", time_opt);
    printf("  带宽: %.2f GB/s\n", bandwidth_opt);
    printf("  结果: %.0f\n\n", *h_output_opt);

    printf("🚀 加速比: %.2fx\n\n", speedup);

    // Verification
    float expected = (float)N;
    float error_naive = fabs(*h_output_naive - expected) / expected;
    float error_opt = fabs(*h_output_opt - expected) / expected;

    printf("验证结果:\n");
    printf("  期望值: %.0f\n", expected);
    printf("  Naive 误差: %.6f%%\n", error_naive * 100);
    printf("  Optimized 误差: %.6f%%\n", error_opt * 100);
    printf("  状态: %s\n", (error_opt < 0.01) ? "PASS" : "FAIL");

    // Cleanup
    free(h_input);
    free(h_output_naive);
    free(h_output_opt);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
