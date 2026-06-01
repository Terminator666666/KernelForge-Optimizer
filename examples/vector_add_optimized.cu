// Vector Add 优化
#include <cuda_runtime.h>
#include <stdio.h>

// Naive Vector Add
__global__ void vector_add_naive(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Optimized Vector Add - 向量化访问
__global__ void vector_add_optimized(float* a, float* b, float* c, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 a4 = *((float4*)&a[i]);
        float4 b4 = *((float4*)&b[i]);
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        *((float4*)&c[i]) = c4;
    }
}

int main() {
    const int N = 32 * 1024 * 1024; // 32M elements
    const int size = N * sizeof(float);

    printf("🚀 Vector Add 优化测试\n");
    printf("================================================================================\n");
    printf("数据大小: %d 个元素 (%.2f MB)\n\n", N, size / 1024.0 / 1024.0);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Naive
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        vector_add_naive<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    time_naive /= 100.0f;

    // Optimized
    int gridSize_opt = (N / 4 + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        vector_add_optimized<<<gridSize_opt, blockSize>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_opt;
    cudaEventElapsedTime(&time_opt, start, stop);
    time_opt /= 100.0f;

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    float speedup = time_naive / time_opt;
    float bandwidth_naive = (3 * N * sizeof(float) / 1e9) / (time_naive / 1000.0);
    float bandwidth_opt = (3 * N * sizeof(float) / 1e9) / (time_opt / 1000.0);

    printf("Naive: %.3f ms, %.2f GB/s\n", time_naive, bandwidth_naive);
    printf("Optimized: %.3f ms, %.2f GB/s\n", time_opt, bandwidth_opt);
    printf("加速比: %.2fx\n", speedup);
    printf("验证: %s\n", (h_c[0] == 3.0f) ? "PASS ✅" : "FAIL ❌");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
