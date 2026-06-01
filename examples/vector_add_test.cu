// 简单的向量加法 kernel - 用于测试优化流程
#include <cuda_runtime.h>
#include <stdio.h>

// Naive 版本 - 未优化
__global__ void vector_add_naive(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host 代码
int main() {
    const int N = 1 << 20;  // 1M 元素
    const int size = N * sizeof(float);

    // 分配 host 内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 分配 device 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 拷贝数据到 device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 启动 kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    for (int i = 0; i < 10; i++) {
        vector_add_naive<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        vector_add_naive<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Average time: %.3f ms\n", milliseconds / 100.0f);

    // 拷贝结果回 host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "PASS" : "FAIL");

    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
