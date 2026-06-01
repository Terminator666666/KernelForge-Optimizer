// Transpose 优化
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_DIM 32

// Naive Transpose
__global__ void transpose_naive(float* input, float* output, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) {
        output[x * N + y] = input[y * N + x];
    }
}

// Optimized Transpose - 共享内存 + 避免 bank conflict
__global__ void transpose_optimized(float* input, float* output, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 避免 bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < N && y < N) {
        output[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    const int N = 4096;
    const int size = N * N * sizeof(float);

    printf("🚀 Transpose 优化测试\n");
    printf("矩阵大小: %d × %d\n\n", N, N);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) h_input[i] = i;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);

    // Naive
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        transpose_naive<<<grid, block>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    time_naive /= 10.0f;

    // Optimized
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        transpose_optimized<<<grid, block>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_opt;
    cudaEventElapsedTime(&time_opt, start, stop);
    time_opt /= 10.0f;

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Naive: %.3f ms\n", time_naive);
    printf("Optimized: %.3f ms\n", time_opt);
    printf("加速比: %.2fx\n", time_naive / time_opt);
    printf("验证: %s\n", (h_output[N] == 1.0f) ? "PASS ✅" : "FAIL ❌");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
