// 矩阵乘法分块优化模板
// 使用共享内存缓存数据块，减少全局内存访问

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32

// 朴素矩阵乘法
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
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

// 分块矩阵乘法（使用共享内存）
__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    // 共享内存：缓存 A 和 B 的块
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍历 K 维度的所有块
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 A 的块到共享内存
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B 的块到共享内存
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算当前块的贡献
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 初始化矩阵
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

// 验证结果
bool verify_result(float *C_naive, float *C_tiled, int M, int N) {
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C_naive[i] - C_tiled[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("最大误差: %f\n", max_diff);
    return max_diff < 1e-3;
}

int main(int argc, char **argv) {
    // 矩阵维度
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("矩阵乘法: %d x %d x %d\n", M, N, K);

    // 分配主机内存
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_naive = (float*)malloc(size_C);
    float *h_C_tiled = (float*)malloc(size_C);

    // 初始化
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 拷贝到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 朴素版本
    dim3 blockDim_naive(16, 16);
    dim3 gridDim_naive((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_naive<<<gridDim_naive, blockDim_naive>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);

    // 分块版本
    dim3 blockDim_tiled(TILE_SIZE, TILE_SIZE);
    dim3 gridDim_tiled((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventRecord(start);
    matmul_tiled<<<gridDim_tiled, blockDim_tiled>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_tiled;
    cudaEventElapsedTime(&time_tiled, start, stop);
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);

    // 计算性能
    double gflops = 2.0 * M * N * K / 1e9;

    printf("\n性能对比:\n");
    printf("朴素版本: %.2f ms, %.2f GFLOPS\n", time_naive, gflops / (time_naive / 1000.0));
    printf("分块版本: %.2f ms, %.2f GFLOPS\n", time_tiled, gflops / (time_tiled / 1000.0));
    printf("加速比: %.2fx\n", time_naive / time_tiled);

    // 验证结果
    printf("\n验证结果: %s\n", verify_result(h_C_naive, h_C_tiled, M, N) ? "通过" : "失败");

    // 清理
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
