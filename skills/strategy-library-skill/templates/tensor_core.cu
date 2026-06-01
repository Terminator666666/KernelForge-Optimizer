// Tensor Core 优化模板
// 使用 WMMA API 加速矩阵乘法（需要 Volta+ GPU）

#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// FP32 矩阵乘法（基准）
__global__ void matmul_fp32(float *A, float *B, float *C, int M, int N, int K) {
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

// FP16 Tensor Core 矩阵乘法
__global__ void matmul_wmma(half *A, half *B, float *C, int M, int N, int K) {
    // Warp 和 lane ID
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 定义 fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);

    // 遍历 K 维度
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

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
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// FP32 转 FP16
__global__ void convert_fp32_to_fp16(float *input, half *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = __float2half(input[i]);
    }
}

// 初始化矩阵
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

// 验证结果
bool verify_result(float *C_fp32, float *C_wmma, int M, int N, float tolerance) {
    float max_diff = 0.0f;
    int error_count = 0;

    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C_fp32[i] - C_wmma[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) error_count++;
    }

    printf("最大误差: %.6f, 错误数量: %d / %d\n", max_diff, error_count, M * N);
    return error_count == 0;
}

int main(int argc, char **argv) {
    // 检查 GPU 是否支持 Tensor Core
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (prop.major < 7) {
        printf("错误: Tensor Core 需要 Compute Capability 7.0+ (Volta+)\n");
        printf("当前 GPU: %s, Compute Capability: %d.%d\n",
               prop.name, prop.major, prop.minor);
        return 1;
    }

    printf("GPU: %s, Compute Capability: %d.%d\n",
           prop.name, prop.major, prop.minor);

    // 矩阵维度（必须是 16 的倍数）
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    // 对齐到 16
    M = ((M + 15) / 16) * 16;
    N = ((N + 15) / 16) * 16;
    K = ((K + 15) / 16) * 16;

    printf("矩阵乘法: %d x %d x %d\n", M, N, K);

    // 分配主机内存
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_fp32 = (float*)malloc(size_C);
    float *h_C_wmma = (float*)malloc(size_C);

    // 初始化
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // 分配设备内存
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    half *d_A_fp16, *d_B_fp16;
    float *d_C_wmma;

    cudaMalloc(&d_A_fp32, size_A);
    cudaMalloc(&d_B_fp32, size_B);
    cudaMalloc(&d_C_fp32, size_C);
    cudaMalloc(&d_A_fp16, M * K * sizeof(half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(half));
    cudaMalloc(&d_C_wmma, size_C);

    // 拷贝到设备
    cudaMemcpy(d_A_fp32, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B, size_B, cudaMemcpyHostToDevice);

    // 转换为 FP16
    int convert_threads = 256;
    int convert_blocks_A = (M * K + convert_threads - 1) / convert_threads;
    int convert_blocks_B = (K * N + convert_threads - 1) / convert_threads;

    convert_fp32_to_fp16<<<convert_blocks_A, convert_threads>>>(d_A_fp32, d_A_fp16, M * K);
    convert_fp32_to_fp16<<<convert_blocks_B, convert_threads>>>(d_B_fp32, d_B_fp16, K * N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // FP32 基准
    dim3 blockDim_fp32(16, 16);
    dim3 gridDim_fp32((N + 15) / 16, (M + 15) / 16);

    cudaEventRecord(start);
    matmul_fp32<<<gridDim_fp32, blockDim_fp32>>>(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_fp32;
    cudaEventElapsedTime(&time_fp32, start, stop);
    cudaMemcpy(h_C_fp32, d_C_fp32, size_C, cudaMemcpyDeviceToHost);

    // FP16 Tensor Core
    dim3 blockDim_wmma(128, 4);
    dim3 gridDim_wmma((M + (WMMA_M * blockDim_wmma.x / 32) - 1) / (WMMA_M * blockDim_wmma.x / 32),
                      (N + (WMMA_N * blockDim_wmma.y) - 1) / (WMMA_N * blockDim_wmma.y));

    cudaEventRecord(start);
    matmul_wmma<<<gridDim_wmma, blockDim_wmma>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_wmma;
    cudaEventElapsedTime(&time_wmma, start, stop);
    cudaMemcpy(h_C_wmma, d_C_wmma, size_C, cudaMemcpyDeviceToHost);

    // 计算性能
    double gflops = 2.0 * M * N * K / 1e9;

    printf("\n性能对比:\n");
    printf("FP32:        %.2f ms, %.2f GFLOPS\n",
           time_fp32, gflops / (time_fp32 / 1000.0));
    printf("FP16 WMMA:   %.2f ms, %.2f GFLOPS\n",
           time_wmma, gflops / (time_wmma / 1000.0));
    printf("加速比: %.2fx\n", time_fp32 / time_wmma);

    // 验证结果（FP16 精度较低，容忍度设为 0.01）
    printf("\n验证结果: %s\n",
           verify_result(h_C_fp32, h_C_wmma, M, N, 0.01f) ? "通过" : "失败");

    // 清理
    free(h_A); free(h_B); free(h_C_fp32); free(h_C_wmma);
    cudaFree(d_A_fp32); cudaFree(d_B_fp32); cudaFree(d_C_fp32);
    cudaFree(d_A_fp16); cudaFree(d_B_fp16); cudaFree(d_C_wmma);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
