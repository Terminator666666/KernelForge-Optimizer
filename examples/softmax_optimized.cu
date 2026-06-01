// Softmax 优化
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Naive Softmax - 多次遍历
__global__ void softmax_naive(float* input, float* output, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        // Find max
        float max_val = input[row * D];
        for (int i = 1; i < D; i++) {
            max_val = fmaxf(max_val, input[row * D + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < D; i++) {
            sum += expf(input[row * D + i] - max_val);
        }

        // Normalize
        for (int i = 0; i < D; i++) {
            output[row * D + i] = expf(input[row * D + i] - max_val) / sum;
        }
    }
}

// Optimized Softmax - 单次遍历 + 共享内存
__global__ void softmax_optimized(float* input, float* output, int N, int D) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= N) return;

    float* row_input = input + row * D;
    float* row_output = output + row * D;

    // Find max using reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        sdata[i] = exp_val;
        thread_sum += exp_val;
    }
    __syncthreads();

    // Reduce sum
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < D; i += blockDim.x) {
        row_output[i] = sdata[i] / sum;
    }
}

int main() {
    const int N = 1024;  // Batch size
    const int D = 1024;  // Feature dimension
    const int size = N * D * sizeof(float);

    printf("🚀 Softmax 优化测试\n");
    printf("================================================================================\n");
    printf("输入大小: %d × %d (%.2f MB)\n\n", N, D, size / 1024.0 / 1024.0);

    // Allocate memory
    float *h_input = (float*)malloc(size);
    float *h_output_naive = (float*)malloc(size);
    float *h_output_opt = (float*)malloc(size);

    // Initialize with random values
    for (int i = 0; i < N * D; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Naive Version ==========
    printf("[1/2] 测试 Naive Softmax...\n");

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    for (int i = 0; i < 3; i++) {
        softmax_naive<<<gridSize, blockSize>>>(d_input, d_output, N, D);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        softmax_naive<<<gridSize, blockSize>>>(d_input, d_output, N, D);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    time_naive /= 10.0f;

    cudaMemcpy(h_output_naive, d_output, size, cudaMemcpyDeviceToHost);

    // ========== Optimized Version ==========
    printf("[2/2] 测试 Optimized Softmax...\n");

    int blockSize_opt = 256;
    int gridSize_opt = N;
    int shmem_size = blockSize_opt * sizeof(float) + D * sizeof(float);

    // Warmup
    for (int i = 0; i < 3; i++) {
        softmax_optimized<<<gridSize_opt, blockSize_opt, shmem_size>>>(d_input, d_output, N, D);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        softmax_optimized<<<gridSize_opt, blockSize_opt, shmem_size>>>(d_input, d_output, N, D);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_opt;
    cudaEventElapsedTime(&time_opt, start, stop);
    time_opt /= 10.0f;

    cudaMemcpy(h_output_opt, d_output, size, cudaMemcpyDeviceToHost);

    // Results
    float speedup = time_naive / time_opt;

    printf("\n");
    printf("================================================================================\n");
    printf("🎯 性能对比结果\n");
    printf("================================================================================\n");
    printf("\n");
    printf("Naive Softmax:\n");
    printf("  执行时间: %.3f ms\n", time_naive);
    printf("  配置: %d blocks × %d threads\n\n", gridSize, blockSize);

    printf("Optimized Softmax:\n");
    printf("  执行时间: %.3f ms\n", time_opt);
    printf("  配置: %d blocks × %d threads\n\n", gridSize_opt, blockSize_opt);

    printf("🚀 加速比: %.2fx\n\n", speedup);

    // Verification
    float max_diff = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float diff = fabsf(h_output_naive[i] - h_output_opt[i]);
        max_diff = fmaxf(max_diff, diff);
    }

    // Check sum = 1 for first row
    float sum_naive = 0.0f, sum_opt = 0.0f;
    for (int i = 0; i < D; i++) {
        sum_naive += h_output_naive[i];
        sum_opt += h_output_opt[i];
    }

    printf("验证结果:\n");
    printf("  最大差异: %.6e\n", max_diff);
    printf("  Naive 第一行和: %.6f\n", sum_naive);
    printf("  Optimized 第一行和: %.6f\n", sum_opt);
    printf("  状态: %s\n", (max_diff < 1e-5 && fabsf(sum_opt - 1.0f) < 1e-5) ? "PASS ✅" : "FAIL ❌");

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
