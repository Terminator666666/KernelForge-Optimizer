// Softmax Baseline - 公平的并行实现
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Baseline Softmax - 并行版本（公平对比）
// 每个线程处理一行，使用 grid-stride loop
__global__ void softmax_baseline(float* input, float* output, int N, int D) {
    // 使用 grid-stride loop 确保所有行都被处理
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < N; row += blockDim.x * gridDim.x) {
        float* row_input = input + row * D;
        float* row_output = output + row * D;

        // Pass 1: Find max (数值稳定性)
        float max_val = row_input[0];
        for (int i = 1; i < D; i++) {
            max_val = fmaxf(max_val, row_input[i]);
        }

        // Pass 2: Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < D; i++) {
            sum += expf(row_input[i] - max_val);
        }

        // Pass 3: Normalize
        for (int i = 0; i < D; i++) {
            row_output[i] = expf(row_input[i] - max_val) / sum;
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

    // Phase 1: Find max using parallel reduction
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

    // Phase 2: Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        sdata[i] = exp_val;  // 存储 exp 值以便后续使用
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

    // Phase 3: Normalize
    for (int i = tid; i < D; i += blockDim.x) {
        row_output[i] = sdata[i] / sum;
    }
}

// CPU 参考实现
void softmax_cpu(float* input, float* output, int N, int D) {
    for (int row = 0; row < N; row++) {
        float* row_input = input + row * D;
        float* row_output = output + row * D;

        // Find max
        float max_val = row_input[0];
        for (int i = 1; i < D; i++) {
            max_val = fmaxf(max_val, row_input[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < D; i++) {
            sum += expf(row_input[i] - max_val);
        }

        // Normalize
        for (int i = 0; i < D; i++) {
            row_output[i] = expf(row_input[i] - max_val) / sum;
        }
    }
}

int main() {
    const int N = 1024;  // Batch size
    const int D = 1024;  // Feature dimension
    const int size = N * D * sizeof(float);

    printf("🚀 Softmax 优化测试\n");
    printf("================================================================================\n");
    printf("输入大小: %d × %d (%.2f MB)\n", N, D, size / 1024.0 / 1024.0);
    printf("GPU: RTX 5070\n");
    printf("================================================================================\n\n");

    // Allocate memory
    float *h_input = (float*)malloc(size);
    float *h_output_baseline = (float*)malloc(size);
    float *h_output_opt = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);

    // Initialize with random values
    srand(42);
    for (int i = 0; i < N * D; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    }

    // CPU reference
    printf("[0/3] 计算 CPU 参考结果...\n");
    softmax_cpu(h_input, h_output_cpu, N, D);

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Baseline Version ==========
    printf("[1/3] 测试 Baseline Softmax (并行版本)...\n");

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    for (int i = 0; i < 3; i++) {
        softmax_baseline<<<gridSize, blockSize>>>(d_input, d_output, N, D);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        softmax_baseline<<<gridSize, blockSize>>>(d_input, d_output, N, D);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_baseline;
    cudaEventElapsedTime(&time_baseline, start, stop);
    time_baseline /= 10.0f;

    cudaMemcpy(h_output_baseline, d_output, size, cudaMemcpyDeviceToHost);

    // ========== Optimized Version ==========
    printf("[2/3] 测试 Optimized Softmax (共享内存 + 归约)...\n");

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

    // ========== Results ==========
    printf("[3/3] 分析结果...\n\n");

    float speedup = time_baseline / time_opt;

    printf("================================================================================\n");
    printf("🎯 性能对比结果\n");
    printf("================================================================================\n\n");

    printf("Baseline Softmax (并行版本):\n");
    printf("  执行时间: %.3f ms\n", time_baseline);
    printf("  配置: %d blocks × %d threads\n", gridSize, blockSize);
    printf("  特点: 3-pass 算法，每个线程处理一行\n\n");

    printf("Optimized Softmax (共享内存 + 归约):\n");
    printf("  执行时间: %.3f ms\n", time_opt);
    printf("  配置: %d blocks × %d threads\n", gridSize_opt, blockSize_opt);
    printf("  特点: 并行归约，共享内存优化\n\n");

    printf("🚀 加速比: %.2fx\n\n", speedup);

    // ========== Verification ==========
    printf("================================================================================\n");
    printf("✅ 精度验证\n");
    printf("================================================================================\n\n");

    // Verify baseline vs CPU
    float max_diff_baseline = 0.0f;
    float avg_diff_baseline = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float diff = fabsf(h_output_baseline[i] - h_output_cpu[i]);
        max_diff_baseline = fmaxf(max_diff_baseline, diff);
        avg_diff_baseline += diff;
    }
    avg_diff_baseline /= (N * D);

    // Verify optimized vs CPU
    float max_diff_opt = 0.0f;
    float avg_diff_opt = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float diff = fabsf(h_output_opt[i] - h_output_cpu[i]);
        max_diff_opt = fmaxf(max_diff_opt, diff);
        avg_diff_opt += diff;
    }
    avg_diff_opt /= (N * D);

    // Check sum = 1 for first row
    float sum_cpu = 0.0f, sum_baseline = 0.0f, sum_opt = 0.0f;
    for (int i = 0; i < D; i++) {
        sum_cpu += h_output_cpu[i];
        sum_baseline += h_output_baseline[i];
        sum_opt += h_output_opt[i];
    }

    printf("Baseline vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_baseline);
    printf("  平均误差: %.6e\n", avg_diff_baseline);
    printf("  第一行和: %.6f (CPU: %.6f)\n", sum_baseline, sum_cpu);
    printf("  状态: %s\n\n", (max_diff_baseline < 1e-5) ? "✅ PASS" : "❌ FAIL");

    printf("Optimized vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_opt);
    printf("  平均误差: %.6e\n", avg_diff_opt);
    printf("  第一行和: %.6f (CPU: %.6f)\n", sum_opt, sum_cpu);
    printf("  状态: %s\n\n", (max_diff_opt < 1e-5) ? "✅ PASS" : "❌ FAIL");

    bool all_pass = (max_diff_baseline < 1e-5) && (max_diff_opt < 1e-5);
    printf("总体状态: %s\n\n", all_pass ? "✅ PASS" : "❌ FAIL");

    // Cleanup
    free(h_input);
    free(h_output_baseline);
    free(h_output_opt);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return all_pass ? 0 : 1;
}
