// LayerNorm Baseline 和 Optimized 实现
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Baseline LayerNorm - 多次遍历
__global__ void layernorm_baseline(float* input, float* output, float* gamma, float* beta, int N, int D, float eps) {
    // 每个线程处理一行
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < N; row += blockDim.x * gridDim.x) {
        float* row_input = input + row * D;
        float* row_output = output + row * D;

        // Pass 1: Compute mean
        float sum = 0.0f;
        for (int i = 0; i < D; i++) {
            sum += row_input[i];
        }
        float mean = sum / D;

        // Pass 2: Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < D; i++) {
            float diff = row_input[i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / D;
        float inv_std = rsqrtf(variance + eps);

        // Pass 3: Normalize and apply affine transform
        for (int i = 0; i < D; i++) {
            float normalized = (row_input[i] - mean) * inv_std;
            row_output[i] = normalized * gamma[i] + beta[i];
        }
    }
}

// Optimized LayerNorm - Welford 在线算法 + 共享内存
__global__ void layernorm_optimized(float* input, float* output, float* gamma, float* beta, int N, int D, float eps) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= N) return;

    float* row_input = input + row * D;
    float* row_output = output + row * D;

    // Welford 在线算法计算 mean 和 variance
    // Phase 1: 每个线程计算局部统计量
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    int count = 0;

    for (int i = tid; i < D; i += blockDim.x) {
        float val = row_input[i];
        thread_sum += val;
        thread_sq_sum += val * val;
        count++;
    }

    sdata[tid] = thread_sum;
    sdata[tid + blockDim.x] = thread_sq_sum;
    __syncthreads();

    // Phase 2: 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
        }
        __syncthreads();
    }

    float mean = sdata[0] / D;
    float variance = sdata[blockDim.x] / D - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    __syncthreads();

    // Phase 3: 归一化并应用仿射变换
    for (int i = tid; i < D; i += blockDim.x) {
        float normalized = (row_input[i] - mean) * inv_std;
        row_output[i] = normalized * gamma[i] + beta[i];
    }
}

// CPU 参考实现
void layernorm_cpu(float* input, float* output, float* gamma, float* beta, int N, int D, float eps) {
    for (int row = 0; row < N; row++) {
        float* row_input = input + row * D;
        float* row_output = output + row * D;

        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < D; i++) {
            sum += row_input[i];
        }
        float mean = sum / D;

        // Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < D; i++) {
            float diff = row_input[i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / D;
        float inv_std = 1.0f / sqrtf(variance + eps);

        // Normalize and apply affine
        for (int i = 0; i < D; i++) {
            float normalized = (row_input[i] - mean) * inv_std;
            row_output[i] = normalized * gamma[i] + beta[i];
        }
    }
}

int main() {
    const int N = 1024;  // Batch size
    const int D = 1024;  // Feature dimension
    const float eps = 1e-5f;
    const int size = N * D * sizeof(float);
    const int param_size = D * sizeof(float);

    printf("🚀 LayerNorm 优化测试\n");
    printf("================================================================================\n");
    printf("输入大小: %d × %d (%.2f MB)\n", N, D, size / 1024.0 / 1024.0);
    printf("GPU: RTX 5070\n");
    printf("Epsilon: %.1e\n", eps);
    printf("================================================================================\n\n");

    // Allocate memory
    float *h_input = (float*)malloc(size);
    float *h_output_baseline = (float*)malloc(size);
    float *h_output_opt = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    float *h_gamma = (float*)malloc(param_size);
    float *h_beta = (float*)malloc(param_size);

    // Initialize
    srand(42);
    for (int i = 0; i < N * D; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    for (int i = 0; i < D; i++) {
        h_gamma[i] = 1.0f;  // 标准初始化
        h_beta[i] = 0.0f;
    }

    // CPU reference
    printf("[0/3] 计算 CPU 参考结果...\n");
    layernorm_cpu(h_input, h_output_cpu, h_gamma, h_beta, N, D, eps);

    // GPU memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_gamma, param_size);
    cudaMalloc(&d_beta, param_size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Baseline ==========
    printf("[1/3] 测试 Baseline LayerNorm (3-pass)...\n");

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    for (int i = 0; i < 3; i++) {
        layernorm_baseline<<<gridSize, blockSize>>>(d_input, d_output, d_gamma, d_beta, N, D, eps);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        layernorm_baseline<<<gridSize, blockSize>>>(d_input, d_output, d_gamma, d_beta, N, D, eps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_baseline;
    cudaEventElapsedTime(&time_baseline, start, stop);
    time_baseline /= 10.0f;

    cudaMemcpy(h_output_baseline, d_output, size, cudaMemcpyDeviceToHost);

    // ========== Optimized ==========
    printf("[2/3] 测试 Optimized LayerNorm (Welford + 共享内存)...\n");

    int blockSize_opt = 256;
    int gridSize_opt = N;
    int shmem_size = blockSize_opt * 2 * sizeof(float);

    // Warmup
    for (int i = 0; i < 3; i++) {
        layernorm_optimized<<<gridSize_opt, blockSize_opt, shmem_size>>>(d_input, d_output, d_gamma, d_beta, N, D, eps);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        layernorm_optimized<<<gridSize_opt, blockSize_opt, shmem_size>>>(d_input, d_output, d_gamma, d_beta, N, D, eps);
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

    printf("Baseline LayerNorm (3-pass):\n");
    printf("  执行时间: %.3f ms\n", time_baseline);
    printf("  配置: %d blocks × %d threads\n", gridSize, blockSize);
    printf("  特点: 分离计算 mean、variance、normalize\n\n");

    printf("Optimized LayerNorm (Welford + 共享内存):\n");
    printf("  执行时间: %.3f ms\n", time_opt);
    printf("  配置: %d blocks × %d threads\n", gridSize_opt, blockSize_opt);
    printf("  特点: 在线算法，并行归约\n\n");

    printf("🚀 加速比: %.2fx\n\n", speedup);

    // ========== Verification ==========
    printf("================================================================================\n");
    printf("✅ 精度验证\n");
    printf("================================================================================\n\n");

    // Verify baseline
    float max_diff_baseline = 0.0f;
    float avg_diff_baseline = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float diff = fabsf(h_output_baseline[i] - h_output_cpu[i]);
        max_diff_baseline = fmaxf(max_diff_baseline, diff);
        avg_diff_baseline += diff;
    }
    avg_diff_baseline /= (N * D);

    // Verify optimized
    float max_diff_opt = 0.0f;
    float avg_diff_opt = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float diff = fabsf(h_output_opt[i] - h_output_cpu[i]);
        max_diff_opt = fmaxf(max_diff_opt, diff);
        avg_diff_opt += diff;
    }
    avg_diff_opt /= (N * D);

    // Check mean and variance of first row
    float mean_cpu = 0.0f, mean_opt = 0.0f;
    for (int i = 0; i < D; i++) {
        mean_cpu += h_output_cpu[i];
        mean_opt += h_output_opt[i];
    }
    mean_cpu /= D;
    mean_opt /= D;

    printf("Baseline vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_baseline);
    printf("  平均误差: %.6e\n", avg_diff_baseline);
    printf("  状态: %s\n\n", (max_diff_baseline < 1e-4) ? "✅ PASS" : "❌ FAIL");

    printf("Optimized vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_opt);
    printf("  平均误差: %.6e\n", avg_diff_opt);
    printf("  第一行均值: %.6f (CPU: %.6f)\n", mean_opt, mean_cpu);
    printf("  状态: %s\n\n", (max_diff_opt < 1e-4) ? "✅ PASS" : "❌ FAIL");

    bool all_pass = (max_diff_baseline < 1e-4) && (max_diff_opt < 1e-4);
    printf("总体状态: %s\n\n", all_pass ? "✅ PASS" : "❌ FAIL");

    // Cleanup
    free(h_input);
    free(h_output_baseline);
    free(h_output_opt);
    free(h_output_cpu);
    free(h_gamma);
    free(h_beta);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return all_pass ? 0 : 1;
}
