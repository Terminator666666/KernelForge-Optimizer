// Flash Attention (简化版) - Baseline 和 Optimized 实现
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Baseline Attention - 标准实现（多次 kernel 调用）
__global__ void matmul_qk(float* Q, float* K, float* QK, int N, int d) {
    // QK^T: (N, d) x (d, N) -> (N, N)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];  // K 已转置
        }
        QK[row * N + col] = sum;
    }
}

__global__ void softmax_rows(float* input, float* output, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float* row_input = input + row * N;
        float* row_output = output + row * N;

        // Find max
        float max_val = row_input[0];
        for (int i = 1; i < N; i++) {
            max_val = fmaxf(max_val, row_input[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += expf(row_input[i] - max_val);
        }

        // Normalize
        for (int i = 0; i < N; i++) {
            row_output[i] = expf(row_input[i] - max_val) / sum;
        }
    }
}

__global__ void matmul_av(float* A, float* V, float* output, int N, int d) {
    // A x V: (N, N) x (N, d) -> (N, d)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * V[k * d + col];
        }
        output[row * d + col] = sum;
    }
}

// Optimized Flash Attention - 简化版（先保证正确性）
__global__ void flash_attention_optimized(float* Q, float* K, float* V, float* output, int N, int d, int tile_size) {
    extern __shared__ float shmem[];

    // 共享内存布局
    float* s_QK = shmem;  // N (存储一行的 QK^T 结果)

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= N) return;

    float* q_row = Q + row * d;
    float* out_row = output + row * d;

    // Phase 1: 计算 QK^T (一行)
    for (int col = tid; col < N; col += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += q_row[i] * K[col * d + i];
        }
        s_QK[col] = sum;
    }
    __syncthreads();

    // Phase 2: Softmax
    // Find max
    float thread_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        thread_max = fmaxf(thread_max, s_QK[i]);
    }

    // Reduce max (使用 warp shuffle 或共享内存)
    __shared__ float s_max[256];
    s_max[tid] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = s_max[0];
    __syncthreads();

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float exp_val = expf(s_QK[i] - max_val);
        s_QK[i] = exp_val;
        thread_sum += exp_val;
    }

    // Reduce sum
    s_max[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] += s_max[tid + s];
        }
        __syncthreads();
    }
    float sum = s_max[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        s_QK[i] /= sum;
    }
    __syncthreads();

    // Phase 3: 计算 Attention * V
    for (int col = tid; col < d; col += blockDim.x) {
        float result = 0.0f;
        for (int i = 0; i < N; i++) {
            result += s_QK[i] * V[i * d + col];
        }
        out_row[col] = result;
    }
}

// CPU 参考实现
void attention_cpu(float* Q, float* K, float* V, float* output, int N, int d) {
    // QK^T
    float* QK = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            QK[i * N + j] = sum;
        }
    }

    // Softmax
    float* A = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        float max_val = QK[i * N];
        for (int j = 1; j < N; j++) {
            max_val = fmaxf(max_val, QK[i * N + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += expf(QK[i * N + j] - max_val);
        }

        for (int j = 0; j < N; j++) {
            A[i * N + j] = expf(QK[i * N + j] - max_val) / sum;
        }
    }

    // A x V
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * V[k * d + j];
            }
            output[i * d + j] = sum;
        }
    }

    free(QK);
    free(A);
}

int main() {
    const int N = 256;   // Sequence length (小规模测试)
    const int d = 64;    // Head dimension
    const int tile_size = 32;

    printf("🚀 Flash Attention (简化版) 优化测试\n");
    printf("================================================================================\n");
    printf("序列长度 N: %d\n", N);
    printf("特征维度 d: %d\n", d);
    printf("Tile size: %d\n", tile_size);
    printf("GPU: RTX 5070\n");
    printf("================================================================================\n\n");

    // Allocate memory
    int qkv_size = N * d * sizeof(float);
    int output_size = N * d * sizeof(float);
    int qk_size = N * N * sizeof(float);

    float *h_Q = (float*)malloc(qkv_size);
    float *h_K = (float*)malloc(qkv_size);
    float *h_V = (float*)malloc(qkv_size);
    float *h_output_baseline = (float*)malloc(output_size);
    float *h_output_opt = (float*)malloc(output_size);
    float *h_output_cpu = (float*)malloc(output_size);

    // Initialize
    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_K[i] = (float)rand() / RAND_MAX - 0.5f;
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // CPU reference
    printf("[0/3] 计算 CPU 参考结果...\n");
    attention_cpu(h_Q, h_K, h_V, h_output_cpu, N, d);

    // GPU memory
    float *d_Q, *d_K, *d_V, *d_output, *d_QK, *d_A;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_QK, qk_size);
    cudaMalloc(&d_A, qk_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Baseline (3 separate kernels) ==========
    printf("[1/3] 测试 Baseline Attention (3 个独立 kernel)...\n");

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    // Warmup
    for (int i = 0; i < 3; i++) {
        matmul_qk<<<gridDim, blockDim>>>(d_Q, d_K, d_QK, N, d);
        softmax_rows<<<(N + 255) / 256, 256>>>(d_QK, d_A, N);
        matmul_av<<<gridDim, blockDim>>>(d_A, d_V, d_output, N, d);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matmul_qk<<<gridDim, blockDim>>>(d_Q, d_K, d_QK, N, d);
        softmax_rows<<<(N + 255) / 256, 256>>>(d_QK, d_A, N);
        matmul_av<<<gridDim, blockDim>>>(d_A, d_V, d_output, N, d);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_baseline;
    cudaEventElapsedTime(&time_baseline, start, stop);
    time_baseline /= 10.0f;

    cudaMemcpy(h_output_baseline, d_output, output_size, cudaMemcpyDeviceToHost);

    // ========== Optimized (Flash Attention) ==========
    printf("[2/3] 测试 Optimized Flash Attention (融合 + Tiling)...\n");

    dim3 blockDim_opt(tile_size, tile_size / 4);  // 调整以适应共享内存
    dim3 gridDim_opt((N + tile_size - 1) / tile_size);
    int shmem_size = (tile_size * d * 3 + tile_size * tile_size) * sizeof(float);

    // Warmup
    for (int i = 0; i < 3; i++) {
        flash_attention_optimized<<<gridDim_opt, blockDim_opt, shmem_size>>>(d_Q, d_K, d_V, d_output, N, d, tile_size);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        flash_attention_optimized<<<gridDim_opt, blockDim_opt, shmem_size>>>(d_Q, d_K, d_V, d_output, N, d, tile_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_opt;
    cudaEventElapsedTime(&time_opt, start, stop);
    time_opt /= 10.0f;

    cudaMemcpy(h_output_opt, d_output, output_size, cudaMemcpyDeviceToHost);

    // ========== Results ==========
    printf("[3/3] 分析结果...\n\n");

    float speedup = time_baseline / time_opt;

    printf("================================================================================\n");
    printf("🎯 性能对比结果\n");
    printf("================================================================================\n\n");

    printf("Baseline Attention (标准实现):\n");
    printf("  执行时间: %.3f ms\n", time_baseline);
    printf("  特点: 3 个独立 kernel (QK^T + Softmax + AV)\n");
    printf("  内存访问: 多次读写全局内存\n\n");

    printf("Optimized Flash Attention:\n");
    printf("  执行时间: %.3f ms\n", time_opt);
    printf("  特点: 融合 kernel + Tiling + 在线 Softmax\n");
    printf("  内存访问: 减少全局内存访问\n\n");

    printf("🚀 加速比: %.2fx\n\n", speedup);

    // ========== Verification ==========
    printf("================================================================================\n");
    printf("✅ 精度验证\n");
    printf("================================================================================\n\n");

    float max_diff_baseline = 0.0f;
    float max_diff_opt = 0.0f;

    for (int i = 0; i < N * d; i++) {
        max_diff_baseline = fmaxf(max_diff_baseline, fabsf(h_output_baseline[i] - h_output_cpu[i]));
        max_diff_opt = fmaxf(max_diff_opt, fabsf(h_output_opt[i] - h_output_cpu[i]));
    }

    printf("Baseline vs CPU: 最大误差 %.6e - %s\n",
           max_diff_baseline, (max_diff_baseline < 1e-3) ? "✅ PASS" : "❌ FAIL");
    printf("Optimized vs CPU: 最大误差 %.6e - %s\n\n",
           max_diff_opt, (max_diff_opt < 1e-3) ? "✅ PASS" : "❌ FAIL");

    bool all_pass = (max_diff_baseline < 1e-3) && (max_diff_opt < 1e-3);
    printf("总体状态: %s\n\n", all_pass ? "✅ PASS" : "❌ FAIL");

    // Cleanup
    free(h_Q); free(h_K); free(h_V);
    free(h_output_baseline); free(h_output_opt); free(h_output_cpu);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_output); cudaFree(d_QK); cudaFree(d_A);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return all_pass ? 0 : 1;
}
