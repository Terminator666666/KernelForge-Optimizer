// Convolution 2D - Baseline 和 Optimized 实现
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Baseline Conv2D - 简单的并行实现
__global__ void conv2d_baseline(float* input, float* kernel, float* output,
                                 int N, int C, int H, int W, int K, int R, int S) {
    // N: batch, C: channels, H: height, W: width
    // K: output channels, R: kernel height, S: kernel width

    int n = blockIdx.z;
    int k = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || k >= K || hw >= H * W) return;

    int h = hw / W;
    int w = hw % W;

    float sum = 0.0f;

    // 对每个输入通道和 kernel 位置求和
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int h_in = h + r - R/2;
                int w_in = w + s - S/2;

                // 边界检查
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float in_val = input[n * C * H * W + c * H * W + h_in * W + w_in];
                    float ker_val = kernel[k * C * R * S + c * R * S + r * S + s];
                    sum += in_val * ker_val;
                }
            }
        }
    }

    output[n * K * H * W + k * H * W + h * W + w] = sum;
}

// Optimized Conv2D - 使用共享内存
__global__ void conv2d_optimized(float* input, float* kernel, float* output,
                                  int N, int C, int H, int W, int K, int R, int S) {
    extern __shared__ float shmem[];

    // 共享内存布局
    int tile_size = blockDim.x;
    float* s_input = shmem;  // tile_size * C
    float* s_kernel = s_input + tile_size * C;  // C * R * S

    int n = blockIdx.z;
    int k = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || k >= K) return;

    int h = hw / W;
    int w = hw % W;

    // 加载 kernel 到共享内存（每个 block 加载一次）
    int kernel_size = C * R * S;
    for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
        s_kernel[i] = kernel[k * kernel_size + i];
    }
    __syncthreads();

    if (hw >= H * W) return;

    float sum = 0.0f;

    // 对每个输入通道和 kernel 位置求和
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int h_in = h + r - R/2;
                int w_in = w + s - S/2;

                // 边界检查
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float in_val = input[n * C * H * W + c * H * W + h_in * W + w_in];
                    float ker_val = s_kernel[c * R * S + r * S + s];
                    sum += in_val * ker_val;
                }
            }
        }
    }

    output[n * K * H * W + k * H * W + h * W + w] = sum;
}

// CPU 参考实现
void conv2d_cpu(float* input, float* kernel, float* output,
                int N, int C, int H, int W, int K, int R, int S) {
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float sum = 0.0f;

                    for (int c = 0; c < C; c++) {
                        for (int r = 0; r < R; r++) {
                            for (int s = 0; s < S; s++) {
                                int h_in = h + r - R/2;
                                int w_in = w + s - S/2;

                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    float in_val = input[n * C * H * W + c * H * W + h_in * W + w_in];
                                    float ker_val = kernel[k * C * R * S + c * R * S + r * S + s];
                                    sum += in_val * ker_val;
                                }
                            }
                        }
                    }

                    output[n * K * H * W + k * H * W + h * W + w] = sum;
                }
            }
        }
    }
}

int main() {
    const int N = 4;    // Batch size
    const int C = 16;   // Input channels
    const int H = 64;   // Height
    const int W = 64;   // Width
    const int K = 32;   // Output channels
    const int R = 3;    // Kernel height
    const int S = 3;    // Kernel width

    printf("🚀 Conv2D 优化测试\n");
    printf("================================================================================\n");
    printf("输入大小: %d × %d × %d × %d (N×C×H×W)\n", N, C, H, W);
    printf("Kernel 大小: %d × %d × %d × %d (K×C×R×S)\n", K, C, R, S);
    printf("输出大小: %d × %d × %d × %d\n", N, K, H, W);
    printf("GPU: RTX 5070\n");
    printf("================================================================================\n\n");

    // Allocate memory
    int input_size = N * C * H * W * sizeof(float);
    int kernel_size = K * C * R * S * sizeof(float);
    int output_size = N * K * H * W * sizeof(float);

    float *h_input = (float*)malloc(input_size);
    float *h_kernel = (float*)malloc(kernel_size);
    float *h_output_baseline = (float*)malloc(output_size);
    float *h_output_opt = (float*)malloc(output_size);
    float *h_output_cpu = (float*)malloc(output_size);

    // Initialize
    srand(42);
    for (int i = 0; i < N * C * H * W; i++) {
        h_input[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < K * C * R * S; i++) {
        h_kernel[i] = (float)rand() / RAND_MAX * 0.1f;
    }

    // CPU reference
    printf("[0/3] 计算 CPU 参考结果...\n");
    conv2d_cpu(h_input, h_kernel, h_output_cpu, N, C, H, W, K, R, S);

    // GPU memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Baseline ==========
    printf("[1/3] 测试 Baseline Conv2D...\n");

    int blockSize = 256;
    dim3 gridSize((H * W + blockSize - 1) / blockSize, K, N);

    // Warmup
    for (int i = 0; i < 3; i++) {
        conv2d_baseline<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, N, C, H, W, K, R, S);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        conv2d_baseline<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, N, C, H, W, K, R, S);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_baseline;
    cudaEventElapsedTime(&time_baseline, start, stop);
    time_baseline /= 10.0f;

    cudaMemcpy(h_output_baseline, d_output, output_size, cudaMemcpyDeviceToHost);

    // ========== Optimized ==========
    printf("[2/3] 测试 Optimized Conv2D (共享内存)...\n");

    int shmem_size = (blockSize * C + C * R * S) * sizeof(float);

    // Warmup
    for (int i = 0; i < 3; i++) {
        conv2d_optimized<<<gridSize, blockSize, shmem_size>>>(d_input, d_kernel, d_output, N, C, H, W, K, R, S);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        conv2d_optimized<<<gridSize, blockSize, shmem_size>>>(d_input, d_kernel, d_output, N, C, H, W, K, R, S);
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

    printf("Baseline Conv2D:\n");
    printf("  执行时间: %.3f ms\n", time_baseline);
    printf("  特点: 直接全局内存访问\n\n");

    printf("Optimized Conv2D (共享内存):\n");
    printf("  执行时间: %.3f ms\n", time_opt);
    printf("  特点: Kernel 缓存到共享内存\n\n");

    printf("🚀 加速比: %.2fx\n\n", speedup);

    // ========== Verification ==========
    printf("================================================================================\n");
    printf("✅ 精度验证\n");
    printf("================================================================================\n\n");

    float max_diff_baseline = 0.0f;
    float max_diff_opt = 0.0f;
    float avg_diff_baseline = 0.0f;
    float avg_diff_opt = 0.0f;

    for (int i = 0; i < N * K * H * W; i++) {
        float diff_baseline = fabsf(h_output_baseline[i] - h_output_cpu[i]);
        float diff_opt = fabsf(h_output_opt[i] - h_output_cpu[i]);

        max_diff_baseline = fmaxf(max_diff_baseline, diff_baseline);
        max_diff_opt = fmaxf(max_diff_opt, diff_opt);
        avg_diff_baseline += diff_baseline;
        avg_diff_opt += diff_opt;
    }
    avg_diff_baseline /= (N * K * H * W);
    avg_diff_opt /= (N * K * H * W);

    printf("Baseline vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_baseline);
    printf("  平均误差: %.6e\n", avg_diff_baseline);
    printf("  状态: %s\n\n", (max_diff_baseline < 1e-4) ? "✅ PASS" : "❌ FAIL");

    printf("Optimized vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_opt);
    printf("  平均误差: %.6e\n", avg_diff_opt);
    printf("  状态: %s\n\n", (max_diff_opt < 1e-4) ? "✅ PASS" : "❌ FAIL");

    bool all_pass = (max_diff_baseline < 1e-4) && (max_diff_opt < 1e-4);
    printf("总体状态: %s\n\n", all_pass ? "✅ PASS" : "❌ FAIL");

    // Cleanup
    free(h_input); free(h_kernel);
    free(h_output_baseline); free(h_output_opt); free(h_output_cpu);
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return all_pass ? 0 : 1;
}
