// Scan (Prefix Sum) - Baseline 和 Optimized 实现
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Baseline Scan - 并行但效率低（Hillis-Steele 算法）
// Exclusive scan: output[i] = sum(input[0..i-1])
__global__ void scan_baseline(float* input, float* output, int N) {
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    if (idx < N) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Hillis-Steele 算法 - O(n log n) work (inclusive scan)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();

        if (tid >= stride) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    // 转换为 exclusive scan: 右移一位，第一个元素为 0
    if (idx < N) {
        output[idx] = (tid > 0) ? temp[tid - 1] : 0.0f;
    }
}

// Optimized Scan - Blelloch 算法（work-efficient）
__global__ void scan_optimized(float* input, float* output, int N) {
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存（使用 bank conflict 避免的 padding）
    int ai = tid;
    int bi = tid + (blockDim.x / 2);
    int bankOffsetA = ai / 32;  // 避免 bank conflict
    int bankOffsetB = bi / 32;

    if (idx < N) {
        temp[ai + bankOffsetA] = (ai < blockDim.x) ? input[idx] : 0.0f;
    }
    if (idx + blockDim.x / 2 < N) {
        temp[bi + bankOffsetB] = input[idx + blockDim.x / 2];
    }
    __syncthreads();

    // Up-sweep (reduce) phase
    int offset = 1;
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += ai / 32;
            bi += bi / 32;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear last element
    if (tid == 0) {
        temp[blockDim.x - 1 + (blockDim.x - 1) / 32] = 0.0f;
    }

    // Down-sweep phase
    for (int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += ai / 32;
            bi += bi / 32;

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // 写回结果
    if (idx < N) {
        output[idx] = temp[ai + bankOffsetA];
    }
    if (idx + blockDim.x / 2 < N) {
        output[idx + blockDim.x / 2] = temp[bi + bankOffsetB];
    }
}

// CPU 参考实现
void scan_cpu(float* input, float* output, int N) {
    output[0] = 0.0f;  // Exclusive scan
    for (int i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

int main() {
    const int N = 1024 * 1024;  // 1M 元素
    const int size = N * sizeof(float);

    printf("🚀 Scan (Prefix Sum) 优化测试\n");
    printf("================================================================================\n");
    printf("输入大小: %d 元素 (%.2f MB)\n", N, size / 1024.0 / 1024.0);
    printf("GPU: RTX 5070\n");
    printf("算法: Exclusive Scan\n");
    printf("================================================================================\n\n");

    // Allocate memory
    float *h_input = (float*)malloc(size);
    float *h_output_baseline = (float*)malloc(size);
    float *h_output_opt = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);

    // Initialize with ones (easy to verify)
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // CPU reference
    printf("[0/3] 计算 CPU 参考结果...\n");
    scan_cpu(h_input, h_output_cpu, N);

    // GPU memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Baseline (Hillis-Steele) ==========
    printf("[1/3] 测试 Baseline Scan (Hillis-Steele)...\n");

    int blockSize = 512;
    int gridSize = (N + blockSize - 1) / blockSize;
    int shmem_baseline = blockSize * sizeof(float);

    // 注意：这个实现只能处理单个 block 内的数据
    // 对于多个 block，需要额外的 kernel 来处理 block 间的依赖
    // 这里我们简化为只测试单个 block
    int N_test = blockSize;  // 限制为单个 block

    // Warmup
    for (int i = 0; i < 3; i++) {
        scan_baseline<<<1, blockSize, shmem_baseline>>>(d_input, d_output, N_test);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        scan_baseline<<<1, blockSize, shmem_baseline>>>(d_input, d_output, N_test);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_baseline;
    cudaEventElapsedTime(&time_baseline, start, stop);
    time_baseline /= 100.0f;

    cudaMemcpy(h_output_baseline, d_output, N_test * sizeof(float), cudaMemcpyDeviceToHost);

    // ========== Optimized (Blelloch) ==========
    printf("[2/3] 测试 Optimized Scan (Blelloch work-efficient)...\n");

    int blockSize_opt = 512;
    int shmem_opt = (blockSize_opt + blockSize_opt / 32) * sizeof(float);  // 加上 padding

    // Warmup
    for (int i = 0; i < 3; i++) {
        scan_optimized<<<1, blockSize_opt, shmem_opt>>>(d_input, d_output, N_test);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        scan_optimized<<<1, blockSize_opt, shmem_opt>>>(d_input, d_output, N_test);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_opt;
    cudaEventElapsedTime(&time_opt, start, stop);
    time_opt /= 100.0f;

    cudaMemcpy(h_output_opt, d_output, N_test * sizeof(float), cudaMemcpyDeviceToHost);

    // ========== Results ==========
    printf("[3/3] 分析结果...\n\n");

    float speedup = time_baseline / time_opt;

    printf("================================================================================\n");
    printf("🎯 性能对比结果 (测试规模: %d 元素)\n", N_test);
    printf("================================================================================\n\n");

    printf("Baseline Scan (Hillis-Steele):\n");
    printf("  执行时间: %.3f μs\n", time_baseline * 1000.0f);
    printf("  配置: 1 block × %d threads\n", blockSize);
    printf("  Work Complexity: O(n log n)\n");
    printf("  特点: 简单但不是 work-efficient\n\n");

    printf("Optimized Scan (Blelloch):\n");
    printf("  执行时间: %.3f μs\n", time_opt * 1000.0f);
    printf("  配置: 1 block × %d threads\n", blockSize_opt);
    printf("  Work Complexity: O(n)\n");
    printf("  特点: Work-efficient + Bank conflict 避免\n\n");

    printf("🚀 加速比: %.2fx\n\n", speedup);

    // ========== Verification ==========
    printf("================================================================================\n");
    printf("✅ 精度验证\n");
    printf("================================================================================\n\n");

    // Verify baseline
    float max_diff_baseline = 0.0f;
    int errors_baseline = 0;
    for (int i = 0; i < N_test; i++) {
        float diff = fabsf(h_output_baseline[i] - h_output_cpu[i]);
        max_diff_baseline = fmaxf(max_diff_baseline, diff);
        if (diff > 1e-3) errors_baseline++;
    }

    // Verify optimized
    float max_diff_opt = 0.0f;
    int errors_opt = 0;
    for (int i = 0; i < N_test; i++) {
        float diff = fabsf(h_output_opt[i] - h_output_cpu[i]);
        max_diff_opt = fmaxf(max_diff_opt, diff);
        if (diff > 1e-3) errors_opt++;
    }

    // 检查前几个和最后几个元素
    printf("前 10 个元素 (应该是 0, 1, 2, ..., 9):\n");
    printf("  CPU:      ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_output_cpu[i]);
    printf("\n  Baseline: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_output_baseline[i]);
    printf("\n  Optimized:");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_output_opt[i]);
    printf("\n\n");

    printf("Baseline vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_baseline);
    printf("  错误数量: %d / %d\n", errors_baseline, N_test);
    printf("  状态: %s\n\n", (errors_baseline == 0) ? "✅ PASS" : "❌ FAIL");

    printf("Optimized vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_opt);
    printf("  错误数量: %d / %d\n", errors_opt, N_test);
    printf("  状态: %s\n\n", (errors_opt == 0) ? "✅ PASS" : "❌ FAIL");

    bool all_pass = (errors_baseline == 0) && (errors_opt == 0);
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
