// Warp 级归约优化模板
// 使用 warp shuffle 指令，避免共享内存和同步开销

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// Warp 内归约（使用 shuffle）
__device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;

    // 树形归约：16 -> 8 -> 4 -> 2 -> 1
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);

    return val;
}

// 传统归约（使用共享内存）
__global__ void reduce_naive(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Warp 归约（单元素）
__global__ void reduce_warp(float *input, float *output, int n) {
    __shared__ float warp_results[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 加载数据
    float val = (tid < n) ? input[tid] : 0.0f;

    // Warp 内归约
    val = warp_reduce_sum(val);

    // 每个 warp 的第一个线程写入共享内存
    if (lane == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    // 第一个 warp 归约所有 warp 的结果
    if (warp_id == 0) {
        val = (lane < (blockDim.x / 32)) ? warp_results[lane] : 0.0f;
        val = warp_reduce_sum(val);

        if (lane == 0) {
            atomicAdd(output, val);
        }
    }
}

// Warp 归约（多元素）
__global__ void reduce_warp_multi(float *input, float *output, int n) {
    __shared__ float warp_results[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 每个线程处理多个元素
    float sum = 0.0f;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * blockDim.x * gridDim.x;
        if (idx < n) {
            sum += input[idx];
        }
    }

    // Warp 内归约
    sum = warp_reduce_sum(sum);

    // Block 级归约
    if (lane == 0) {
        warp_results[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < (blockDim.x / 32)) ? warp_results[lane] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

// CPU 归约（验证用）
float reduce_cpu(float *data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 16777216;  // 16M 元素

    printf("数组归约: %d 元素 (%.2f MB)\n", n, n * sizeof(float) / 1e6);

    // 分配主机内存
    float *h_input = (float*)malloc(n * sizeof(float));

    // 初始化（使用小值避免溢出）
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f / n;  // 总和应该是 1.0
    }

    // CPU 归约（验证用）
    float cpu_result = reduce_cpu(h_input, n);

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 传统归约
    int grid_naive = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMemset(d_output, 0, sizeof(float));

    cudaEventRecord(start);
    reduce_naive<<<grid_naive, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive, result_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    cudaMemcpy(&result_naive, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Warp 归约（单元素）
    cudaMemset(d_output, 0, sizeof(float));

    cudaEventRecord(start);
    reduce_warp<<<grid_naive, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_warp, result_warp;
    cudaEventElapsedTime(&time_warp, start, stop);
    cudaMemcpy(&result_warp, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Warp 归约（多元素）
    int grid_multi = (n + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    cudaMemset(d_output, 0, sizeof(float));

    cudaEventRecord(start);
    reduce_warp_multi<<<grid_multi, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_multi, result_multi;
    cudaEventElapsedTime(&time_multi, start, stop);
    cudaMemcpy(&result_multi, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // 计算带宽
    double bytes = n * sizeof(float);

    printf("\n性能对比:\n");
    printf("传统归约:     %.2f ms, %.2f GB/s, 结果=%.6f\n",
           time_naive, bytes / time_naive / 1e6, result_naive);
    printf("Warp 归约:    %.2f ms, %.2f GB/s, 结果=%.6f, 加速 %.2fx\n",
           time_warp, bytes / time_warp / 1e6, result_warp, time_naive / time_warp);
    printf("Warp 多元素:  %.2f ms, %.2f GB/s, 结果=%.6f, 加速 %.2fx\n",
           time_multi, bytes / time_multi / 1e6, result_multi, time_naive / time_multi);

    printf("\nCPU 结果: %.6f\n", cpu_result);
    printf("误差: 传统=%.2e, Warp=%.2e, 多元素=%.2e\n",
           fabs(result_naive - cpu_result),
           fabs(result_warp - cpu_result),
           fabs(result_multi - cpu_result));

    // 清理
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
