// 向量化内存访问优化模板
// 使用 float4 向量类型，减少内存事务数量

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// 标量访问
__global__ void elementwise_scalar(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Element-wise 操作（例如 ReLU）
        output[i] = fmaxf(input[i], 0.0f);
    }
}

// 向量化访问（float4）
__global__ void elementwise_vectorized(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < n) {
        // 加载 4 个元素
        float4 val = ((float4*)input)[idx];

        // 处理 4 个元素
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);

        // 存储 4 个元素
        ((float4*)output)[idx] = val;
    }

    // 处理剩余元素
    for (int i = (n / 4) * 4 + threadIdx.x; i < n; i += blockDim.x) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

// 向量化加法（多输入）
__global__ void add_vectorized(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < n) {
        float4 a_vec = ((float4*)a)[idx];
        float4 b_vec = ((float4*)b)[idx];

        float4 c_vec;
        c_vec.x = a_vec.x + b_vec.x;
        c_vec.y = a_vec.y + b_vec.y;
        c_vec.z = a_vec.z + b_vec.z;
        c_vec.w = a_vec.w + b_vec.w;

        ((float4*)c)[idx] = c_vec;
    }

    // 处理剩余元素
    for (int i = (n / 4) * 4 + threadIdx.x; i < n; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

// 向量化 + 循环展开
__global__ void elementwise_vectorized_unrolled(float *input, float *output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        // 使用循环展开
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            output[idx + i] = fmaxf(input[idx + i], 0.0f);
        }
    }
}

// 检查内存对齐
bool is_aligned(void *ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

// 初始化数组
void init_array(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 200 - 100) / 10.0f;  // -10.0 到 10.0
    }
}

// 验证结果
bool verify_result(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            printf("不匹配: a[%d]=%.6f, b[%d]=%.6f\n", i, a[i], i, b[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 67108864;  // 64M 元素

    // 确保 n 是 4 的倍数
    n = (n / 4) * 4;

    printf("向量化内存访问: %d 元素 (%.2f MB)\n", n, n * sizeof(float) / 1e6);

    // 分配主机内存（16-byte 对齐）
    float *h_input, *h_output_scalar, *h_output_vec;
    cudaMallocHost(&h_input, n * sizeof(float));
    cudaMallocHost(&h_output_scalar, n * sizeof(float));
    cudaMallocHost(&h_output_vec, n * sizeof(float));

    // 初始化
    init_array(h_input, n);

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // 检查对齐
    printf("设备内存对齐: input=%s, output=%s\n",
           is_aligned(d_input, 16) ? "是" : "否",
           is_aligned(d_output, 16) ? "是" : "否");

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 标量版本
    int grid_scalar = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(start);
    elementwise_scalar<<<grid_scalar, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_scalar;
    cudaEventElapsedTime(&time_scalar, start, stop);
    cudaMemcpy(h_output_scalar, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 向量化版本
    int grid_vec = (n / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(start);
    elementwise_vectorized<<<grid_vec, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_vec;
    cudaEventElapsedTime(&time_vec, start, stop);
    cudaMemcpy(h_output_vec, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 计算带宽（读 + 写）
    double bytes = 2.0 * n * sizeof(float);

    printf("\n性能对比:\n");
    printf("标量访问:   %.2f ms, %.2f GB/s\n",
           time_scalar, bytes / time_scalar / 1e6);
    printf("向量化访问: %.2f ms, %.2f GB/s\n",
           time_vec, bytes / time_vec / 1e6);
    printf("加速比: %.2fx\n", time_scalar / time_vec);

    // 验证结果
    printf("\n验证结果: %s\n", verify_result(h_output_scalar, h_output_vec, n) ? "通过" : "失败");

    // 清理
    cudaFreeHost(h_input);
    cudaFreeHost(h_output_scalar);
    cudaFreeHost(h_output_vec);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
