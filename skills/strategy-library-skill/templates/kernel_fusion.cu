// Kernel Fusion 优化模板
// 将多个 element-wise 操作融合为单个 kernel

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// 未融合：三个独立的 kernel
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void mul_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void relu_kernel(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = fmaxf(a[i], 0.0f);
    }
}

// 融合：单个 kernel
__global__ void fused_add_mul_relu(float *x, float *y, float *z, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // 所有操作在寄存器中完成
        float temp1 = x[i] + y[i];      // Add
        float temp2 = temp1 * z[i];     // Mul
        output[i] = fmaxf(temp2, 0.0f); // ReLU
    }
}

// 融合 + 向量化
__global__ void fused_vectorized(float *x, float *y, float *z, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < n) {
        float4 x_vec = ((float4*)x)[idx];
        float4 y_vec = ((float4*)y)[idx];
        float4 z_vec = ((float4*)z)[idx];

        float4 out_vec;
        out_vec.x = fmaxf((x_vec.x + y_vec.x) * z_vec.x, 0.0f);
        out_vec.y = fmaxf((x_vec.y + y_vec.y) * z_vec.y, 0.0f);
        out_vec.z = fmaxf((x_vec.z + y_vec.z) * z_vec.z, 0.0f);
        out_vec.w = fmaxf((x_vec.w + y_vec.w) * z_vec.w, 0.0f);

        ((float4*)output)[idx] = out_vec;
    }
}

void init_array(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 200 - 100) / 10.0f;
    }
}

bool verify_result(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > 1e-4) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 67108864;
    n = (n / 4) * 4;

    printf("Kernel Fusion: %d 元素 (%.2f MB)\n", n, n * sizeof(float) / 1e6);

    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    float *h_z = (float*)malloc(n * sizeof(float));
    float *h_output_separate = (float*)malloc(n * sizeof(float));
    float *h_output_fused = (float*)malloc(n * sizeof(float));

    init_array(h_x, n);
    init_array(h_y, n);
    init_array(h_z, n);

    float *d_x, *d_y, *d_z, *d_output, *d_temp1, *d_temp2;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMalloc(&d_temp1, n * sizeof(float));
    cudaMalloc(&d_temp2, n * sizeof(float));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, n * sizeof(float), cudaMemcpyHostToDevice);

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 未融合版本
    cudaEventRecord(start);
    add_kernel<<<grid, BLOCK_SIZE>>>(d_x, d_y, d_temp1, n);
    mul_kernel<<<grid, BLOCK_SIZE>>>(d_temp1, d_z, d_temp2, n);
    relu_kernel<<<grid, BLOCK_SIZE>>>(d_temp2, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_separate;
    cudaEventElapsedTime(&time_separate, start, stop);
    cudaMemcpy(h_output_separate, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 融合版本
    cudaEventRecord(start);
    fused_add_mul_relu<<<grid, BLOCK_SIZE>>>(d_x, d_y, d_z, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_fused;
    cudaEventElapsedTime(&time_fused, start, stop);
    cudaMemcpy(h_output_fused, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 融合 + 向量化
    int grid_vec = (n / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEventRecord(start);
    fused_vectorized<<<grid_vec, BLOCK_SIZE>>>(d_x, d_y, d_z, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_fused_vec;
    cudaEventElapsedTime(&time_fused_vec, start, stop);

    double bytes_separate = 8.0 * n * sizeof(float);
    double bytes_fused = 4.0 * n * sizeof(float);

    printf("\n性能对比:\n");
    printf("未融合: %.2f ms, %.2f GB/s\n",
           time_separate, bytes_separate / time_separate / 1e6);
    printf("融合:   %.2f ms, %.2f GB/s, 加速 %.2fx\n",
           time_fused, bytes_fused / time_fused / 1e6, time_separate / time_fused);
    printf("融合+向量化: %.2f ms, %.2f GB/s, 加速 %.2fx\n",
           time_fused_vec, bytes_fused / time_fused_vec / 1e6, time_separate / time_fused_vec);

    printf("\n验证: %s\n", verify_result(h_output_separate, h_output_fused, n) ? "通过" : "失败");

    free(h_x); free(h_y); free(h_z);
    free(h_output_separate); free(h_output_fused);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_output); cudaFree(d_temp1); cudaFree(d_temp2);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
