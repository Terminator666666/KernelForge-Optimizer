// Naive matrix multiplication kernel
// This is a baseline implementation with no optimizations

__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
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

// Expected bottlenecks:
// 1. Memory bandwidth - repeated loads from global memory
// 2. No data reuse - each element loaded K times
// 3. Uncoalesced memory access for matrix B
//
// Optimization opportunities:
// - Shared memory tiling (2-5x speedup)
// - Vectorized loads (1.5-2x additional)
// - Tensor Cores for FP16 (5-10x additional on Ampere+)
