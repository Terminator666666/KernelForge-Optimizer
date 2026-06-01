// Fused MoE (简化版) - Baseline 和 Optimized 实现
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int NUM_EXPERTS = 8;

// Baseline MoE - 分离的 Gating 和 Expert 计算
__global__ void gating_network(float* input, float* gate_weights, float* gate_scores, int N, int d, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float* x = input + idx * d;

        // 简单的线性 gating: gate_scores = x * W_gate
        for (int e = 0; e < num_experts; e++) {
            float score = 0.0f;
            for (int i = 0; i < d; i++) {
                score += x[i] * gate_weights[e * d + i];
            }
            gate_scores[idx * num_experts + e] = score;
        }

        // Softmax over experts
        float max_score = gate_scores[idx * num_experts];
        for (int e = 1; e < num_experts; e++) {
            max_score = fmaxf(max_score, gate_scores[idx * num_experts + e]);
        }

        float sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            gate_scores[idx * num_experts + e] = expf(gate_scores[idx * num_experts + e] - max_score);
            sum += gate_scores[idx * num_experts + e];
        }

        for (int e = 0; e < num_experts; e++) {
            gate_scores[idx * num_experts + e] /= sum;
        }
    }
}

__global__ void expert_computation(float* input, float* expert_weights, float* expert_outputs,
                                   int N, int d, int expert_id, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float* x = input + idx * d;
        float* out = expert_outputs + idx * num_experts * d + expert_id * d;

        // 简单的线性变换: out = x * W_expert
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += x[j] * expert_weights[expert_id * d * d + i * d + j];
            }
            out[i] = sum;
        }
    }
}

__global__ void weighted_combine(float* expert_outputs, float* gate_scores, float* output,
                                 int N, int d, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float* out = output + idx * d;

        // 加权组合所有 expert 的输出
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int e = 0; e < num_experts; e++) {
                float weight = gate_scores[idx * num_experts + e];
                float expert_out = expert_outputs[idx * num_experts * d + e * d + i];
                sum += weight * expert_out;
            }
            out[i] = sum;
        }
    }
}

// Optimized Fused MoE - 简化版（先保证正确性）
__global__ void fused_moe_optimized(float* input, float* gate_weights, float* expert_weights,
                                    float* output, int N, int d, int num_experts, int top_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    float* x = input + idx * d;
    float* out = output + idx * d;

    // Phase 1: 计算 gate scores
    float gate_scores[8];  // 假设最多 8 个 experts
    for (int e = 0; e < num_experts; e++) {
        float score = 0.0f;
        for (int i = 0; i < d; i++) {
            score += x[i] * gate_weights[e * d + i];
        }
        gate_scores[e] = score;
    }

    // Softmax over experts
    float max_score = gate_scores[0];
    for (int e = 1; e < num_experts; e++) {
        max_score = fmaxf(max_score, gate_scores[e]);
    }

    float sum = 0.0f;
    for (int e = 0; e < num_experts; e++) {
        gate_scores[e] = expf(gate_scores[e] - max_score);
        sum += gate_scores[e];
    }

    for (int e = 0; e < num_experts; e++) {
        gate_scores[e] /= sum;
    }

    // Phase 2: 融合 expert 计算和加权组合
    for (int i = 0; i < d; i++) {
        float result = 0.0f;

        for (int e = 0; e < num_experts; e++) {
            // 计算 expert e 的输出的第 i 个维度
            float expert_out_i = 0.0f;
            for (int j = 0; j < d; j++) {
                expert_out_i += x[j] * expert_weights[e * d * d + i * d + j];
            }

            // 加权累加
            result += gate_scores[e] * expert_out_i;
        }

        out[i] = result;
    }
}

// CPU 参考实现
void moe_cpu(float* input, float* gate_weights, float* expert_weights, float* output,
             int N, int d, int num_experts) {
    for (int n = 0; n < N; n++) {
        float* x = input + n * d;
        float* out = output + n * d;

        // Gating
        float gate_scores[NUM_EXPERTS];
        for (int e = 0; e < num_experts; e++) {
            float score = 0.0f;
            for (int i = 0; i < d; i++) {
                score += x[i] * gate_weights[e * d + i];
            }
            gate_scores[e] = score;
        }

        // Softmax
        float max_score = gate_scores[0];
        for (int e = 1; e < num_experts; e++) {
            max_score = fmaxf(max_score, gate_scores[e]);
        }

        float sum = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            gate_scores[e] = expf(gate_scores[e] - max_score);
            sum += gate_scores[e];
        }

        for (int e = 0; e < num_experts; e++) {
            gate_scores[e] /= sum;
        }

        // Expert computation and weighted combine
        for (int i = 0; i < d; i++) {
            float result = 0.0f;

            for (int e = 0; e < num_experts; e++) {
                float expert_out_i = 0.0f;
                for (int j = 0; j < d; j++) {
                    expert_out_i += x[j] * expert_weights[e * d * d + i * d + j];
                }
                result += gate_scores[e] * expert_out_i;
            }

            out[i] = result;
        }
    }
}

int main() {
    const int N = 1024;      // Batch size
    const int d = 256;       // Feature dimension
    const int num_experts = NUM_EXPERTS;
    const int top_k = 2;     // Top-K experts (简化版使用全部)

    printf("🚀 Fused MoE (简化版) 优化测试\n");
    printf("================================================================================\n");
    printf("Batch size: %d\n", N);
    printf("Feature dim: %d\n", d);
    printf("Num experts: %d\n", num_experts);
    printf("Top-K: %d\n", top_k);
    printf("GPU: RTX 5070\n");
    printf("================================================================================\n\n");

    // Allocate memory
    int input_size = N * d * sizeof(float);
    int output_size = N * d * sizeof(float);
    int gate_size = num_experts * d * sizeof(float);
    int expert_size = num_experts * d * d * sizeof(float);
    int gate_scores_size = N * num_experts * sizeof(float);
    int expert_outputs_size = N * num_experts * d * sizeof(float);

    float *h_input = (float*)malloc(input_size);
    float *h_gate_weights = (float*)malloc(gate_size);
    float *h_expert_weights = (float*)malloc(expert_size);
    float *h_output_baseline = (float*)malloc(output_size);
    float *h_output_opt = (float*)malloc(output_size);
    float *h_output_cpu = (float*)malloc(output_size);

    // Initialize
    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_input[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < num_experts * d; i++) {
        h_gate_weights[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < num_experts * d * d; i++) {
        h_expert_weights[i] = (float)rand() / RAND_MAX * 0.1f;
    }

    // CPU reference
    printf("[0/3] 计算 CPU 参考结果...\n");
    moe_cpu(h_input, h_gate_weights, h_expert_weights, h_output_cpu, N, d, num_experts);

    // GPU memory
    float *d_input, *d_gate_weights, *d_expert_weights, *d_output;
    float *d_gate_scores, *d_expert_outputs;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_gate_weights, gate_size);
    cudaMalloc(&d_expert_weights, expert_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_gate_scores, gate_scores_size);
    cudaMalloc(&d_expert_outputs, expert_outputs_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate_weights, h_gate_weights, gate_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_expert_weights, h_expert_weights, expert_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========== Baseline (3 separate kernels) ==========
    printf("[1/3] 测试 Baseline MoE (分离的 kernel)...\n");

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    for (int i = 0; i < 3; i++) {
        gating_network<<<gridSize, blockSize>>>(d_input, d_gate_weights, d_gate_scores, N, d, num_experts);
        for (int e = 0; e < num_experts; e++) {
            expert_computation<<<gridSize, blockSize>>>(d_input, d_expert_weights, d_expert_outputs, N, d, e, num_experts);
        }
        weighted_combine<<<gridSize, blockSize>>>(d_expert_outputs, d_gate_scores, d_output, N, d, num_experts);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        gating_network<<<gridSize, blockSize>>>(d_input, d_gate_weights, d_gate_scores, N, d, num_experts);
        for (int e = 0; e < num_experts; e++) {
            expert_computation<<<gridSize, blockSize>>>(d_input, d_expert_weights, d_expert_outputs, N, d, e, num_experts);
        }
        weighted_combine<<<gridSize, blockSize>>>(d_expert_outputs, d_gate_scores, d_output, N, d, num_experts);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_baseline;
    cudaEventElapsedTime(&time_baseline, start, stop);
    time_baseline /= 10.0f;

    cudaMemcpy(h_output_baseline, d_output, output_size, cudaMemcpyDeviceToHost);

    // ========== Optimized (Fused MoE) ==========
    printf("[2/3] 测试 Optimized Fused MoE...\n");

    int shmem_size = (num_experts + d) * sizeof(float);

    // Warmup
    for (int i = 0; i < 3; i++) {
        fused_moe_optimized<<<gridSize, blockSize, shmem_size>>>(
            d_input, d_gate_weights, d_expert_weights, d_output, N, d, num_experts, top_k);
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        fused_moe_optimized<<<gridSize, blockSize, shmem_size>>>(
            d_input, d_gate_weights, d_expert_weights, d_output, N, d, num_experts, top_k);
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

    printf("Baseline MoE (分离 kernel):\n");
    printf("  执行时间: %.3f ms\n", time_baseline);
    printf("  特点: Gating + %d × Expert + Combine\n", num_experts);
    printf("  Kernel 数量: %d 个\n\n", 2 + num_experts);

    printf("Optimized Fused MoE:\n");
    printf("  执行时间: %.3f ms\n", time_opt);
    printf("  特点: 融合所有操作到单个 kernel\n");
    printf("  Kernel 数量: 1 个\n\n");

    printf("🚀 加速比: %.2fx\n\n", speedup);

    // ========== Verification ==========
    printf("================================================================================\n");
    printf("✅ 精度验证\n");
    printf("================================================================================\n\n");

    float max_diff_baseline = 0.0f;
    float max_diff_opt = 0.0f;
    float avg_diff_baseline = 0.0f;
    float avg_diff_opt = 0.0f;

    for (int i = 0; i < N * d; i++) {
        float diff_baseline = fabsf(h_output_baseline[i] - h_output_cpu[i]);
        float diff_opt = fabsf(h_output_opt[i] - h_output_cpu[i]);

        max_diff_baseline = fmaxf(max_diff_baseline, diff_baseline);
        max_diff_opt = fmaxf(max_diff_opt, diff_opt);
        avg_diff_baseline += diff_baseline;
        avg_diff_opt += diff_opt;
    }
    avg_diff_baseline /= (N * d);
    avg_diff_opt /= (N * d);

    printf("Baseline vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_baseline);
    printf("  平均误差: %.6e\n", avg_diff_baseline);
    printf("  状态: %s\n\n", (max_diff_baseline < 1e-3) ? "✅ PASS" : "❌ FAIL");

    printf("Optimized vs CPU:\n");
    printf("  最大误差: %.6e\n", max_diff_opt);
    printf("  平均误差: %.6e\n", avg_diff_opt);
    printf("  状态: %s\n\n", (max_diff_opt < 1e-3) ? "✅ PASS" : "❌ FAIL");

    bool all_pass = (max_diff_baseline < 1e-3) && (max_diff_opt < 1e-3);
    printf("总体状态: %s\n\n", all_pass ? "✅ PASS" : "❌ FAIL");

    // Cleanup
    free(h_input); free(h_gate_weights); free(h_expert_weights);
    free(h_output_baseline); free(h_output_opt); free(h_output_cpu);
    cudaFree(d_input); cudaFree(d_gate_weights); cudaFree(d_expert_weights);
    cudaFree(d_output); cudaFree(d_gate_scores); cudaFree(d_expert_outputs);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return all_pass ? 0 : 1;
}
