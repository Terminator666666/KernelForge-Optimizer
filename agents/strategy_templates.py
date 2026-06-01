"""
Strategy Templates Library - Provides verified optimization strategy templates.

This module contains a library of optimization strategies with code templates,
parameter selection rules, and applicability conditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class OperatorType(Enum):
    """Supported operator types."""
    MATMUL = "matmul"
    CONV2D = "conv2d"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    TRANSPOSE = "transpose"
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    UNKNOWN = "unknown"


class BottleneckType(Enum):
    """Performance bottleneck types."""
    MEMORY_BANDWIDTH = "memory_bandwidth"
    MEMORY_LATENCY = "memory_latency"
    COMPUTE_BOUND = "compute_bound"
    OCCUPANCY = "occupancy"


@dataclass
class StrategyTemplate:
    """Optimization strategy template."""

    name: str
    description: str

    # Applicability conditions
    operator_types: List[OperatorType]
    bottleneck_types: List[BottleneckType]
    min_gpu_compute_capability: float  # e.g., 7.0 for Volta, 8.0 for Ampere

    # Expected improvements
    expected_speedup: Tuple[float, float]  # (min, max) speedup range
    memory_reduction: Optional[float]  # Percentage reduction in memory traffic

    # Code template
    code_template: str

    # Parameter selection rules
    parameter_rules: Dict[str, any]

    # Implementation notes
    notes: str


class StrategyLibrary:
    """Library of optimization strategy templates."""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, StrategyTemplate]:
        """Initialize the strategy template library."""
        templates = {}

        # Template 1: Matrix Multiplication with Tiling
        templates['matmul_tiling'] = StrategyTemplate(
            name='matmul_tiling',
            description='Shared memory tiling for matrix multiplication',
            operator_types=[OperatorType.MATMUL],
            bottleneck_types=[BottleneckType.MEMORY_BANDWIDTH, BottleneckType.MEMORY_LATENCY],
            min_gpu_compute_capability=3.0,
            expected_speedup=(2.0, 5.0),
            memory_reduction=50.0,
            code_template=self._get_matmul_tiling_template(),
            parameter_rules={
                'TILE_SIZE': {
                    'type': 'int',
                    'candidates': [16, 32, 64, 128],
                    'selection': 'Choose based on shared memory size and register pressure',
                    'default': 32
                },
                'BLOCK_SIZE': {
                    'type': 'int',
                    'formula': 'TILE_SIZE * TILE_SIZE',
                    'constraints': 'Must be <= 1024 (max threads per block)'
                }
            },
            notes='Works best for square matrices with dimensions divisible by TILE_SIZE. '
                  'Reduces global memory accesses by factor of TILE_SIZE.'
        )

        # Template 2: Reduction with Warp Primitives
        templates['reduction_warp_primitives'] = StrategyTemplate(
            name='reduction_warp_primitives',
            description='Warp-level reduction using shuffle instructions',
            operator_types=[OperatorType.REDUCTION],
            bottleneck_types=[BottleneckType.MEMORY_LATENCY, BottleneckType.OCCUPANCY],
            min_gpu_compute_capability=3.0,
            expected_speedup=(3.0, 8.0),
            memory_reduction=80.0,
            code_template=self._get_reduction_warp_template(),
            parameter_rules={
                'BLOCK_SIZE': {
                    'type': 'int',
                    'candidates': [128, 256, 512, 1024],
                    'selection': 'Higher is better for occupancy, but limited by registers',
                    'default': 256
                },
                'ELEMENTS_PER_THREAD': {
                    'type': 'int',
                    'candidates': [1, 2, 4, 8],
                    'selection': 'Increase to reduce kernel launches, but watch register usage',
                    'default': 4
                }
            },
            notes='Eliminates shared memory atomics by using warp shuffle. '
                  'Requires compute capability 3.0+ for __shfl_down_sync.'
        )

        # Template 3: Vectorized Memory Access
        templates['vectorized_memory'] = StrategyTemplate(
            name='vectorized_memory',
            description='Vectorized loads/stores using float4/int4',
            operator_types=[OperatorType.ELEMENTWISE, OperatorType.MATMUL],
            bottleneck_types=[BottleneckType.MEMORY_BANDWIDTH],
            min_gpu_compute_capability=3.0,
            expected_speedup=(1.5, 3.0),
            memory_reduction=0.0,
            code_template=self._get_vectorized_memory_template(),
            parameter_rules={
                'VECTOR_SIZE': {
                    'type': 'int',
                    'candidates': [2, 4],
                    'selection': 'Use 4 for float4/int4, 2 for float2/int2',
                    'default': 4,
                    'constraints': 'Data must be aligned to VECTOR_SIZE * sizeof(type)'
                }
            },
            notes='Reduces number of memory transactions. Requires aligned memory access. '
                  'Works best when data size is divisible by vector size.'
        )

        # Template 4: Kernel Fusion
        templates['kernel_fusion'] = StrategyTemplate(
            name='kernel_fusion',
            description='Fuse multiple elementwise operations into single kernel',
            operator_types=[OperatorType.ELEMENTWISE],
            bottleneck_types=[BottleneckType.MEMORY_BANDWIDTH, BottleneckType.MEMORY_LATENCY],
            min_gpu_compute_capability=3.0,
            expected_speedup=(2.0, 4.0),
            memory_reduction=60.0,
            code_template=self._get_kernel_fusion_template(),
            parameter_rules={
                'BLOCK_SIZE': {
                    'type': 'int',
                    'candidates': [128, 256, 512],
                    'selection': 'Balance between occupancy and register usage',
                    'default': 256
                },
                'ELEMENTS_PER_THREAD': {
                    'type': 'int',
                    'candidates': [1, 2, 4],
                    'selection': 'Increase to amortize indexing overhead',
                    'default': 4
                }
            },
            notes='Eliminates intermediate memory writes. Best for chains of elementwise ops. '
                  'Watch for register spilling with complex fused operations.'
        )

        # Template 5: Tensor Core Optimization
        templates['tensor_core'] = StrategyTemplate(
            name='tensor_core',
            description='Use Tensor Cores for matrix multiplication (WMMA API)',
            operator_types=[OperatorType.MATMUL],
            bottleneck_types=[BottleneckType.COMPUTE_BOUND],
            min_gpu_compute_capability=7.0,
            expected_speedup=(5.0, 15.0),
            memory_reduction=0.0,
            code_template=self._get_tensor_core_template(),
            parameter_rules={
                'WMMA_M': {'type': 'int', 'default': 16, 'constraints': 'Must be 16'},
                'WMMA_N': {'type': 'int', 'default': 16, 'constraints': 'Must be 16'},
                'WMMA_K': {'type': 'int', 'default': 16, 'constraints': 'Must be 16'},
                'WARP_TILE_M': {
                    'type': 'int',
                    'candidates': [64, 128, 256],
                    'selection': 'Larger tiles improve reuse but increase register pressure',
                    'default': 128
                },
                'WARP_TILE_N': {
                    'type': 'int',
                    'candidates': [64, 128, 256],
                    'default': 128
                }
            },
            notes='Requires Volta+ GPU (compute capability 7.0+). Use FP16 for best performance. '
                  'Matrix dimensions must be multiples of 16.'
        )

        # Template 6: Shared Memory Bank Conflict Avoidance
        templates['bank_conflict_free'] = StrategyTemplate(
            name='bank_conflict_free',
            description='Pad shared memory to avoid bank conflicts',
            operator_types=[OperatorType.MATMUL, OperatorType.TRANSPOSE],
            bottleneck_types=[BottleneckType.MEMORY_LATENCY],
            min_gpu_compute_capability=3.0,
            expected_speedup=(1.2, 2.0),
            memory_reduction=0.0,
            code_template=self._get_bank_conflict_free_template(),
            parameter_rules={
                'PADDING': {
                    'type': 'int',
                    'formula': '1 if (TILE_SIZE % 32 == 0) else 0',
                    'selection': 'Add 1 element padding if tile size is multiple of 32',
                    'default': 1
                }
            },
            notes='Adds padding to shared memory arrays to avoid bank conflicts. '
                  'Small memory overhead but can significantly improve performance.'
        )

        # Template 7: Occupancy Optimization
        templates['occupancy_tuning'] = StrategyTemplate(
            name='occupancy_tuning',
            description='Adjust block size and resource usage for better occupancy',
            operator_types=[OperatorType.MATMUL, OperatorType.ELEMENTWISE, OperatorType.REDUCTION],
            bottleneck_types=[BottleneckType.OCCUPANCY],
            min_gpu_compute_capability=3.0,
            expected_speedup=(1.5, 3.0),
            memory_reduction=0.0,
            code_template=self._get_occupancy_tuning_template(),
            parameter_rules={
                'BLOCK_SIZE': {
                    'type': 'int',
                    'candidates': [128, 192, 256, 384, 512],
                    'selection': 'Choose to maximize occupancy given register/smem constraints',
                    'default': 256
                },
                'MAX_REGISTERS': {
                    'type': 'int',
                    'candidates': [32, 48, 64],
                    'selection': 'Lower limit increases occupancy but may hurt ILP',
                    'default': 64
                }
            },
            notes='Use __launch_bounds__ to control register allocation. '
                  'Profile with different block sizes to find optimal configuration.'
        )

        # Template 8: Cooperative Groups
        templates['cooperative_groups'] = StrategyTemplate(
            name='cooperative_groups',
            description='Use cooperative groups for flexible thread synchronization',
            operator_types=[OperatorType.REDUCTION, OperatorType.SOFTMAX],
            bottleneck_types=[BottleneckType.MEMORY_LATENCY, BottleneckType.OCCUPANCY],
            min_gpu_compute_capability=6.0,
            expected_speedup=(1.3, 2.5),
            memory_reduction=0.0,
            code_template=self._get_cooperative_groups_template(),
            parameter_rules={
                'TILE_SIZE': {
                    'type': 'int',
                    'candidates': [32, 64, 128],
                    'selection': 'Match to natural problem granularity',
                    'default': 32
                }
            },
            notes='More flexible than __syncthreads(). Enables warp-level and block-level patterns. '
                  'Requires compute capability 6.0+.'
        )

        # Template 9: Persistent Threads
        templates['persistent_threads'] = StrategyTemplate(
            name='persistent_threads',
            description='Persistent thread blocks for better load balancing',
            operator_types=[OperatorType.MATMUL, OperatorType.CONV2D],
            bottleneck_types=[BottleneckType.OCCUPANCY],
            min_gpu_compute_capability=3.0,
            expected_speedup=(1.2, 2.0),
            memory_reduction=0.0,
            code_template=self._get_persistent_threads_template(),
            parameter_rules={
                'BLOCKS_PER_SM': {
                    'type': 'int',
                    'candidates': [1, 2, 4],
                    'selection': 'Balance between occupancy and work granularity',
                    'default': 2
                }
            },
            notes='Launch fewer blocks that process multiple tiles. Reduces launch overhead. '
                  'Best for irregular workloads or when grid size >> SM count.'
        )

        return templates

    def get_applicable_strategies(self,
                                  operator_type: OperatorType,
                                  bottleneck: BottleneckType,
                                  gpu_compute_capability: float,
                                  current_performance: Optional[Dict] = None) -> List[StrategyTemplate]:
        """
        Get list of applicable strategies sorted by expected effectiveness.

        Args:
            operator_type: Type of operator being optimized
            bottleneck: Primary performance bottleneck
            gpu_compute_capability: GPU compute capability (e.g., 8.6 for RTX 3090)
            current_performance: Optional dict with current performance metrics

        Returns:
            List of applicable StrategyTemplate objects, sorted by priority
        """
        applicable = []

        for template in self.templates.values():
            # Check operator type compatibility
            if operator_type not in template.operator_types:
                continue

            # Check bottleneck type compatibility
            if bottleneck not in template.bottleneck_types:
                continue

            # Check GPU compute capability
            if gpu_compute_capability < template.min_gpu_compute_capability:
                continue

            applicable.append(template)

        # Sort by expected speedup (higher is better)
        applicable.sort(key=lambda t: t.expected_speedup[1], reverse=True)

        return applicable

    def select_parameters(self,
                         template: StrategyTemplate,
                         gpu_specs: Dict[str, any],
                         kernel_info: Dict[str, any]) -> Dict[str, any]:
        """
        Select optimal parameters for a strategy template.

        Args:
            template: Strategy template
            gpu_specs: GPU specifications (SM count, shared memory size, etc.)
            kernel_info: Current kernel information (block size, register usage, etc.)

        Returns:
            Dictionary of parameter name -> value
        """
        parameters = {}

        for param_name, param_rules in template.parameter_rules.items():
            if 'formula' in param_rules:
                # Evaluate formula (simplified - in practice would need safe eval)
                parameters[param_name] = param_rules.get('default', 0)
            elif 'candidates' in param_rules:
                # Select from candidates based on heuristics
                candidates = param_rules['candidates']

                # Simple heuristic: choose middle value by default
                # In practice, would use more sophisticated selection
                parameters[param_name] = candidates[len(candidates) // 2]
            else:
                parameters[param_name] = param_rules.get('default', 0)

        return parameters

    def instantiate_template(self,
                            template: StrategyTemplate,
                            parameters: Dict[str, any],
                            original_code: str) -> str:
        """
        Instantiate a strategy template with specific parameters.

        Args:
            template: Strategy template
            parameters: Parameter values
            original_code: Original kernel code (for context)

        Returns:
            Instantiated code with parameters filled in
        """
        code = template.code_template

        # Replace parameter placeholders
        for param_name, param_value in parameters.items():
            placeholder = f"{{{param_name}}}"
            code = code.replace(placeholder, str(param_value))

        return code

    # Code template methods
    def _get_matmul_tiling_template(self) -> str:
        return """
// Matrix multiplication with shared memory tiling
// Optimizes memory bandwidth by reusing data in shared memory

#define TILE_SIZE {TILE_SIZE}

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && t * TILE_SIZE + ty < K)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"""

    def _get_reduction_warp_template(self) -> str:
        return """
// Reduction using warp shuffle primitives
// Eliminates shared memory and synchronization overhead

#define BLOCK_SIZE {BLOCK_SIZE}
#define ELEMENTS_PER_THREAD {ELEMENTS_PER_THREAD}

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_warp(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Each thread accumulates multiple elements
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[32];
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane < (BLOCK_SIZE / 32)) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}
"""

    def _get_vectorized_memory_template(self) -> str:
        return """
// Vectorized memory access using float4
// Reduces number of memory transactions

__global__ void elementwise_vectorized(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;  // Each thread processes 4 elements

    if (idx + 3 < n) {
        // Vectorized load
        float4 data = reinterpret_cast<float4*>(input)[tid];

        // Process elements
        data.x = /* operation on data.x */;
        data.y = /* operation on data.y */;
        data.z = /* operation on data.z */;
        data.w = /* operation on data.w */;

        // Vectorized store
        reinterpret_cast<float4*>(output)[tid] = data;
    } else {
        // Handle remaining elements
        for (int i = idx; i < n; i++) {
            output[i] = /* operation on input[i] */;
        }
    }
}
"""

    def _get_kernel_fusion_template(self) -> str:
        return """
// Fused elementwise operations
// Eliminates intermediate memory writes

#define BLOCK_SIZE {BLOCK_SIZE}

__global__ void fused_elementwise(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float x = input[tid];

        // Fused operations (example: ReLU + scale + bias)
        x = fmaxf(x, 0.0f);  // ReLU
        x = x * 2.0f;         // Scale
        x = x + 1.0f;         // Bias

        output[tid] = x;
    }
}
"""

    def _get_tensor_core_template(self) -> str:
        return """
// Matrix multiplication using Tensor Cores (WMMA API)
// Requires compute capability 7.0+ and FP16 data

#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_wmma(half* A, half* B, float* C, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}
"""

    def _get_bank_conflict_free_template(self) -> str:
        return """
// Shared memory with padding to avoid bank conflicts

#define TILE_SIZE {TILE_SIZE}
#define PADDING {PADDING}

__global__ void transpose_no_conflicts(float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + PADDING];  // Add padding

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load into shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed data
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
"""

    def _get_occupancy_tuning_template(self) -> str:
        return """
// Occupancy-optimized kernel with launch bounds

#define BLOCK_SIZE {BLOCK_SIZE}
#define MAX_REGISTERS {MAX_REGISTERS}

__global__
__launch_bounds__(BLOCK_SIZE, 4)  // 4 blocks per SM
void optimized_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // Kernel logic here
        // Compiler will limit register usage to MAX_REGISTERS per thread
    }
}
"""

    def _get_cooperative_groups_template(self) -> str:
        return """
// Cooperative groups for flexible synchronization

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void reduce_cg(float* input, float* output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = (tid < n) ? input[tid] : 0.0f;

    // Warp-level reduction using cooperative groups
    for (int offset = tile32.size() / 2; offset > 0; offset >>= 1) {
        sum += tile32.shfl_down(sum, offset);
    }

    // Block-level reduction
    __shared__ float warp_sums[32];
    if (tile32.thread_rank() == 0) {
        warp_sums[threadIdx.x / 32] = sum;
    }
    block.sync();

    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            block_sum += warp_sums[i];
        }
        atomicAdd(output, block_sum);
    }
}
"""

    def _get_persistent_threads_template(self) -> str:
        return """
// Persistent thread blocks for better load balancing

#define TILE_SIZE 32

__global__ void matmul_persistent(float* A, float* B, float* C,
                                  int M, int N, int K, int num_tiles) {
    // Each block processes multiple tiles
    int tiles_per_block = (num_tiles + gridDim.x - 1) / gridDim.x;

    for (int tile_idx = 0; tile_idx < tiles_per_block; tile_idx++) {
        int global_tile = blockIdx.x * tiles_per_block + tile_idx;
        if (global_tile >= num_tiles) break;

        // Compute tile coordinates
        int tiles_per_row = (N + TILE_SIZE - 1) / TILE_SIZE;
        int tile_row = global_tile / tiles_per_row;
        int tile_col = global_tile % tiles_per_row;

        // Process this tile
        // ... (standard tiled matmul logic)
    }
}
"""


def create_strategy_library() -> StrategyLibrary:
    """Factory function to create strategy library."""
    return StrategyLibrary()
