"""
Enhanced Optimization Prompt - Provides strategy templates and implementation guidance.

This module builds prompts for the Generator LLM that include specific optimization
strategies, code templates, and parameter recommendations.
"""

from typing import Dict, List, Optional
from agents.strategy_templates import StrategyTemplate
from agents.ncu_interpreter import PerformanceDiagnosis


def build_enhanced_optimization_prompt(
    diagnosis: PerformanceDiagnosis,
    strategy: StrategyTemplate,
    operator_type: str,
    current_code: str,
    gpu_specs: Dict[str, any],
    judge_recommendations: Optional[str] = None
) -> str:
    """
    Build enhanced optimization prompt with strategy template and guidance.

    Args:
        diagnosis: Performance diagnosis from NCU Interpreter
        strategy: Selected optimization strategy template
        operator_type: Type of operator
        current_code: Current kernel code
        gpu_specs: GPU specifications
        judge_recommendations: Optional recommendations from Judge LLM

    Returns:
        Formatted prompt string for Generator LLM
    """

    # Build diagnosis summary
    diagnosis_summary = _build_diagnosis_summary(diagnosis)

    # Build strategy-specific sections (only if strategy is provided)
    if strategy:
        strategy_guidance = _build_strategy_guidance(strategy, gpu_specs)
        template_section = _build_template_section(strategy)
        param_recommendations = _build_parameter_recommendations(strategy, gpu_specs, diagnosis)

        strategy_header = f"""
## Optimization Strategy: {strategy.name}

{strategy.description}

**Expected Speedup**: {strategy.expected_speedup[0]:.1f}x - {strategy.expected_speedup[1]:.1f}x

{strategy_guidance}

## Parameter Recommendations

{param_recommendations}

{template_section}
"""
    else:
        # No specific strategy - use general optimization guidance
        strategy_header = """
## Optimization Goal

Optimize this kernel for better performance. Focus on:
- Reducing memory bandwidth bottlenecks
- Improving compute utilization
- Optimizing memory access patterns
- Increasing occupancy if needed
"""

    # Build judge context if available
    judge_context = ""
    if judge_recommendations:
        judge_context = f"""
## Judge Analysis

{judge_recommendations}

---
"""

    prompt = f"""You are an expert CUDA kernel optimizer. Your task is to optimize the following kernel.

## Operator Information
- **Type**: {operator_type}
- **GPU**: {gpu_specs.get('name', 'Unknown')} ({gpu_specs.get('architecture', 'Unknown')} architecture)
- **Compute Capability**: {gpu_specs.get('compute_capability', 'Unknown')}

## Performance Diagnosis

{diagnosis_summary}

{judge_context}

{strategy_header}

## Current Kernel Code

```python
{current_code}
```

## Your Task

{f"Optimize the kernel using the **{strategy.name}** strategy. Follow these guidelines:" if strategy else "Optimize the kernel for better performance. Follow these guidelines:"}

1. **Apply the Strategy**: Implement the optimization technique described above
2. **Use Recommended Parameters**: Start with the suggested parameter values
3. **Preserve Correctness**: Ensure the optimized kernel produces the same results
4. **Add Comments**: Explain key optimization decisions in the code
5. **Handle Edge Cases**: Ensure the kernel works for all valid input sizes

## Output Format

**CRITICAL**: You MUST output a complete Python file in the EXACT format shown below. Do NOT output raw CUDA code.

The output must be a valid Python file that can be executed with `python script.py --repeat 100`.

```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    \"\"\"
    Optimized model using the selected strategy
    \"\"\"
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Optimized forward pass
        \"\"\"
        # Your optimized PyTorch/CUDA implementation here
        return torch.matmul(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    model = ModelNew().cuda()
    inputs = get_inputs()
    inputs = [x.cuda() for x in inputs]

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model(*inputs)

    torch.cuda.synchronize()

    # Timing runs
    start = time.time()
    for _ in range(args.repeat):
        with torch.no_grad():
            output = model(*inputs)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / args.repeat * 1000
    print(f"Average time: {{{{avg_time_ms:.3f}}}} ms")
```

**IMPORTANT**:
- Output ONLY the complete Python code above
- You CAN use `torch.utils.cpp_extension.load_inline` for custom CUDA kernels
- You CAN write custom CUDA kernels with `__global__` functions
- CRITICAL: Add this at the very beginning of your code to fix GPU architecture compatibility:
  ```python
  import os
  os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # Force Ada Lovelace arch for compatibility
  ```
- Focus on algorithmic optimizations: shared memory tiling, warp-level primitives, vectorized memory access
- The code must be executable as-is with `python script.py`

## Important Notes

{f"- Focus on the specific optimization strategy: **{strategy.name}**" if strategy else "- Focus on general performance optimization techniques"}
- Use the recommended parameters as a starting point
- Ensure memory accesses are coalesced
- Avoid bank conflicts in shared memory
- Consider occupancy vs. resource usage tradeoffs
- Test boundary conditions (e.g., when dimensions are not multiples of tile size)

{_build_strategy_specific_notes(strategy, diagnosis) if strategy else ""}
"""

    return prompt


def _build_diagnosis_summary(diagnosis: PerformanceDiagnosis) -> str:
    """Build concise diagnosis summary for optimization context."""

    summary_lines = [
        f"**Primary Bottleneck**: {diagnosis.bottleneck} (confidence: {diagnosis.bottleneck_confidence:.0%})",
        "",
        "**Key Metrics**:",
        f"- Memory bandwidth utilization: {diagnosis.memory_bandwidth_util:.1f}%",
        f"- Occupancy: {diagnosis.achieved_occupancy:.1f}% (theoretical: {diagnosis.theoretical_occupancy:.1f}%)",
        f"- Arithmetic intensity: {diagnosis.arithmetic_intensity:.2f} FLOPs/byte",
        f"- Roofline region: {diagnosis.roofline_region}",
    ]

    # Add top issues
    if diagnosis.issues:
        summary_lines.append("")
        summary_lines.append("**Top Issues**:")
        for issue in diagnosis.issues[:3]:
            summary_lines.append(f"- {issue['description']}")

    return "\n".join(summary_lines)


def _build_strategy_guidance(strategy: StrategyTemplate, gpu_specs: Dict[str, any]) -> str:
    """Build strategy-specific implementation guidance."""

    guidance_lines = [
        "### Implementation Guidance",
        "",
        strategy.notes,
        "",
        "**Applicability**:",
        f"- Operator types: {', '.join([op.value for op in strategy.operator_types])}",
        f"- Bottleneck types: {', '.join([bt.value for bt in strategy.bottleneck_types])}",
        f"- Minimum compute capability: {strategy.min_gpu_compute_capability}",
    ]

    # Add GPU-specific notes
    if gpu_specs.get('compute_capability', 0) < strategy.min_gpu_compute_capability:
        guidance_lines.append("")
        guidance_lines.append(f"⚠️ **Warning**: Current GPU (compute capability {gpu_specs.get('compute_capability')}) "
                            f"is below minimum requirement ({strategy.min_gpu_compute_capability}). "
                            f"This strategy may not work or may have reduced effectiveness.")

    # Add architecture-specific guidance
    arch = gpu_specs.get('architecture', '')
    if 'Tensor' in strategy.name and arch in ['Volta', 'Turing', 'Ampere', 'Ada Lovelace', 'Hopper']:
        guidance_lines.append("")
        guidance_lines.append(f"✓ **GPU Support**: {arch} architecture supports Tensor Cores (Generation {gpu_specs.get('tensor_core_generation', 'N/A')})")

    return "\n".join(guidance_lines)


def _build_template_section(strategy: StrategyTemplate) -> str:
    """Build code template section."""

    if not strategy.code_template:
        return ""

    template_lines = [
        "## Code Template",
        "",
        "Use this template as a reference for implementing the optimization:",
        "",
        "```cuda",
        strategy.code_template.strip(),
        "```",
        "",
        "**Note**: Adapt this template to your specific kernel. Replace placeholders with actual values.",
    ]

    return "\n".join(template_lines)


def _build_parameter_recommendations(
    strategy: StrategyTemplate,
    gpu_specs: Dict[str, any],
    diagnosis: PerformanceDiagnosis
) -> str:
    """Build parameter recommendations based on strategy and GPU specs."""

    param_lines = []

    for param_name, param_rules in strategy.parameter_rules.items():
        param_lines.append(f"### {param_name}")

        # Add description
        if 'selection' in param_rules:
            param_lines.append(f"**Selection Guidance**: {param_rules['selection']}")

        # Add candidates
        if 'candidates' in param_rules:
            candidates = param_rules['candidates']
            param_lines.append(f"**Candidates**: {', '.join(map(str, candidates))}")

            # Recommend specific value based on context
            recommended = _select_parameter_value(
                param_name, param_rules, gpu_specs, diagnosis
            )
            param_lines.append(f"**Recommended**: {recommended}")

        # Add formula if available
        if 'formula' in param_rules:
            param_lines.append(f"**Formula**: `{param_rules['formula']}`")

        # Add constraints
        if 'constraints' in param_rules:
            param_lines.append(f"**Constraints**: {param_rules['constraints']}")

        # Add default
        if 'default' in param_rules:
            param_lines.append(f"**Default**: {param_rules['default']}")

        param_lines.append("")

    return "\n".join(param_lines)


def _select_parameter_value(
    param_name: str,
    param_rules: Dict[str, any],
    gpu_specs: Dict[str, any],
    diagnosis: PerformanceDiagnosis
) -> any:
    """Select optimal parameter value based on context."""

    candidates = param_rules.get('candidates', [])
    if not candidates:
        return param_rules.get('default', 'N/A')

    # Tile size selection
    if 'TILE' in param_name.upper():
        # Prefer larger tiles for memory-bound kernels
        if diagnosis.bottleneck == 'memory_bandwidth':
            return candidates[-1] if len(candidates) > 0 else 32
        # Prefer smaller tiles for occupancy-limited kernels
        elif diagnosis.bottleneck == 'occupancy':
            return candidates[0] if len(candidates) > 0 else 16
        else:
            return candidates[len(candidates) // 2]

    # Block size selection
    elif 'BLOCK_SIZE' in param_name.upper():
        # Consider occupancy
        if diagnosis.achieved_occupancy < 50:
            # Try larger block size to improve occupancy
            return max(candidates)
        else:
            # Use medium block size
            return candidates[len(candidates) // 2] if len(candidates) > 1 else 256

    # Vector size selection
    elif 'VECTOR' in param_name.upper():
        # Prefer larger vectors for memory-bound kernels
        if diagnosis.bottleneck == 'memory_bandwidth':
            return max(candidates)
        else:
            return candidates[-1] if len(candidates) > 0 else 4

    # Default: middle value
    return candidates[len(candidates) // 2] if len(candidates) > 0 else param_rules.get('default', 'N/A')


def _build_strategy_specific_notes(
    strategy: StrategyTemplate,
    diagnosis: PerformanceDiagnosis
) -> str:
    """Build strategy-specific implementation notes."""

    notes_lines = ["## Strategy-Specific Notes", ""]

    strategy_name = strategy.name.lower()

    if 'tiling' in strategy_name or 'tile' in strategy_name:
        notes_lines.extend([
            "**Tiling Strategy**:",
            "- Load tiles into shared memory to reduce global memory accesses",
            "- Ensure tile dimensions are multiples of warp size (32) for coalescing",
            "- Use `__syncthreads()` after loading tiles and before computation",
            "- Handle boundary cases when matrix dimensions are not multiples of tile size",
        ])

    elif 'warp' in strategy_name or 'shuffle' in strategy_name:
        notes_lines.extend([
            "**Warp-Level Optimization**:",
            "- Use `__shfl_down_sync()` or `__shfl_xor_sync()` for warp-level communication",
            "- No `__syncthreads()` needed within a warp",
            "- Ensure all threads in warp participate (use 0xffffffff mask)",
            "- Final reduction across warps may still need shared memory",
        ])

    elif 'vector' in strategy_name:
        notes_lines.extend([
            "**Vectorized Memory Access**:",
            "- Use `float4`, `int4`, or `float2` for vectorized loads/stores",
            "- Ensure memory addresses are aligned (16-byte for float4)",
            "- Cast pointers: `reinterpret_cast<float4*>(ptr)`",
            "- Handle remainder elements separately if size not divisible by vector width",
        ])

    elif 'tensor' in strategy_name:
        notes_lines.extend([
            "**Tensor Core Usage**:",
            "- Include `<mma.h>` and use `nvcuda::wmma` namespace",
            "- Matrix dimensions must be multiples of 16 (WMMA_M/N/K)",
            "- Use FP16 (`half`) for input matrices for best performance",
            "- Accumulator can be FP32 for better precision",
            "- Load/store with `wmma::load_matrix_sync()` and `wmma::store_matrix_sync()`",
        ])

    elif 'fusion' in strategy_name:
        notes_lines.extend([
            "**Kernel Fusion**:",
            "- Combine multiple operations in a single kernel",
            "- Eliminate intermediate memory writes",
            "- Watch for register pressure with complex fused operations",
            "- Consider using local variables to hold intermediate results",
        ])

    elif 'occupancy' in strategy_name:
        notes_lines.extend([
            "**Occupancy Optimization**:",
            "- Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)`",
            "- Reduce register usage by simplifying computations or using shared memory",
            "- Reduce shared memory usage if it's the limiting factor",
            "- Profile with different block sizes to find optimal configuration",
        ])

    # Add bottleneck-specific notes
    if diagnosis.bottleneck == 'memory_bandwidth':
        notes_lines.extend([
            "",
            "**Memory Bandwidth Bottleneck**:",
            "- Focus on reducing global memory traffic",
            "- Maximize data reuse through shared memory or registers",
            "- Ensure coalesced memory accesses",
        ])
    elif diagnosis.bottleneck == 'occupancy':
        notes_lines.extend([
            "",
            "**Occupancy Bottleneck**:",
            f"- Current occupancy: {diagnosis.achieved_occupancy:.1f}%",
            f"- Limiting factor: {diagnosis.occupancy_limiting_factor}",
            "- Adjust block size or reduce resource usage",
        ])

    return "\n".join(notes_lines)


def build_simple_optimization_prompt(
    operator_type: str,
    current_code: str,
    optimization_goal: str
) -> str:
    """
    Build a simple optimization prompt without strategy template (fallback mode).

    Args:
        operator_type: Type of operator
        current_code: Current kernel code
        optimization_goal: High-level optimization goal

    Returns:
        Formatted prompt string
    """

    prompt = f"""You are an expert CUDA kernel optimizer. Optimize the following kernel.

## Operator Type
{operator_type}

## Optimization Goal
{optimization_goal}

## Current Kernel Code

```python
{current_code}
```

## Your Task

Optimize this kernel to improve performance. Focus on:
- Memory access patterns (coalescing, bank conflicts)
- Compute efficiency (occupancy, instruction throughput)
- Resource utilization (registers, shared memory)

## Output Format

**CRITICAL**: You MUST output a complete Python file that can be executed with `python script.py --repeat 100`.

The output must include:
1. `import torch` and `import torch.nn as nn`
2. A `class ModelNew(nn.Module)` with `__init__` and `forward` methods
3. A `get_inputs()` function that returns input tensors
4. A `get_init_inputs()` function
5. A `if __name__ == "__main__":` block with timing code

Do NOT output raw CUDA kernel code. Use PyTorch operations or PyTorch CUDA extensions.

```python
# Your complete Python benchmark file here
```
"""

    return prompt
