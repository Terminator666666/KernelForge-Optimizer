"""
Enhanced Optimization Prompt V2 - 重写版本，参考 CudaForge 的设计

核心改进：
1. 只要求生成核心代码，不要求完整的可执行文件
2. 提供完整的代码模板和 few-shot 示例
3. 明确禁止使用 load_inline（避免编译超时）
4. 严格的输出格式约束
5. 添加代码验证步骤
"""

from typing import Dict, List, Optional
from agents.strategy_templates import StrategyTemplate
from agents.ncu_interpreter import PerformanceDiagnosis


# Few-shot 示例 1: Elementwise Add (纯 PyTorch)
EXAMPLE_ELEMENTWISE_ADD = """
# Example 1: Elementwise Addition (Pure PyTorch)

## Original Code (Model):
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b
```

## Optimized Code (ModelNew):
```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        # 使用 fused add 操作，减少内存访问
        return torch.add(a, b)
```

Note: This is a simple example. For elementwise operations, PyTorch is already highly optimized.
"""


# Few-shot 示例 2: GELU Activation (纯 PyTorch)
EXAMPLE_GELU = """
# Example 2: GELU Activation (Pure PyTorch)

## Original Code (Model):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)
```

## Optimized Code (ModelNew):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 使用 tanh 近似版本的 GELU，计算更快
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608 * (x + 0.044715 * x * x * x)
        ))
```

Note: This uses the tanh approximation which is faster than the exact GELU.
"""


# Few-shot 示例 3: Softmax (纯 PyTorch)
EXAMPLE_SOFTMAX = """
# Example 3: Softmax (Pure PyTorch)

## Original Code (Model):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softmax(x, dim=-1)
```

## Optimized Code (ModelNew):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 使用 numerically stable softmax
        # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        x_max = torch.max(x, dim=-1, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
```

Note: This is numerically stable and avoids overflow. PyTorch's F.softmax already does this internally.
"""


# Few-shot 示例 4: LayerNorm (纯 PyTorch)
EXAMPLE_LAYERNORM = """
# Example 4: Layer Normalization (Pure PyTorch)

## Original Code (Model):
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        return self.ln(x)
```

## Optimized Code (ModelNew):
```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        # 使用 PyTorch 的 LayerNorm，但可以尝试 RMSNorm 作为更快的替代
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        # 标准 LayerNorm
        return self.ln(x)

        # 或者使用 RMSNorm（更快但精度略低）：
        # variance = x.pow(2).mean(-1, keepdim=True)
        # x = x * torch.rsqrt(variance + 1e-5)
        # return x * self.ln.weight + self.ln.bias
```

Note: For LayerNorm, PyTorch's implementation is already highly optimized. RMSNorm is a faster alternative.
"""


def build_enhanced_optimization_prompt(
    diagnosis: PerformanceDiagnosis,
    strategy: Optional[StrategyTemplate],
    operator_type: str,
    current_code: str,
    gpu_specs: Dict[str, any],
    judge_recommendations: Optional[str] = None
) -> str:
    """
    构建增强的优化 prompt（V2 版本）

    核心改进：
    1. 只要求生成 ModelNew 类，不要求完整的可执行文件
    2. 提供 few-shot 示例
    3. 明确禁止 load_inline
    4. 严格的输出格式约束
    """

    # 构建性能诊断摘要
    diagnosis_summary = _build_diagnosis_summary(diagnosis)

    # 构建策略指导（如果有）
    if strategy:
        strategy_guidance = f"""
## Selected Optimization Strategy: {strategy.name}

**Description**: {strategy.description}

**Expected Speedup**: {strategy.expected_speedup[0]:.1f}x - {strategy.expected_speedup[1]:.1f}x

**Implementation Notes**:
{strategy.notes}

**Key Parameters**:
{_build_parameter_summary(strategy)}
"""
    else:
        strategy_guidance = """
## Optimization Goal

No specific strategy template matched. Focus on general performance optimization:
- Reduce memory bandwidth bottlenecks
- Improve compute utilization
- Optimize memory access patterns
- Increase occupancy if needed
"""

    # 构建 Judge 上下文（如果有）
    judge_context = ""
    if judge_recommendations:
        judge_context = f"""
## Judge Analysis

{judge_recommendations}

---
"""

    # 选择相关的 few-shot 示例
    examples = _select_relevant_examples(operator_type)

    prompt = f"""You are an expert CUDA kernel optimizer. Your task is to optimize the following PyTorch model.

## Target GPU
- **Name**: {gpu_specs.get('name', 'Unknown')}
- **Architecture**: {gpu_specs.get('architecture', 'Unknown')}
- **Compute Capability**: {gpu_specs.get('compute_capability', 'Unknown')}
- **Memory Bandwidth**: {gpu_specs.get('memory_bandwidth_gb_s', 'Unknown')} GB/s
- **Peak FP32**: {gpu_specs.get('peak_fp32_tflops', 'Unknown')} TFLOPS

## Operator Information
- **Type**: {operator_type}

## Performance Diagnosis

{diagnosis_summary}

{judge_context}

{strategy_guidance}

## Current Code

```python
{current_code}
```

## Few-Shot Examples

{examples}

## Your Task

Optimize the model to improve performance on the target GPU. Follow these guidelines:

1. **Use Pure PyTorch Operations**: Do NOT use `torch.utils.cpp_extension.load_inline` or custom CUDA kernels
   - Reason: Compilation takes too long (>120 seconds) and causes timeouts
   - Use PyTorch's built-in operations which are already highly optimized

2. **Preserve Correctness**: The optimized model must produce the same results as the original
   - Same input/output shapes
   - Same data types
   - Numerical accuracy within atol=1e-4, rtol=1e-4

3. **Focus on Algorithmic Optimizations**:
   - Kernel fusion (combine multiple operations)
   - Memory access patterns (coalescing, vectorization)
   - Numerical stability (avoid overflow/underflow)
   - Use faster approximations when appropriate

4. **Preserve the Model Interface**:
   - Keep the same `__init__` signature
   - Keep the same `forward` signature
   - The model should be a drop-in replacement

## Output Format (STRICT)

**CRITICAL**: You MUST output ONLY the optimized `ModelNew` class in the following format:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, <same parameters as original Model>):
        super().__init__()
        # Your initialization code here

    def forward(self, <same parameters as original Model>):
        # Your optimized forward pass here
        return <output>
```

**RULES**:
1. Output ONLY the code block above (imports + ModelNew class)
2. Do NOT include:
   - Testing code
   - `if __name__ == "__main__"` block
   - `get_inputs()` or `get_init_inputs()` functions
   - Comments explaining what you did (put comments in the code itself)
3. Do NOT use `load_inline` or custom CUDA kernels
4. Use ONLY PyTorch built-in operations
5. The code must be syntactically correct Python

## Important Notes

- **Memory Bottleneck**: If the bottleneck is memory bandwidth, focus on reducing memory traffic
  - Use kernel fusion to eliminate intermediate tensors
  - Use in-place operations when possible
  - Optimize memory access patterns

- **Compute Bottleneck**: If the bottleneck is compute, focus on reducing FLOPs
  - Use faster approximations (e.g., tanh GELU instead of exact GELU)
  - Simplify mathematical expressions
  - Use lower precision when appropriate (FP16)

- **Occupancy Bottleneck**: If occupancy is low, the kernel may be too complex
  - Simplify the computation
  - Reduce register pressure
  - Use smaller block sizes

{_build_operator_specific_notes(operator_type, diagnosis)}

Now generate the optimized code following the format above.
"""

    return prompt


def _build_diagnosis_summary(diagnosis: PerformanceDiagnosis) -> str:
    """构建性能诊断摘要"""

    summary_lines = [
        f"**Primary Bottleneck**: {diagnosis.bottleneck} (confidence: {diagnosis.bottleneck_confidence:.0%})",
        "",
        "**Key Metrics**:",
        f"- Memory bandwidth utilization: {diagnosis.memory_bandwidth_util:.1f}%",
        f"- Compute utilization (SM efficiency): {diagnosis.sm_efficiency:.1f}%",
        f"- Occupancy: {diagnosis.achieved_occupancy:.1f}% (theoretical: {diagnosis.theoretical_occupancy:.1f}%)",
        f"- Arithmetic intensity: {diagnosis.arithmetic_intensity:.2f} FLOPs/byte",
        f"- Roofline region: {diagnosis.roofline_region}",
    ]

    # 添加主要问题
    if diagnosis.issues:
        summary_lines.append("")
        summary_lines.append("**Top Issues**:")
        for issue in diagnosis.issues[:3]:
            summary_lines.append(f"- {issue['description']}")

    return "\n".join(summary_lines)


def _build_parameter_summary(strategy: StrategyTemplate) -> str:
    """构建参数摘要"""

    if not strategy.parameter_rules:
        return "No specific parameters"

    lines = []
    for param_name, param_rules in strategy.parameter_rules.items():
        candidates = param_rules.get('candidates', [])
        default = param_rules.get('default', 'N/A')
        lines.append(f"- {param_name}: candidates={candidates}, default={default}")

    return "\n".join(lines)


def _select_relevant_examples(operator_type: str) -> str:
    """根据算子类型选择相关的 few-shot 示例"""

    operator_type_lower = operator_type.lower()

    examples = []

    # 总是包含 elementwise add 作为基础示例
    examples.append(EXAMPLE_ELEMENTWISE_ADD)

    # 根据算子类型添加相关示例
    if 'elementwise' in operator_type_lower or 'relu' in operator_type_lower or 'gelu' in operator_type_lower:
        examples.append(EXAMPLE_GELU)

    if 'reduction' in operator_type_lower or 'softmax' in operator_type_lower:
        examples.append(EXAMPLE_SOFTMAX)

    if 'layernorm' in operator_type_lower or 'norm' in operator_type_lower:
        examples.append(EXAMPLE_LAYERNORM)

    # 限制最多 3 个示例
    return "\n\n".join(examples[:3])


def _build_operator_specific_notes(operator_type: str, diagnosis: PerformanceDiagnosis) -> str:
    """构建算子特定的优化建议"""

    operator_type_lower = operator_type.lower()

    notes = []

    if 'elementwise' in operator_type_lower:
        notes.append("""
**Elementwise Operations**:
- PyTorch's elementwise ops are already highly optimized
- Focus on kernel fusion to reduce memory traffic
- Use in-place operations when possible (e.g., `x.add_(y)` instead of `x + y`)
- Consider using `torch.jit.script` for automatic fusion
""")

    if 'reduction' in operator_type_lower or 'softmax' in operator_type_lower:
        notes.append("""
**Reduction Operations**:
- Use numerically stable algorithms (e.g., subtract max before exp)
- Specify the reduction dimension explicitly
- Consider using `torch.jit.script` for fusion with surrounding ops
- For softmax, PyTorch's implementation is already optimal
""")

    if 'layernorm' in operator_type_lower or 'norm' in operator_type_lower:
        notes.append("""
**Normalization Operations**:
- PyTorch's LayerNorm is already highly optimized
- Consider RMSNorm as a faster alternative (removes mean centering)
- Use `torch.nn.functional.layer_norm` for more control
- Fuse normalization with surrounding operations when possible
""")

    if 'matmul' in operator_type_lower or 'conv' in operator_type_lower:
        notes.append("""
**Matrix/Convolution Operations**:
- PyTorch uses cuBLAS/cuDNN which are already optimal
- Focus on reducing data movement before/after the operation
- Use appropriate data types (FP16 for Tensor Cores)
- Consider using `torch.backends.cudnn.benchmark = True`
""")

    return "\n".join(notes) if notes else ""


def build_simple_optimization_prompt(
    operator_type: str,
    current_code: str,
    optimization_goal: str
) -> str:
    """
    构建简单的优化 prompt（fallback 模式）
    """

    prompt = f"""You are an expert CUDA kernel optimizer. Optimize the following PyTorch model.

## Operator Type
{operator_type}

## Optimization Goal
{optimization_goal}

## Current Code

```python
{current_code}
```

## Your Task

Optimize this model to improve performance. Use ONLY PyTorch built-in operations.

**CRITICAL**: Output ONLY the optimized `ModelNew` class:

```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, <same parameters>):
        super().__init__()
        # Your code here

    def forward(self, <same parameters>):
        # Your optimized code here
        return <output>
```

Do NOT use `load_inline` or custom CUDA kernels. Do NOT include testing code.
"""

    return prompt
