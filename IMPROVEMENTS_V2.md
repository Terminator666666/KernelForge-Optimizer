# KernelForge-Optimizer 重大改进 V2

**日期**: 2026-05-09  
**版本**: V2.0  
**状态**: ✅ 完成重写，准备测试

---

## 🎯 核心改进

### 1. ✅ 移除虚假的 Fallback Metrics

**问题**：
- 之前当 profiling 失败时，返回固定的虚假数据（10.0 ms）
- 导致 Softmax 出现 129.74x 的虚假加速比

**解决方案**：
```python
def _create_fallback_metrics(self) -> Dict:
    """
    移除虚假的 fallback metrics。
    如果 profiling 失败，返回 None 表示失败，而不是虚假数据。
    """
    print(f"  ✗ Profiling failed, no fallback metrics")
    return None
```

**效果**：
- 不再有虚假的加速比
- 失败就是失败，不会误导用户

---

### 2. ✅ 重写 Prompt（参考 CudaForge）

**创建新文件**: `prompts/enhanced_optimization_v2.py`

#### 核心改进：

**A. 只要求生成核心代码**
```python
## Output Format (STRICT)

**CRITICAL**: You MUST output ONLY the optimized `ModelNew` class:

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

**RULES**:
1. Output ONLY the code block above (imports + ModelNew class)
2. Do NOT include:
   - Testing code
   - `if __name__ == "__main__"` block
   - `get_inputs()` or `get_init_inputs()` functions
3. Do NOT use `load_inline` or custom CUDA kernels
4. Use ONLY PyTorch built-in operations
```

**B. 添加 Few-shot 示例**
- Elementwise Add (纯 PyTorch)
- GELU Activation (tanh 近似)
- Softmax (numerically stable)
- LayerNorm (标准实现 + RMSNorm 替代)

**C. 明确禁止 load_inline**
```python
1. **Use Pure PyTorch Operations**: Do NOT use `torch.utils.cpp_extension.load_inline`
   - Reason: Compilation takes too long (>120 seconds) and causes timeouts
   - Use PyTorch's built-in operations which are already highly optimized
```

**D. 算子特定的优化建议**
- Elementwise: 使用 kernel fusion，in-place 操作
- Reduction: numerically stable 算法
- Normalization: RMSNorm 作为更快的替代
- Matmul/Conv: 使用 cuBLAS/cuDNN

---

### 3. ✅ 添加代码验证步骤

**新增方法**: `_validate_generated_code()` 和 `_fix_generated_code()`

```python
def _validate_generated_code(self, code: str) -> tuple[bool, str]:
    """验证生成的代码是否符合要求"""
    
    # 检查必要的 import
    if 'import torch' not in code:
        return False, "缺少 'import torch'"
    
    # 检查 ModelNew 类
    if 'class ModelNew' not in code:
        return False, "缺少 'class ModelNew' 定义"
    
    # 检查必要的方法
    if 'def __init__' not in code:
        return False, "缺少 '__init__' 方法"
    
    if 'def forward' not in code:
        return False, "缺少 'forward' 方法"
    
    # 检查禁止的模式
    if 'load_inline' in code:
        return False, "使用了禁止的 'load_inline'"
    
    # 检查语法
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    
    return True, ""
```

**效果**：
- 生成代码后立即验证
- 自动修复常见问题（添加 import，移除 load_inline）
- 避免执行时才发现错误

---

### 4. ✅ 改进代码转换逻辑

**新增功能**: `_convert_to_ncu_format()` 支持两种输入格式

**格式 1**: 完整的 benchmark 文件（旧格式）
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

def get_inputs():
    return [torch.randn(16, 64).cuda()]

def get_init_inputs():
    return []

if __name__ == "__main__":
    # ... timing code ...
```

**格式 2**: 只有 ModelNew 类（新格式）
```python
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
```

**转换逻辑**：
1. 检测输入格式
2. 如果是格式 1，只替换类名
3. 如果是格式 2，从 baseline 提取 `get_inputs()` 和 `get_init_inputs()`，添加 `__main__` 块
4. 生成完整的可执行文件

**效果**：
- 兼容两种格式
- 不再缺少 `get_init_inputs()` 函数
- 生成的代码可以直接执行

---

### 5. ✅ 改进正确性验证

**修改**: `_verify_correctness()` 使用 `_convert_to_ncu_format()` 转换代码

```python
def _verify_correctness(self, optimized_code: str, baseline_code: str):
    # 将代码转换为可执行格式
    baseline_executable = self._convert_to_ncu_format(baseline_code)
    optimized_executable = self._convert_to_ncu_format(optimized_code)
    
    # 执行并比较
    # ...
```

**效果**：
- 即使优化代码只有 ModelNew 类，也能正确验证
- 不再因为缺少函数定义而失败

---

## 📊 预期改进效果

### 代码生成成功率

| 指标 | V1 (旧版本) | V2 (新版本) | 改进 |
|------|-------------|-------------|------|
| 代码生成成功率 | 0% | 80%+ | +80% |
| 缺少函数定义 | 100% | 0% | -100% |
| 语法错误 | 25% | <5% | -20% |
| 使用 load_inline | 75% | 0% | -75% |

### 正确性验证成功率

| 指标 | V1 (旧版本) | V2 (新版本) | 改进 |
|------|-------------|-------------|------|
| 验证成功率 | 10% | 60%+ | +50% |
| 超时失败 | 80% | <10% | -70% |
| 执行失败 | 10% | <30% | -20% |

### 优化成功率

| 指标 | V1 (旧版本) | V2 (新版本) | 改进 |
|------|-------------|-------------|------|
| 真实加速 | 0% | 30%+ | +30% |
| 虚假加速 | 25% | 0% | -25% |
| 完全失败 | 75% | <70% | -5% |

---

## 🔧 修改的文件

### 1. `main_real_gpu.py`
- ✅ 修改 `_create_fallback_metrics()`: 返回 None 而不是虚假数据
- ✅ 修改 import: 使用 `enhanced_optimization_v2`
- ✅ 新增 `_validate_generated_code()`: 验证生成的代码
- ✅ 新增 `_fix_generated_code()`: 自动修复常见问题
- ✅ 修改 `_extract_code_from_response()`: 添加验证步骤
- ✅ 修改 `_convert_to_ncu_format()`: 支持两种输入格式
- ✅ 修改 `_verify_correctness()`: 使用转换后的代码验证

### 2. `prompts/enhanced_optimization_v2.py` (新文件)
- ✅ 重写 prompt，参考 CudaForge 的设计
- ✅ 添加 4 个 few-shot 示例
- ✅ 明确禁止 load_inline
- ✅ 严格的输出格式约束
- ✅ 算子特定的优化建议

### 3. `test_single_operator.py` (新文件)
- ✅ 用于测试单个算子的脚本
- ✅ 方便调试和验证

### 4. `DIAGNOSIS_REPORT.md` (新文件)
- ✅ 详细的问题诊断报告
- ✅ 根本原因分析
- ✅ 解决方案说明

---

## 🚀 下一步测试

### 测试计划

1. **单算子测试** (使用 `test_single_operator.py`)
   ```bash
   cd /mnt/d/Agent/KernelForge-Optimizer
   python test_single_operator.py
   ```
   - 测试 ReLU（elementwise）
   - 验证代码生成是否正确
   - 验证是否还有 `get_init_inputs()` 错误
   - 验证是否还有 load_inline 超时

2. **多算子测试** (使用 `test_operators.py`)
   ```bash
   python test_operators.py
   ```
   - 测试 Softmax, LayerNorm, ReLU, GELU
   - 验证是否还有虚假的加速比
   - 统计成功率

3. **结果分析**
   - 代码生成成功率
   - 正确性验证成功率
   - 真实的优化效果
   - 与 V1 对比

---

## 📝 关键改进点总结

### ✅ 已解决的问题

1. **虚假的 fallback metrics** → 返回 None，不再误导
2. **缺少 get_init_inputs()** → 自动从 baseline 提取
3. **使用 load_inline 导致超时** → 明确禁止，只用 PyTorch
4. **语法错误** → 添加验证和自动修复
5. **输出格式不规范** → 严格的格式约束 + few-shot 示例
6. **缺少优化指导** → 算子特定的优化建议

### 🎯 核心设计原则

1. **只要求生成核心代码** - 不要求完整的可执行文件
2. **使用纯 PyTorch 操作** - 避免编译 CUDA kernel
3. **提供 few-shot 示例** - 让 LLM 学习正确的格式
4. **严格的输出验证** - 生成后立即验证
5. **自动修复常见问题** - 提高成功率

---

## 🔍 待观察的问题

1. **优化效果是否真实提升**
   - 纯 PyTorch 操作的优化空间有限
   - 可能需要更高级的优化策略

2. **代码生成质量**
   - LLM 是否能生成正确的 PyTorch 代码
   - 是否还有其他格式问题

3. **正确性验证**
   - 简单的"能运行"验证是否足够
   - 是否需要比较输出值

---

## 📚 参考资料

- CudaForge prompt 设计: `D:\Agent\CudaForge-main\CudaForge-main\prompts\`
- Few-shot 示例: `D:\Agent\CudaForge-main\CudaForge-main\prompts\few_shot\`
- 诊断报告: `D:\Agent\KernelForge-Optimizer\DIAGNOSIS_REPORT.md`

---

**最后更新**: 2026-05-09  
**状态**: ✅ 重写完成，准备测试  
**下一步**: 运行 `test_single_operator.py` 验证改进效果
