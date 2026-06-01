---
name: strategy-library-skill
description: CUDA 优化策略知识库，提供经过验证的优化模板、代码示例和参数选择规则
version: 1.0.0
author: KernelForge-Optimizer
tags: [cuda, optimization, strategy, templates]
---

# Strategy Library Skill

## 概述

Strategy Library Skill 提供经过验证的 CUDA 优化策略模板库，包含代码示例、参数选择规则和适用场景。帮助 LLM 和开发者快速选择和应用正确的优化策略。

## 核心功能

### 1. 优化策略模板

提供 9+ 个经过验证的优化策略：

- **matmul_tiling** - 矩阵乘法共享内存分块
- **reduction_warp_primitives** - Warp 级归约优化
- **vectorized_memory** - 向量化内存访问
- **kernel_fusion** - 算子融合
- **tensor_core** - Tensor Core 优化
- **bank_conflict_free** - 避免 Bank Conflict
- **occupancy_tuning** - 占用率优化
- **cooperative_groups** - 协作组优化
- **persistent_threads** - 持久线程优化

### 2. 策略匹配

根据以下条件自动匹配适用策略：

- **算子类型**：matmul, conv2d, elementwise, reduction, etc.
- **瓶颈类型**：memory bandwidth/latency, compute, occupancy
- **GPU 架构**：compute capability 要求
- **性能特征**：算术强度、带宽利用率等

### 3. 参数选择

为每个策略提供参数选择规则：

- **TILE_SIZE**：基于共享内存大小和寄存器压力
- **BLOCK_SIZE**：基于占用率和资源限制
- **VECTOR_SIZE**：基于数据对齐和内存带宽
- **ELEMENTS_PER_THREAD**：基于寄存器使用和并行度

### 4. 代码模板

提供完整的 CUDA 代码模板：

- 可直接使用的代码示例
- 参数化的模板变量
- 详细的实现注释
- 性能优化要点

## 使用方法

### 基本用法

```python
from strategy_library import StrategyLibrary

# 初始化策略库
library = StrategyLibrary()

# 根据条件查询适用策略
strategies = library.get_applicable_strategies(
    operator_type='matmul',
    bottleneck='memory_bandwidth',
    gpu_compute_capability=8.9
)

# 选择最佳策略
best_strategy = strategies[0]

# 获取参数建议
params = library.select_parameters(
    strategy_name='matmul_tiling',
    matrix_size=1024,
    gpu_specs={'shared_memory_per_sm': 102400}
)

# 实例化代码模板
code = library.instantiate_template(
    strategy_name='matmul_tiling',
    parameters=params
)
```

### 在 Agent 工作流中使用

1. **NCU Interpreter** 识别瓶颈
2. **Strategy Library** 推荐策略
3. **Agent** 实现优化代码
4. **Verification** 验证性能提升

## 策略选择流程

```
1. 识别算子类型
   ├─ matmul → matmul_tiling, tensor_core
   ├─ reduction → reduction_warp_primitives
   ├─ elementwise → vectorized_memory, kernel_fusion
   └─ ...

2. 识别瓶颈类型
   ├─ memory_bandwidth → vectorized_memory, kernel_fusion
   ├─ memory_latency → matmul_tiling, bank_conflict_free
   ├─ compute_bound → tensor_core
   └─ occupancy → occupancy_tuning

3. 检查 GPU 架构
   ├─ compute_capability >= 7.0 → tensor_core
   ├─ compute_capability >= 3.0 → warp_primitives
   └─ ...

4. 选择最佳策略
   └─ 根据预期加速比和适用性排序
```

## 策略详情

### matmul_tiling

**适用场景**：
- 算子：矩阵乘法
- 瓶颈：内存带宽、内存延迟
- 预期加速：2-5×

**关键参数**：
- `TILE_SIZE`: 16, 32, 64, 128（默认 32）
- `BLOCK_SIZE`: TILE_SIZE × TILE_SIZE

**优化原理**：
- 使用共享内存缓存数据块
- 减少全局内存访问次数
- 提高数据重用率

### reduction_warp_primitives

**适用场景**：
- 算子：归约操作（sum, max, min）
- 瓶颈：内存延迟、占用率
- 预期加速：3-8×

**关键参数**：
- `BLOCK_SIZE`: 128, 256, 512（默认 256）
- `ELEMENTS_PER_THREAD`: 1, 2, 4, 8（默认 4）

**优化原理**：
- 使用 warp shuffle 指令
- 消除共享内存原子操作
- 减少同步开销

### vectorized_memory

**适用场景**：
- 算子：Element-wise 操作、矩阵乘法
- 瓶颈：内存带宽
- 预期加速：1.5-3×

**关键参数**：
- `VECTOR_SIZE`: 2, 4（默认 4）

**优化原理**：
- 使用 float4/int4 向量化访问
- 减少内存事务数量
- 提高内存带宽利用率

### kernel_fusion

**适用场景**：
- 算子：多个 element-wise 操作
- 瓶颈：内存带宽、kernel 启动开销
- 预期加速：2-4×

**优化原理**：
- 合并多个 kernel 为一个
- 减少中间数据传输
- 降低 kernel 启动开销

### tensor_core

**适用场景**：
- 算子：矩阵乘法、卷积
- 瓶颈：计算能力
- 预期加速：2-10×（取决于精度）

**关键参数**：
- `PRECISION`: FP16, BF16, INT8
- `WMMA_M/N/K`: 16×16×16 或 8×8×4

**优化原理**：
- 使用 Tensor Core 专用硬件
- FP16/BF16 性能提升 2-4×
- INT8 性能提升 4-10×

## 参考文档

详细的策略说明和代码示例请参考：

- `reference/00-overview.md` - 策略库概述
- `reference/01-matmul-tiling.md` - 矩阵乘法分块
- `reference/02-reduction-warp.md` - Warp 级归约
- `reference/03-vectorized-memory.md` - 向量化访问
- `reference/04-kernel-fusion.md` - 算子融合
- `reference/05-tensor-core.md` - Tensor Core 优化

## 代码模板

完整的 CUDA 代码模板位于 `templates/` 目录：

- `matmul_tiling.cu` - 矩阵乘法分块模板
- `reduction_warp.cu` - Warp 归约模板
- `vectorized_memory.cu` - 向量化访问模板
- `kernel_fusion.cu` - 算子融合模板
- `tensor_core.cu` - Tensor Core 模板

## 注意事项

1. **GPU 架构兼容性**：检查 compute capability 要求
2. **内存对齐**：向量化访问需要正确的内存对齐
3. **参数调优**：根据具体硬件和数据大小调整参数
4. **性能验证**：使用 NCU 验证优化效果

## 扩展策略库

添加新策略的步骤：

1. 在 `reference/` 中创建策略文档
2. 在 `templates/` 中添加代码模板
3. 在 `data/strategy_rules.yaml` 中定义规则
4. 更新本文档的策略列表

## 参考资料

- CUDA C Best Practices Guide
- NVIDIA GPU Architecture Whitepapers
- Kernel Design Agents (KDA) - MIT HAN Lab
- CudaForge Optimization Framework
