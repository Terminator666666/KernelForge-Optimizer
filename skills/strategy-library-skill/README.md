# Strategy Library Skill

CUDA 优化策略知识库，提供经过验证的优化模板、代码示例和参数选择规则。

## 概述

Strategy Library Skill 是 KernelForge-Optimizer 的核心组件之一，提供：

- **9+ 个优化策略模板**：经过验证的 CUDA 优化技术
- **智能策略匹配**：根据算子类型、瓶颈、GPU 架构自动推荐
- **参数选择规则**：为每个策略提供参数调优指导
- **完整代码模板**：可直接使用的 CUDA 代码示例

## 安装

```bash
# 链接到 Claude Code skills 目录
ln -s "$(pwd)" ~/.claude/skills/strategy-library-skill

# 或者复制到 skills 目录
cp -r . ~/.claude/skills/strategy-library-skill
```

## 快速开始

### 在 Claude Code 中使用

当 Claude Code 分析 CUDA kernel 性能时，会自动调用此 skill 推荐优化策略。

### 查询适用策略

```
User: 我的矩阵乘法 kernel 有内存带宽瓶颈，GPU 是 RTX 4090，如何优化？

Claude: [调用 strategy-library-skill]
推荐策略：
1. matmul_tiling - 共享内存分块（预期加速 2-5×）
2. vectorized_memory - 向量化访问（预期加速 1.5-3×）
3. tensor_core - 使用 Tensor Cores（预期加速 2-10×）

最佳策略：matmul_tiling
参数建议：TILE_SIZE=32, BLOCK_SIZE=1024
```

### 获取代码模板

```
User: 给我 matmul_tiling 的完整代码

Claude: [调用 strategy-library-skill，返回 templates/matmul_tiling.cu]
```

## 支持的优化策略

| 策略 | 适用算子 | 瓶颈类型 | 预期加速 |
|-----|---------|---------|---------|
| matmul_tiling | MatMul | Memory BW/Latency | 2-5× |
| reduction_warp_primitives | Reduction | Memory Latency | 3-8× |
| vectorized_memory | Element-wise | Memory BW | 1.5-3× |
| kernel_fusion | Multiple ops | Memory BW | 2-4× |
| tensor_core | MatMul/Conv | Compute | 2-10× |
| bank_conflict_free | Shared memory | Memory Latency | 1.5-3× |
| occupancy_tuning | Any | Low Occupancy | 1.5-4× |
| cooperative_groups | Complex sync | Occupancy | 1.2-2× |
| persistent_threads | Small kernels | Launch overhead | 2-5× |

## 目录结构

```
strategy-library-skill/
├── SKILL.md                    # Skill 定义和使用说明
├── README.md                   # 本文件
├── reference/                  # 策略参考文档
│   ├── 00-overview.md         # 策略库概述
│   ├── 01-matmul-tiling.md    # 矩阵乘法分块
│   ├── 02-reduction-warp.md   # Warp 级归约
│   ├── 03-vectorized-memory.md # 向量化访问
│   ├── 04-kernel-fusion.md    # 算子融合
│   └── 05-tensor-core.md      # Tensor Core 优化
├── templates/                  # CUDA 代码模板
│   ├── matmul_tiling.cu       # 矩阵乘法分块模板
│   ├── reduction_warp.cu      # Warp 归约模板
│   ├── vectorized_memory.cu   # 向量化访问模板
│   ├── kernel_fusion.cu       # 算子融合模板
│   └── tensor_core.cu         # Tensor Core 模板
└── data/                       # 策略规则数据
    └── strategy_rules.yaml    # 策略匹配规则

```

## 策略选择流程

```
输入：算子类型、瓶颈类型、GPU 架构
  ↓
策略匹配：根据适用条件筛选策略
  ↓
策略排序：根据预期加速比排序
  ↓
参数选择：根据硬件规格选择参数
  ↓
输出：推荐策略 + 参数 + 代码模板
```

## 关键概念

### 算子类型 (Operator Type)

- **matmul** - 矩阵乘法
- **conv2d** - 2D 卷积
- **elementwise** - 逐元素操作（ReLU, Add, Mul）
- **reduction** - 归约操作（Sum, Max, Min）
- **transpose** - 矩阵转置
- **softmax** - Softmax 激活
- **layernorm** - Layer Normalization

### 瓶颈类型 (Bottleneck Type)

- **memory_bandwidth** - 内存带宽饱和
- **memory_latency** - 内存访问延迟高
- **compute_bound** - 计算能力限制
- **occupancy** - SM 占用率过低

### GPU 架构 (Compute Capability)

- **3.0+** - Kepler（支持 warp shuffle）
- **7.0+** - Volta（支持 Tensor Cores）
- **8.0+** - Ampere（改进的 Tensor Cores）
- **8.9+** - Ada Lovelace（第 4 代 Tensor Cores）
- **9.0+** - Hopper（Transformer Engine）

## 使用示例

### 示例 1：优化矩阵乘法

**场景**：1024×1024 矩阵乘法，内存带宽瓶颈，RTX 4090

**推荐策略**：matmul_tiling

**参数**：
- TILE_SIZE = 32
- BLOCK_SIZE = 1024 (32×32)

**预期效果**：
- 全局内存访问减少 32×
- 性能提升 3-4×

### 示例 2：优化归约操作

**场景**：大数组求和，内存延迟瓶颈，A100

**推荐策略**：reduction_warp_primitives

**参数**：
- BLOCK_SIZE = 256
- ELEMENTS_PER_THREAD = 4

**预期效果**：
- 消除共享内存原子操作
- 性能提升 5-7×

### 示例 3：优化 Element-wise 操作

**场景**：ReLU + Add + Mul 三个操作，内存带宽瓶颈

**推荐策略**：kernel_fusion + vectorized_memory

**参数**：
- VECTOR_SIZE = 4 (float4)

**预期效果**：
- 减少 2 次 kernel 启动
- 减少 2 次中间数据传输
- 性能提升 3-5×

## 与其他 Skills 的协作

### 与 ncu-interpreter-skill

1. NCU Interpreter 分析性能，识别瓶颈
2. Strategy Library 根据瓶颈推荐策略
3. Agent 实现优化代码

### 与 optimization-history-skill

1. Strategy Library 推荐策略
2. Agent 实现并测试
3. Optimization History 记录效果
4. 下次优化时参考历史数据

## 扩展策略库

### 添加新策略

1. **创建参考文档**：`reference/XX-strategy-name.md`
2. **编写代码模板**：`templates/strategy_name.cu`
3. **定义匹配规则**：在 `data/strategy_rules.yaml` 中添加
4. **更新文档**：更新 SKILL.md 和 README.md

### 策略文档模板

```markdown
# Strategy Name

## 概述
简要说明策略的优化原理

## 适用场景
- 算子类型
- 瓶颈类型
- GPU 架构要求

## 参数说明
- 参数 1：说明和选择规则
- 参数 2：说明和选择规则

## 代码示例
完整的 CUDA 代码

## 性能分析
预期加速比和优化效果

## 注意事项
使用限制和注意事项
```

## 常见问题

### Q: 如何选择 TILE_SIZE？

A: 考虑以下因素：
- 共享内存大小：TILE_SIZE² × sizeof(type) ≤ 48KB
- 寄存器压力：更大的 tile 需要更多寄存器
- 矩阵大小：最好能整除矩阵维度
- 推荐值：16, 32, 64（32 是常用默认值）

### Q: 什么时候使用 Tensor Cores？

A: 满足以下条件时使用：
- GPU 支持（compute capability ≥ 7.0）
- 算子是矩阵乘法或卷积
- 可以使用 FP16/BF16/INT8 精度
- 矩阵维度是 8 或 16 的倍数

### Q: Kernel Fusion 适用于哪些场景？

A: 适用于：
- 多个 element-wise 操作串联
- 中间结果只使用一次
- 内存带宽是瓶颈
- 不适用于需要全局同步的操作

## 参考资料

- [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)
- [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
- [CudaForge](https://github.com/CudaForge/CudaForge)

## 贡献

欢迎贡献新的优化策略！请参考"扩展策略库"部分。

## 许可证

MIT License
