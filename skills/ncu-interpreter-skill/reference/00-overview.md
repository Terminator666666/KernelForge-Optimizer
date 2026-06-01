# NCU Interpreter Overview

## Purpose

NCU Interpreter 将原始的 NVIDIA Nsight Compute (NCU) 性能指标转换为高层次的性能诊断，帮助 LLM 和开发者快速理解 CUDA kernel 的性能瓶颈。

**核心问题**：
- 原始 NCU 指标数量多（100+ 个指标）
- 指标含义复杂，需要深入的 GPU 架构知识
- LLM 难以直接从原始指标推断优化方向

**解决方案**：
- 计算派生指标（带宽利用率、算术强度等）
- 识别主要瓶颈类型
- 提供可操作的优化建议

## Core Capabilities

### 1. 派生指标计算

从原始 NCU 指标计算高层次的派生指标：

- **带宽利用率**：`实际带宽 / 理论峰值带宽`
- **算术强度**：`FLOPs / 内存访问字节数`
- **占用率**：`实际活跃 warps / 理论最大 warps`
- **SM 效率**：`有效计算周期 / 总周期`

这些派生指标更容易理解，直接反映性能特征。

### 2. 瓶颈识别

自动识别主要性能瓶颈：

- **Memory Bandwidth Bound**：内存带宽饱和，数据传输速度限制性能
- **Memory Latency Bound**：内存延迟高，等待数据时间长
- **Compute Bound**：计算单元饱和，ALU/FPU 利用率高
- **Low Occupancy**：SM 占用率低，并行度不足

每个瓶颈类型都有对应的优化策略。

### 3. Roofline 模型分析

使用 Roofline 模型判断 kernel 的性能特征：

- 计算算术强度（AI = FLOPs / Bytes）
- 确定 kernel 在 Roofline 图上的位置
- 判断是 memory-bound 还是 compute-bound
- 计算距离理论峰值的差距

Roofline 分析帮助确定优化的主要方向。

### 4. 内存访问模式识别

分析内存访问模式：

- **Coalesced**（合并访问）：相邻线程访问相邻内存，效率高
- **Strided**（跨步访问）：线程访问间隔固定，效率中等
- **Random**（随机访问）：访问模式不规则，效率低

访问模式直接影响内存带宽利用率。

### 5. 问题优先级排序

根据性能影响对问题进行优先级排序：

- **Critical**：严重影响性能（>30% 性能损失）
- **Important**：显著影响性能（10-30% 性能损失）
- **Minor**：轻微影响性能（<10% 性能损失）

优先解决 Critical 和 Important 问题。

## Workflow

NCU Interpreter 的典型工作流程：

```
1. 输入：NCU profiling 报告（原始指标）
   ↓
2. 解析：提取关键指标
   ↓
3. 计算：派生指标（带宽利用率、算术强度、占用率）
   ↓
4. 分析：
   - 内存子系统分析
   - 计算子系统分析
   - Roofline 模型分析
   ↓
5. 识别：主要瓶颈类型和置信度
   ↓
6. 生成：
   - 问题列表（按优先级排序）
   - 优化建议
   - 诊断报告
   ↓
7. 输出：PerformanceDiagnosis 对象
```

## Key Concepts

### 派生指标 vs 原始指标

**原始指标**：
- NCU 直接测量的值
- 例如：`dram__bytes_read.sum`、`sm__cycles_elapsed.avg`
- 数量多，含义复杂

**派生指标**：
- 从原始指标计算得出
- 例如：带宽利用率 = `实际带宽 / 峰值带宽`
- 数量少，含义清晰

### 瓶颈类型

**Memory Bandwidth Bound**：
- 特征：带宽利用率高（>80%），内存访问密集
- 优化方向：减少内存访问、提高数据重用

**Memory Latency Bound**：
- 特征：带宽利用率低，但内存延迟高
- 优化方向：提高占用率、隐藏延迟

**Compute Bound**：
- 特征：计算单元利用率高，算术强度高
- 优化方向：优化计算逻辑、使用更快的指令

**Low Occupancy**：
- 特征：SM 占用率低（<50%）
- 优化方向：增加并行度、减少资源使用

### Roofline 模型

Roofline 模型是性能分析的经典工具：

- **X 轴**：算术强度（FLOPs/Byte）
- **Y 轴**：性能（GFLOPS）
- **Roofline**：理论性能上限

**三个区域**：
1. **Memory-bound**：算术强度低，受内存带宽限制
2. **Compute-bound**：算术强度高，受计算能力限制
3. **Balanced**：接近两者的交界点

## Integration

NCU Interpreter 集成到 CUDA 优化工作流：

### 在 Agent 工作流中使用

```
1. Agent 运行 NCU profiling
2. Agent 调用 ncu-interpreter-skill
3. Skill 返回诊断结果
4. Agent 根据诊断选择优化策略（从 strategy-library-skill）
5. Agent 实现优化
6. Agent 重新 profiling 验证
```

### 与其他 Skills 协作

- **strategy-library-skill**：根据瓶颈类型推荐优化策略
- **optimization-history-skill**：记录历史诊断，检测瓶颈转移

### 输出格式

NCU Interpreter 输出 `PerformanceDiagnosis` 对象，包含：

```python
{
    "bottleneck": "memory_bandwidth",
    "bottleneck_confidence": 0.85,
    "memory_bandwidth_util": 78.5,
    "achieved_occupancy": 45.2,
    "arithmetic_intensity": 2.3,
    "roofline_region": "memory_bound",
    "issues": [
        {
            "severity": "critical",
            "category": "memory",
            "description": "Low memory bandwidth utilization",
            "suggestion": "Consider memory access coalescing"
        }
    ]
}
```

## Next Steps

查看其他参考文档了解详细实现：

- `01-derived-metrics.md` - 派生指标计算方法
- `02-memory-analysis.md` - 内存子系统分析
- `03-compute-analysis.md` - 计算子系统分析
- `04-roofline-model.md` - Roofline 模型实现
- `05-bottleneck-identification.md` - 瓶颈识别逻辑
- `06-access-patterns.md` - 访问模式识别
- `07-gpu-specs.md` - GPU 架构规格
