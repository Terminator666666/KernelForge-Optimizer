---
name: ncu-interpreter-skill
description: 将原始 NCU 指标转换为高层次性能诊断。用于分析 CUDA kernel 的 NCU profiling 结果，识别瓶颈（memory bandwidth/latency, compute, occupancy），计算派生指标，进行 Roofline 分析。当用户需要理解 NCU 报告或诊断性能问题时使用。
argument-hint: "[ncu-report-path] | [--metric bandwidth_util] | [--gpu RTX4090]"
allowed-tools: "Bash Read Grep Glob"
---

# ncu-interpreter-skill — NCU 指标解释和瓶颈诊断

将原始 NVIDIA Nsight Compute (NCU) 指标转换为高层次性能诊断，帮助识别 CUDA kernel 的性能瓶颈。

## When To Use This Skill

当用户需要：
- 分析 NCU profiling 报告
- 识别 kernel 性能瓶颈（memory/compute/occupancy）
- 计算派生指标（带宽利用率、算术强度、占用率）
- 进行 Roofline 模型分析
- 识别内存访问模式（coalesced/strided/random）
- 获取优化建议和问题优先级

**触发词**：
- "分析这个 NCU 报告"
- "为什么这个 kernel 慢"
- "瓶颈在哪里"
- "带宽利用率是多少"
- "Roofline 分析"

## How To Use

### 基本用法

```bash
# 在 skill 目录中
cd skills/ncu-interpreter-skill

# 使用 Python 工具分析 NCU 报告
python helpers/analyze_ncu_report.py /path/to/report.ncu-rep

# 识别瓶颈
python helpers/identify_bottleneck.py /path/to/report.ncu-rep

# 检测访问模式
python helpers/detect_access_pattern.py /path/to/report.ncu-rep

# 生成完整诊断
python helpers/generate_diagnosis.py /path/to/report.ncu-rep
```

### 参考文档

查阅 `reference/` 目录下的文档了解详细原理：
- `00-overview.md` - NCU 解释器概述
- `01-derived-metrics.md` - 派生指标计算方法
- `02-memory-analysis.md` - 内存子系统分析规则
- `03-compute-analysis.md` - 计算子系统分析规则
- `04-roofline-model.md` - Roofline 模型原理
- `05-bottleneck-identification.md` - 瓶颈识别逻辑
- `06-access-patterns.md` - 访问模式识别
- `07-gpu-specs.md` - GPU 架构规格

## Key Features

### 1. 派生指标计算
- **带宽利用率**：实际带宽 / 理论峰值带宽
- **算术强度**：FLOPs / 内存访问字节数
- **占用率**：实际活跃 warps / 理论最大 warps

### 2. 瓶颈识别
- Memory Bandwidth Bound（内存带宽瓶颈）
- Memory Latency Bound（内存延迟瓶颈）
- Compute Bound（计算瓶颈）
- Low Occupancy（低占用率）

### 3. Roofline 模型分析
- 计算 kernel 在 Roofline 图上的位置
- 判断是 memory-bound 还是 compute-bound
- 提供优化方向建议

### 4. 访问模式识别
- Coalesced（合并访问）
- Strided（跨步访问）
- Random（随机访问）

### 5. 问题优先级排序
根据性能影响对问题进行优先级排序，提供可操作的优化建议。

## Output Pattern

诊断输出包含：

1. **Summary** - 性能摘要（GFLOPS, 带宽利用率, 占用率）
2. **Bottleneck** - 主要瓶颈类型和严重程度
3. **Derived Metrics** - 派生指标详情
4. **Memory Analysis** - 内存子系统分析
5. **Compute Analysis** - 计算子系统分析
6. **Roofline Position** - Roofline 模型位置
7. **Issues** - 按优先级排序的问题列表
8. **Recommendations** - 优化建议

## Supported GPUs

- NVIDIA RTX 50 Series (Blackwell)
- NVIDIA RTX 40 Series (Ada Lovelace): 4090, 4080, 4070
- NVIDIA RTX 30 Series (Ampere): 3090, 3080, 3070
- NVIDIA Data Center: A100, H100, V100, T4
- NVIDIA RTX 20 Series (Turing): 2080 Ti

GPU 规格存储在 `data/gpu_specs.yaml`。

## Example Workflow

```
1. 用户运行 NCU profiling：ncu --set full kernel.cu
2. 用户请求分析：「分析这个 NCU 报告」
3. Skill 读取 NCU 报告
4. 计算派生指标
5. 识别瓶颈
6. 进行 Roofline 分析
7. 生成诊断报告
8. 提供优化建议
```
