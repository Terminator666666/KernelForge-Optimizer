# ncu-interpreter-skill

将原始 NVIDIA Nsight Compute (NCU) 指标转换为高层次性能诊断。

## Overview

ncu-interpreter-skill 帮助理解 CUDA kernel 的 NCU profiling 结果，自动识别性能瓶颈并提供优化建议。

**核心功能**：
- 计算派生指标（带宽利用率、算术强度、占用率）
- 识别瓶颈类型（memory bandwidth/latency, compute, occupancy）
- Roofline 模型分析
- 内存访问模式识别（coalesced/strided/random）
- 问题优先级排序

## Installation

此 skill 是 KernelForge-Optimizer 的一部分。

```bash
# 链接到 Claude Code
ln -s "$(pwd)/skills/ncu-interpreter-skill" ~/.claude/skills/ncu-interpreter-skill

# 安装 Python 依赖（如需要）
pip install pyyaml numpy
```

## Usage

### 在 Claude Code 中使用

当分析 NCU 报告时，此 skill 会自动可用。触发词包括：
- "分析这个 NCU 报告"
- "为什么这个 kernel 慢"
- "瓶颈在哪里"

### 直接使用 Python 工具

```bash
cd skills/ncu-interpreter-skill

# 生成完整诊断
python helpers/generate_diagnosis.py /path/to/report.ncu-rep

# 只计算派生指标
python helpers/analyze_ncu_report.py /path/to/report.ncu-rep

# 只识别瓶颈
python helpers/identify_bottleneck.py /path/to/report.ncu-rep
```

## File Structure

```
ncu-interpreter-skill/
├── SKILL.md                    # Skill 入口点
├── README.md                   # 本文件
├── reference/                  # 参考文档（原理和方法）
│   ├── 00-overview.md
│   ├── 01-derived-metrics.md
│   ├── 02-memory-analysis.md
│   ├── 03-compute-analysis.md
│   ├── 04-roofline-model.md
│   ├── 05-bottleneck-identification.md
│   ├── 06-access-patterns.md
│   └── 07-gpu-specs.md
├── helpers/                    # Python 分析工具
│   ├── analyze_ncu_report.py   # 解析 NCU 报告，计算派生指标
│   ├── identify_bottleneck.py  # 瓶颈识别
│   ├── detect_access_pattern.py # 访问模式检测
│   └── generate_diagnosis.py   # 生成完整诊断
└── data/
    └── gpu_specs.yaml          # GPU 架构规格数据库
```

## Key Concepts

### 派生指标

- **带宽利用率** = 实际带宽 / 理论峰值带宽
- **算术强度** = FLOPs / 内存访问字节数
- **占用率** = 实际活跃 warps / 理论最大 warps

### 瓶颈类型

1. **Memory Bandwidth Bound** - 内存带宽饱和
2. **Memory Latency Bound** - 内存延迟高
3. **Compute Bound** - 计算单元饱和
4. **Low Occupancy** - SM 占用率低

### Roofline 模型

Roofline 模型帮助判断 kernel 是 memory-bound 还是 compute-bound：
- 算术强度低 + 性能低 → memory-bound
- 算术强度高 + 性能低 → compute-bound

## Supported GPUs

支持的 GPU 架构（规格存储在 `data/gpu_specs.yaml`）：
- RTX 50/40/30/20 系列
- A100, H100, V100, T4

## Examples

查看 `reference/` 目录下的文档获取详细示例和原理说明。

## Contributing

欢迎贡献！改进方向：
- 添加更多 GPU 架构支持
- 改进瓶颈识别算法
- 添加更多派生指标

## License

MIT License - 与 KernelForge-Optimizer 相同
