# KernelForge-Optimizer KDA 对齐改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 KernelForge-Optimizer 从自动化 CUDA 优化工具改造为基于 kernel-design-agents-main 的 Agent 中心化工作流

**Architecture:** 采用轻量级单 Agent + Skills 架构，完全对齐 KDA。将现有 3500 行代码转换为 3 个专业 skills（混合型：文档 + Python 工具），实现工作空间分离和证据驱动的优化流程。

**Tech Stack:** Python 3.8+, Markdown, YAML, CUDA, NCU (Nsight Compute)

**设计文档:** `docs/superpowers/specs/2026-06-01-kda-alignment-design.md`

---

## 文件结构规划

### 新增目录结构
```
KernelForge-Optimizer/
├── docs/
│   ├── agent-flow.md                     # CUDA 优化工作流
│   └── superpowers/
│       ├── specs/                        # 设计文档（已存在）
│       └── plans/                        # 实施计划（本文件）
├── prompts/
│   ├── README.md                         # Prompt 使用说明
│   └── kernel-optimization-flow.md       # CUDA 优化 prompt 模板
├── skills/
│   ├── ncu-interpreter-skill/
│   │   ├── SKILL.md
│   │   ├── README.md
│   │   ├── reference/                    # 7 个参考文档
│   │   ├── helpers/                      # 4 个 Python 工具
│   │   └── data/gpu_specs.yaml
│   ├── strategy-library-skill/
│   │   ├── SKILL.md
│   │   ├── README.md
│   │   ├── index.md
│   │   ├── wiki/
│   │   │   ├── strategies/               # 9 个策略文档
│   │   │   └── patterns/                 # 4 个问题模式文档
│   │   ├── scripts/                      # 3 个查询工具
│   │   └── data/
│   └── optimization-history-skill/
│       ├── SKILL.md
│       ├── README.md
│       ├── reference/                    # 4 个参考文档
│       ├── helpers/                      # 5 个 Python 工具
│       └── data/history_schema.yaml
└── examples/
    └── matmul-optimization/              # 完整示例
        ├── docs/
        ├── kernels/
        ├── profile/
        ├── tests/
        ├── benchmark.csv
        ├── candidates.jsonl
        └── README.md
```

### 需要修改的文件
- `README.md` - 完全重写，对齐 KDA 风格
- `CLAUDE.md` - 更新 Agent 工作规范

### 需要删除/移动的文件
- `agents/` - 旧代码，迁移到 skills 后删除
- `utils/` - 部分集成到 skills，部分删除
- `prompts/enhanced_*.py` - 删除
- `main_enhanced.py` - 移到 examples/ 或删除

---

## 阶段 1：基础架构搭建（2-3 天）

### Task 1.1: 创建目录结构

**Files:**
- Create: `docs/agent-flow.md`
- Create: `prompts/README.md`
- Create: `prompts/kernel-optimization-flow.md`
- Create: `skills/.gitkeep`
- Create: `examples/.gitkeep`

- [ ] **Step 1: 创建 docs 目录和 agent-flow.md**

```bash
mkdir -p docs
touch docs/agent-flow.md
```

- [ ] **Step 2: 创建 prompts 目录和文件**

```bash
mkdir -p prompts
touch prompts/README.md
touch prompts/kernel-optimization-flow.md
```

- [ ] **Step 3: 创建 skills 和 examples 目录**

```bash
mkdir -p skills
mkdir -p examples
touch skills/.gitkeep
touch examples/.gitkeep
```

- [ ] **Step 4: 验证目录结构**

```bash
ls -la docs/
ls -la prompts/
ls -la skills/
ls -la examples/
```

Expected: 所有目录和文件都已创建

- [ ] **Step 5: 提交**

```bash
git add docs/ prompts/ skills/ examples/
git commit -m "chore: create base directory structure for KDA alignment"
```

---

### Task 1.2: 编写 CUDA 优化工作流文档

**Files:**
- Modify: `docs/agent-flow.md`

- [ ] **Step 1: 编写 agent-flow.md 内容**

```markdown
# CUDA Kernel Optimization Agent Flow

基于 Kernel Design Agents (KDA) 的 CUDA kernel 优化工作流。

## Minimal Loop

1. **定义任务契约 (Task Contract)** - 明确优化目标、约束、验证命令
2. **Agent 检查工作空间** - 读取 baseline kernel、测试、文档
3. **Profile baseline** - 使用 NCU 分析性能
4. **Agent 写 docs/draft.md** - 初始优化计划草稿
5. **转换为可执行计划** - docs/plan.md
6. **实现第一个候选方案** - 应用优化策略
7. **验证正确性** - 运行测试
8. **测量性能** - NCU profiling + benchmark
9. **记录证据** - 更新 candidates.jsonl 和 benchmark.csv
10. **决策** - 保留/修改/拒绝候选方案
11. **重复** - 直到满足晋升标准或明确阻塞因素

## Task Contract（任务契约）

每个优化任务应明确：

- **Objective**: 优化目标（如：提升 matmul 性能）
- **Baseline**: 当前 kernel 实现和性能
- **Target metric**: 目标指标（TFLOPS、带宽利用率、延迟）
- **Target value**: 目标值（如：> 500 GFLOPS 或 speedup > 2x）
- **Correctness requirements**: 正确性要求（如：数值精度 < 1e-5）
- **Constraints**: 约束条件（如：使用 CUDA C++，不使用 cuBLAS）
- **Validation command**: 正确性验证命令
- **Evaluation command**: 性能测试命令
- **Promotion criteria**: 晋升标准（如：正确性通过 + speedup > 1.5x）

## Evidence Records（证据记录）

在任务工作空间中使用简单文件：

- `docs/draft.md` - 初始计划草稿
- `docs/plan.md` - 可执行计划
- `kernels/` - 候选 kernel 实现
- `profile/` - NCU 报告
- `benchmark.csv` - 性能测试结果
- `candidates.jsonl` - 候选方案记录

## Promotion Rule（晋升规则）

只有当候选方案满足任务契约且有证据表明改进或保持目标指标时，才晋升候选方案。
如果拒绝候选方案，记录原因而不是默默丢弃。

## Skills Usage

### ncu-interpreter-skill
用于分析 NCU 报告，识别瓶颈：
- 计算派生指标（带宽利用率、算术强度）
- 识别访问模式（coalesced/strided/random）
- Roofline 模型分析
- 瓶颈诊断（memory/compute/occupancy）

### strategy-library-skill
用于查询优化策略：
- 根据瓶颈类型查询适用策略
- 获取策略详情和代码模板
- 参数选择建议（tile size、block size）

### optimization-history-skill
用于历史分析和推荐：
- 记录每轮优化结果
- 分析趋势（improving/stagnant/degrading）
- 检测瓶颈转移
- 推荐下一步策略

## Workflow Principles

1. **证据驱动** - 每个决策都基于 NCU 报告和 benchmark 数据
2. **小步迭代** - 一次一个优化策略，验证后再继续
3. **完整记录** - 记录所有候选方案，包括失败的
4. **人机协作** - Agent 提出方案，人类做最终决策
```

- [ ] **Step 2: 验证文档格式**

```bash
cat docs/agent-flow.md
```

Expected: 文档内容完整，格式正确

- [ ] **Step 3: 提交**

```bash
git add docs/agent-flow.md
git commit -m "docs: add CUDA optimization agent flow"
```

---

### Task 1.3: 编写 Prompt 模板文档

**Files:**
- Modify: `prompts/README.md`
- Modify: `prompts/kernel-optimization-flow.md`

- [ ] **Step 1: 编写 prompts/README.md**

```markdown
# Prompt Templates

本目录存储 KernelForge-Optimizer 的 prompt 模板。

模板是任务无关的。在开始 Agent 会话前，填入任务目标、约束、验证命令和晋升标准。

## Available Templates

| Path | Purpose |
|---|---|
| `kernel-optimization-flow.md` | CUDA kernel 优化任务的最小化 prompt |

## How To Use

1. 创建或进入任务实现工作空间
2. 将相关模板内容复制到 Agent 会话中
3. 用任务特定的详细信息替换占位符
4. 让 Agent 读取工作空间并写 `docs/draft.md`
5. 将草稿转换为可执行计划
6. 运行实现循环，每次有意义的变更后验证

任务特定的 prompts 应该与它们描述的任务放在一起。不要将特定 benchmark 的数据集、验收表或私有评估器详细信息添加到这个通用仓库。
```

- [ ] **Step 2: 编写 prompts/kernel-optimization-flow.md**

内容见下一步骤（内容较长，分步写入）

- [ ] **Step 3: 写入 kernel-optimization-flow.md 第一部分**

```markdown
# CUDA Kernel Optimization Flow Prompt

你正在一个 CUDA kernel 优化任务工作空间中工作。你的任务是为下述任务产生最佳的正确实现。

## Task Contract

- Task name: `<填写任务名称>`
- Objective: `<填写优化目标，如：优化 matmul kernel 性能>`
- Baseline: `<填写当前实现和性能，如：naive matmul, 100 GFLOPS>`
- Target metric: `<填写目标指标，如：TFLOPS, 带宽利用率>`
- Target value: `<填写目标值，如：> 500 GFLOPS 或 speedup > 2x>`
- Correctness requirements: `<填写正确性要求，如：数值精度 < 1e-5>`
- Constraints: `<填写约束条件，如：使用 CUDA C++，不使用 cuBLAS>`
- Validation command: `<填写验证命令，如：python test_correctness.py>`
- Evaluation command: `<填写性能测试命令，如：python benchmark.py>`
- Promotion criteria: `<填写晋升标准，如：正确性通过 + speedup > 1.5x>`

## Workflow

1. **读取工作空间** - 检查 baseline kernel、测试、文档
2. **Profile baseline** - 使用 ncu-interpreter-skill 分析 NCU 报告
3. **研究相关知识** - 使用 strategy-library-skill 查询优化策略
4. **写优化计划草稿** - 写到 `docs/draft.md`
5. **转换为可执行计划** - 在开始编码前完成
6. **实现候选方案** - 一次一个候选方案
7. **验证正确性** - 每个候选方案后运行验证
8. **测量性能** - NCU profiling + benchmark
9. **记录证据** - 更新 candidates.jsonl、benchmark.csv、profile/
10. **使用 optimization-history-skill** - 分析趋势，推荐下一步
11. **保持变更范围** - 聚焦于任务契约

## Plan Draft Requirements

`docs/draft.md` 应包含：

- **Current baseline** - 当前实现和性能分析
- **NCU diagnosis** - 使用 ncu-interpreter-skill 的诊断结果
- **Bottleneck identification** - 主要瓶颈（memory/compute/occupancy）
- **Candidate strategies** - 候选优化策略（从 strategy-library-skill）
- **Risk assessment** - 主要风险和未知因素
- **Implementation steps** - 具体实现步骤
- **Validation plan** - 验证和评估命令
- **Promotion criteria** - 晋升/修改/拒绝的证据要求

在计划草稿存在之前，不要开始实现。

## Skills Usage

### ncu-interpreter-skill
用于分析 NCU 报告，识别瓶颈：
- 计算派生指标（带宽利用率、算术强度）
- 识别访问模式（coalesced/strided/random）
- Roofline 模型分析
- 瓶颈诊断（memory/compute/occupancy）

### strategy-library-skill
用于查询优化策略：
- 根据瓶颈类型查询适用策略
- 获取策略详情和代码模板
- 参数选择建议（tile size、block size）

### optimization-history-skill
用于历史分析和推荐：
- 记录每轮优化结果
- 分析趋势（improving/stagnant/degrading）
- 检测瓶颈转移
- 推荐下一步策略

## Evidence-Based Decision Making

每个候选方案的决策必须基于证据：

- **保留（Promote）**: 正确性通过 + 性能改进 + 满足晋升标准
- **修改（Revise）**: 正确性通过但性能不足，或有改进空间
- **拒绝（Reject）**: 正确性失败或性能退化，记录原因

记录所有证据到工作空间：
- NCU 报告 → `profile/<candidate_name>.ncu-rep`
- Benchmark 结果 → `benchmark.csv`
- 候选方案元数据 → `candidates.jsonl`
```

- [ ] **Step 4: 验证 prompt 模板**

```bash
cat prompts/README.md
cat prompts/kernel-optimization-flow.md
```

Expected: 两个文件内容完整

- [ ] **Step 5: 提交**

```bash
git add prompts/
git commit -m "docs: add prompt templates for CUDA optimization"
```

---

### Task 1.4: 更新项目 README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 备份现有 README**

```bash
cp README.md README.md.backup
```

- [ ] **Step 2: 编写新的 README.md**

```markdown
# KernelForge-Optimizer

Agent-centric workflow for CUDA kernel optimization, aligned with [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents).

## Overview

KernelForge-Optimizer 提供了一个轻量级的 Agent 工作流框架，用于 CUDA kernel 性能优化。通过 3 个专业 skills 和证据驱动的迭代流程，帮助开发者系统化地优化 GPU kernel 性能。

**核心特性**：
- **Agent 中心化工作流** - 基于 KDA 的成熟方法论
- **3 个专业 Skills** - NCU 解释、策略库、历史分析
- **证据驱动决策** - 完整的性能分析和优化记录
- **工作空间分离** - 通用流程与具体任务分离

## Contents

| Path | Purpose |
|---|---|
| `docs/agent-flow.md` | CUDA 优化工作流说明 |
| `prompts/kernel-optimization-flow.md` | CUDA 优化任务 prompt 模板 |
| `skills/ncu-interpreter-skill/` | NCU 指标解释和瓶颈诊断 |
| `skills/strategy-library-skill/` | 优化策略知识库和代码模板 |
| `skills/optimization-history-skill/` | 历史分析和趋势推荐 |
| `examples/matmul-optimization/` | 完整的优化示例 |

## Getting Started

### 安装 Skills

```bash
# 克隆项目
git clone <repository-url>
cd KernelForge-Optimizer

# 链接 skills 到 Claude Code
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/ncu-interpreter-skill" ~/.claude/skills/ncu-interpreter-skill
ln -s "$(pwd)/skills/strategy-library-skill" ~/.claude/skills/strategy-library-skill
ln -s "$(pwd)/skills/optimization-history-skill" ~/.claude/skills/optimization-history-skill

# 安装 Python 依赖
pip install -r requirements.txt
```

### 快速开始

1. **创建任务工作空间**

```bash
mkdir -p my-optimization-task
cd my-optimization-task
```

2. **定义任务契约**

创建 `README.md`，明确优化目标、baseline、约束条件等。

3. **启动 Agent 会话**

在任务工作空间中启动 Claude Code，使用 `prompts/kernel-optimization-flow.md` 模板。

4. **遵循工作流**

- Profile baseline
- Agent 写 `docs/draft.md`
- 转换为 `docs/plan.md`
- 实现候选方案
- 验证 + 测量 + 记录
- 重复直到满足晋升标准

## Minimal Flow

```
1. 定义任务契约
2. Agent 检查工作空间
3. Profile baseline (NCU)
4. Agent 写 docs/draft.md
5. 转换为 docs/plan.md
6. 实现候选方案
7. 验证正确性
8. 测量性能 (NCU + benchmark)
9. 记录证据 (candidates.jsonl, benchmark.csv, profile/)
10. 决策 (保留/修改/拒绝)
11. 重复
```

## Skills

### ncu-interpreter-skill

将原始 NCU 指标转换为高层次性能诊断：
- 计算派生指标（带宽利用率、算术强度、占用率）
- 识别访问模式（coalesced/strided/random）
- Roofline 模型分析
- 瓶颈诊断（memory bandwidth/latency, compute, occupancy）

### strategy-library-skill

提供经过验证的 CUDA 优化策略：
- 9 个优化策略（matmul tiling、vectorized、tensor core 等）
- 问题模式匹配（memory-bound → 策略推荐）
- 代码模板和参数选择规则
- 性能预期和适用场景

### optimization-history-skill

跟踪优化历史，智能推荐：
- 记录每轮优化结果
- 趋势分析（improving/stagnant/degrading）
- 瓶颈转移检测
- 策略有效性分析

## Recommended Workspace Layout

```
task-workspace/
├── docs/
│   ├── draft.md          # 初始计划草稿
│   └── plan.md           # 可执行计划
├── kernels/              # Kernel 实现
│   ├── baseline.cu
│   ├── candidate_v1.cu
│   └── best.cu
├── profile/              # NCU 报告
├── tests/                # 测试文件
├── benchmark.csv         # 性能测试结果
├── candidates.jsonl      # 候选方案记录
└── README.md             # 任务说明
```

## Examples

查看 `examples/matmul-optimization/` 获取完整的端到端示例。

## Architecture

基于 [Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents) 的工作流：

- **工作空间分离** - 通用流程（本仓库）与具体任务（task workspace）分离
- **证据驱动** - 记录 draft、plan、candidates、benchmark、profile
- **Skills 扩展** - 通过专业 skills 获取领域知识
- **人机协作** - Agent 提出方案，人类做决策

## Contributing

欢迎贡献！改进方向：
- 添加更多优化策略到 strategy-library-skill
- 支持更多 GPU 架构
- 改进 NCU 指标解释
- 添加更多示例

## License

MIT License - see LICENSE file

## Acknowledgments

本项目基于：
- [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents) - MIT HAN Lab
- [CudaForge](https://github.com/CudaForge/CudaForge) - 原始优化框架

## Contact

For questions or issues, please open a GitHub issue.
```

- [ ] **Step 3: 验证 README**

```bash
cat README.md
```

Expected: README 内容完整，格式正确

- [ ] **Step 4: 提交**

```bash
git add README.md
git commit -m "docs: rewrite README to align with KDA"
```

---

### Task 1.5: 更新 CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 备份现有 CLAUDE.md**

```bash
cp CLAUDE.md CLAUDE.md.backup
```

- [ ] **Step 2: 在现有 CLAUDE.md 顶部添加 KDA 工作流说明**

在现有内容之前添加：

```markdown
# KernelForge-Optimizer Agent Instructions

本仓库是 CUDA kernel 优化的 Agent 工作流参考，基于 [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)。

## Repository Rules

- 本仓库保持小型和任务无关
- 任务特定的 prompts、数据集、验证器、生成的实现、benchmark 日志和候选方案应该放在任务工作空间中
- 生成的输出放在任务工作空间的 `kernels/`、`profile/`、`tests/` 中
- 优先记录可重用的工作流机制，而不是记录单个任务的私有工具或验收阈值

## Expected Agent Workflow

对于新的优化任务：

1. 创建或进入独立的任务工作空间
2. 定义任务目标、约束、验证命令和晋升标准
3. 使用 `prompts/kernel-optimization-flow.md` 作为启动 prompt
4. 读取本地任务代码和文档，然后再提出实现变更
5. 将初始计划草稿写到任务工作空间内的 `docs/draft.md`
6. 将草稿转换为可执行计划
7. 小步迭代实现，验证每个有意义的候选方案
8. 记录候选方案关系、评估结果和性能分析证据
9. 保持本仓库专注于可重用的流程

## Skills Usage

任务相关时使用外部 skills：

- `ncu-interpreter-skill` - NCU 报告分析和瓶颈诊断
- `strategy-library-skill` - 优化策略查询和代码模板
- `optimization-history-skill` - 历史分析和趋势推荐

---

# 原有的项目开发规范

（保留原有的 CLAUDE.md 内容）
```

- [ ] **Step 3: 验证 CLAUDE.md**

```bash
head -50 CLAUDE.md
```

Expected: 新内容已添加到顶部

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with KDA workflow instructions"
```

---

## 阶段 1 完成检查点

- [ ] **验证所有文档已创建**

```bash
ls -la docs/agent-flow.md
ls -la prompts/README.md
ls -la prompts/kernel-optimization-flow.md
ls -la README.md
ls -la CLAUDE.md
```

Expected: 所有文件都存在

- [ ] **验证目录结构**

```bash
tree -L 2 -d
```

Expected: docs/, prompts/, skills/, examples/ 目录都存在

- [ ] **验证 Git 历史**

```bash
git log --oneline -10
```

Expected: 看到所有阶段 1 的提交

---

**阶段 1 完成！** 基础架构已搭建完成。

继续阶段 2：ncu-interpreter-skill 实现...

（由于内容过长，计划文档将分多个文件或分批次完成）
