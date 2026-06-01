# KernelForge-Optimizer Agent Instructions

本仓库是 CUDA kernel 优化的 Agent 工作流参考，基于 [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)。

---

## 🚧 当前实施进度（2026-06-01）

### 总体进度：阶段 1-2 完成，阶段 3 进行中（10%）

**设计文档**：`docs/superpowers/specs/2026-06-01-kda-alignment-design.md`  
**实施计划**：`docs/superpowers/plans/2026-06-01-kda-alignment-implementation.md`

### ✅ 阶段 1：基础架构搭建 - 100% 完成

**完成时间**：2026-06-01  
**Git Commits**：5 个

| 任务 | 状态 | Commit | 说明 |
|------|------|--------|------|
| Task 1.1: 创建目录结构 | ✅ | `9a58a6e` | docs/, prompts/, skills/, examples/ |
| Task 1.2: 编写工作流文档 | ✅ | `b34eff5` | docs/agent-flow.md (76 行) |
| Task 1.3: 编写 Prompt 模板 | ✅ | `dd4ff57` | prompts/README.md + kernel-optimization-flow.md (102 行) |
| Task 1.4: 更新 README | ✅ | `d1bee6a` | 重写为 KDA 风格 |
| Task 1.5: 更新 CLAUDE.md | ✅ | `ba0fcd8` | 添加 KDA 工作流说明 |

**成果**：
- ✅ 基础目录结构已创建
- ✅ CUDA 优化工作流文档已编写
- ✅ Prompt 模板已编写
- ✅ README.md 已重写（对齐 KDA）
- ✅ CLAUDE.md 已更新

### ✅ 阶段 2：ncu-interpreter-skill 实现 - 100% 完成

**完成时间**：2026-06-01  
**总代码量**：4141 行

| 任务 | 状态 | 说明 |
|------|------|------|
| Task 2.1: 创建目录结构 | ✅ | 15 个文件（目录和占位符） |
| Task 2.2: 编写 SKILL.md 和 README.md | ✅ | 238 行文档 |
| Task 2.3: 编写参考文档 | ✅ | 8 个文档，共 3054 行 |
| Task 2.4: 实现 Python 工具 | ✅ | 4 个脚本，共 852 行 |
| Task 2.5: 准备 GPU 规格数据 | ✅ | gpu_specs.yaml，235 行 |

**已完成文件清单**：

**参考文档**（8 个，3054 行）：
- ✅ reference/00-overview.md（196 行）- NCU 解释器概述
- ✅ reference/01-derived-metrics.md（307 行）- 派生指标计算方法
- ✅ reference/02-memory-analysis.md（380 行）- 内存子系统分析
- ✅ reference/03-compute-analysis.md（382 行）- 计算子系统分析
- ✅ reference/04-roofline-model.md（385 行）- Roofline 模型
- ✅ reference/05-bottleneck-identification.md（452 行）- 瓶颈识别逻辑
- ✅ reference/06-access-patterns.md（454 行）- 访问模式识别
- ✅ reference/07-gpu-specs.md（498 行）- GPU 架构规格

**Python 工具**（4 个，852 行）：
- ✅ helpers/analyze_ncu_report.py（185 行）- NCU 报告解析
- ✅ helpers/identify_bottleneck.py（184 行）- 瓶颈识别
- ✅ helpers/detect_access_pattern.py（200 行）- 访问模式检测
- ✅ helpers/generate_diagnosis.py（283 行）- 生成诊断报告

**数据文件**（1 个，235 行）：
- ✅ data/gpu_specs.yaml（235 行）- GPU 规格数据库（支持 RTX 50/40/30、H100、A100、V100、T4）

**核心功能**：
- ✅ 解析 NCU CSV/JSON 报告
- ✅ 计算 6 个派生指标（带宽利用率、算术强度、占用率等）
- ✅ 内存子系统分析（缓存命中率、访问模式、效率）
- ✅ 计算子系统分析（占用率、限制因素、SM 效率）
- ✅ Roofline 模型分析（memory-bound vs compute-bound）
- ✅ 瓶颈识别（4 种类型，带置信度）
- ✅ 访问模式检测（coalesced/strided/mixed/random）
- ✅ 生成诊断报告和优化建议

### ⏳ 阶段 3：strategy-library-skill 实现 - 10% 完成

**开始时间**：2026-06-01  
**当前代码量**：398 行

| 任务 | 状态 | 说明 |
|------|------|------|
| Task 3.1: 创建目录结构 | ✅ | 15 个文件（目录和占位符） |
| Task 3.2: 编写 SKILL.md 和 README.md | ✅ | 已完成（从总结中得知） |
| Task 3.3: 编写策略规则 | ✅ | strategy_rules.yaml，398 行 |
| Task 3.4: 编写参考文档 | ⏳ 0% | 6 个文档，待完成 |
| Task 3.5: 编写 CUDA 模板 | ⏳ 0% | 5 个模板，待完成 |

**已完成**：
- ✅ strategy-library-skill 目录结构（15 个文件）
- ✅ SKILL.md - 策略库使用说明
- ✅ README.md - 概述、安装、使用示例
- ✅ data/strategy_rules.yaml（398 行）- 9 个优化策略的完整定义

**策略规则包含**：
- ✅ matmul_tiling - 矩阵乘法共享内存分块（2-5× 加速）
- ✅ reduction_warp_primitives - Warp 级归约（3-8× 加速）
- ✅ vectorized_memory - 向量化内存访问（1.5-3× 加速）
- ✅ kernel_fusion - 算子融合（2-4× 加速）
- ✅ tensor_core - Tensor Core 优化（2-10× 加速）
- ✅ bank_conflict_free - 避免 Bank Conflict（1.5-3× 加速）
- ✅ occupancy_tuning - 占用率优化（1.5-4× 加速）
- ✅ cooperative_groups - 协作组优化（1.2-2× 加速）
- ✅ persistent_threads - 持久线程优化（2-5× 加速）

**待完成**：

**参考文档**（6 个，预计 1500-2000 行）：
- ⏳ reference/00-overview.md - 策略库概述
- ⏳ reference/01-matmul-tiling.md - 矩阵乘法分块详解
- ⏳ reference/02-reduction-warp.md - Warp 级归约详解
- ⏳ reference/03-vectorized-memory.md - 向量化访问详解
- ⏳ reference/04-kernel-fusion.md - 算子融合详解
- ⏳ reference/05-tensor-core.md - Tensor Core 详解

**CUDA 代码模板**（5 个，预计 1000-1500 行）：
- ⏳ templates/matmul_tiling.cu - 矩阵乘法分块模板
- ⏳ templates/reduction_warp.cu - Warp 归约模板
- ⏳ templates/vectorized_memory.cu - 向量化访问模板
- ⏳ templates/kernel_fusion.cu - 算子融合模板
- ⏳ templates/tensor_core.cu - Tensor Core 模板

**预计工作量**：
- 参考文档：6 个文档，约 1500-2000 行
- CUDA 模板：5 个模板，约 1000-1500 行
- 总计：约 2500-3500 行代码

### 📋 后续阶段（待开始）

- ⏳ **阶段 4**：optimization-history-skill 实现（2-3 天）
- ⏳ **阶段 5**：清理和文档完善（2-3 天）
- ⏳ **阶段 6**：验证和优化（2-3 天）

### 🎯 下一步行动

**继续阶段 3**：
1. 完成 Task 3.4：编写 6 个参考文档（详细说明每个优化策略）
2. 完成 Task 3.5：编写 5 个 CUDA 代码模板（可直接使用的完整代码）

**参考资料**：
- 策略定义：`skills/strategy-library-skill/data/strategy_rules.yaml`
- 原始实现：`agents/strategy_templates.py`（约 900 行）

### 📊 统计数据

**总体进度**：
- ✅ 阶段 1：100%（5 个文件，~500 行）
- ✅ 阶段 2：100%（13 个文件，4141 行）
- ⏳ 阶段 3：10%（3 个文件，398 行）
- **总计**：21 个文件，~5039 行

**预计最终规模**：
- 阶段 1-3：约 8000-9000 行
- 阶段 4-6：约 2000-3000 行
- **项目总计**：约 10000-12000 行

---

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

# 项目开发规范

以下是本项目的开发规范，继承自父目录的 CLAUDE.md。

## 🌐 语言规范

**所有回复必须使用中文**
- 代码注释使用中文
- 文档说明使用中文
- 技术解释使用中文
- 变量命名可以使用英文，但需要添加中文注释

---

## 🚫 严格禁止事项

**绝对不允许的行为**：

1. **❌ 禁止简化任何逻辑**
   - 不要创建简化版本的代码
   - 不要删除任何功能
   - 不要用占位符代替实际实现
   - 必须保持完整的功能和复杂度

2. **❌ 禁止使用模拟数据**
   - 不要创建 mock 数据
   - 不要创建 fake 数据
   - 不要创建测试数据
   - 除非用户明确要求创建模拟/测试数据

3. **❌ 禁止创建总结文档**
   - 不要自动创建 SUMMARY.md
   - 不要自动创建 FIX_SUMMARY.md
   - 不要自动创建 REPORT.md
   - 不要自动创建任何总结性文档
   - **除非用户明确要求创建**

4. **❌ 禁止过度简化**
   - 不要说"这个太复杂，我们简化一下"
   - 不要说"我们先做个简单版本"
   - 不要说"我们用模拟数据测试"
   - 用户要求完整实现就必须完整实现

**正确做法**：
- ✅ 完整实现所有功能
- ✅ 使用真实数据和真实逻辑
- ✅ 只在对话中口头总结，不创建文件
- ✅ 询问用户是否需要文档，得到确认后再创建

---

## 💻 代码执行规范

**重要：你不需要运行代码**

你的职责是：
- ✅ **分析代码**：理解代码逻辑、发现问题、提出优化建议
- ✅ **生成代码**：根据需求编写新的代码
- ✅ **修改代码**：优化现有代码、修复 bug、重构代码
- ❌ **不要运行代码**：不要尝试执行任何代码
- ❌ **不要测试代码**：不要尝试运行测试

**代码交付标准**：
- 提供完整的代码实现
- 添加详细的中文注释
- 说明代码的使用方法
- 指出可能的问题和注意事项
- 由用户自行在 Linux 环境中运行和测试

---

## ✅ 完成验证规范

**核心原则：每次宣称成功必须附证据**

### 必须提供的证据类型

1. **代码生成/修改**
   - ✅ 提供文件路径：`已创建文件：D:\Agent\xxx.py`
   - ✅ 提供代码片段：展示关键代码实现
   - ✅ 提供行号范围：`修改了第 10-25 行`

2. **文档创建/更新**
   - ✅ 提供文件路径：`已更新文档：D:\Agent\README.md`
   - ✅ 提供更新内容摘要：列出新增/修改的章节

3. **分析结果**
   - ✅ 提供分析报告：详细的分析结论
   - ✅ 提供数据支撑：代码行数、文件数量、依赖关系等
   - ✅ 提供问题清单：发现的问题及建议

4. **方案设计**
   - ✅ 提供设计文档：架构图、流程图、技术选型
   - ✅ 提供实现步骤：详细的执行计划
   - ✅ 提供风险评估：可能的问题和应对方案

### 总结报告规范

**核心原则：创建总结报告前必须询问用户**

**禁止行为**：
- ❌ 不要自动创建总结报告
- ❌ 不要自动创建分析报告
- ❌ 不要自动创建项目文档
- ❌ 不要自动创建任何 Markdown 文件（除非用户明确要求）

**正确做法**：
1. **完成工作后，先口头总结**
   - 在对话中简要说明完成了什么
   - 列出关键成果和证据
   - 不创建文件

2. **询问用户是否需要报告**
   - ❓ "是否需要我创建一份详细的总结报告？"
   - ❓ "是否需要我将分析结果保存为文档？"
   - ❓ "是否需要我生成项目文档？"

3. **用户确认后再创建**
   - ✅ 用户明确说"需要"、"创建"、"生成报告"
   - ✅ 然后再创建相应的文件

---

## 🖥️ 系统环境说明

### 开发环境（Windows）
- **操作系统**：Windows 11 Home China 10.0.26200
- **用途**：代码编写、文档编辑、项目管理
- **限制**：
  - ❌ **不要在 Windows 上安装任何依赖包**
  - ❌ **不要在 Windows 上运行任何代码**
  - ❌ **不要在 Windows 上执行任何构建命令**
  - ✅ 只用于代码编写和文档管理

### 运行环境（Linux）
- **操作系统**：Linux（具体发行版待确认）
- **用途**：代码运行、测试、部署
- **说明**：
  - ✅ 所有代码将在 Linux 环境中运行
  - ✅ 所有依赖包将在 Linux 环境中安装
  - ✅ 所有测试将在 Linux 环境中执行

---

## 🔧 代码编写规范

### 代码风格

1. **Python 代码**
   - 遵循 PEP 8 规范
   - 使用 4 个空格缩进
   - 函数和类添加中文文档字符串
   - 关键逻辑添加中文注释

2. **CUDA 代码**
   - 遵循 CUDA 编程最佳实践
   - 添加中文注释说明关键优化点
   - 标注性能关键路径

---

**最后更新**：2026-06-01  
**维护者**：用户 + Claude  
**版本**：v2.0 (KDA 对齐版本)
