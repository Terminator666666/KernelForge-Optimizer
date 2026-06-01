# KernelForge-Optimizer 改造设计文档

**设计日期**：2026-06-01  
**目标**：将 KernelForge-Optimizer 改造为基于 kernel-design-agents-main 的 Agent 中心化工作流

---

## 一、设计概述

### 1.1 改造目标

**从**：自动化的 CUDA 优化工具（3500 行代码驱动）  
**到**：Agent 中心化的优化工作流（轻量级框架 + Skills）

### 1.2 核心设计原则

1. **完全对齐 KDA**：采用相同的工作流、目录结构、文档风格
2. **轻量级单 Agent + Skills**：一个主 Agent + 3 个专业 skills
3. **工作空间分离**：通用流程（KernelForge-Optimizer repo）与具体任务（task workspace）分离
4. **证据驱动**：记录 draft、plan、candidates、benchmark、profile
5. **混合型 Skills**：文档 + Python 工具，类似 KernelWiki 和 ncu-report-skill

### 1.3 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 架构模式 | 轻量级单 Agent + Skills | 完全对齐 KDA 的成熟设计 |
| Skills 实现 | 混合型（文档 + 工具） | 参考 KernelWiki 和 ncu-report-skill |
| 代码处理 | 转换为 Skills | 保留技术深度，提升灵活性 |
| 工作流 | Agent 协作 + 证据驱动 | 提升可追溯性和人机协作 |

---

## 二、项目结构设计

### 2.1 改造后的目录结构

```
KernelForge-Optimizer/                    # 通用流程仓库（类似 KDA）
├── README.md                             # 项目说明
├── CLAUDE.md                             # Agent 工作规范
├── docs/
│   └── agent-flow.md                     # CUDA 优化工作流说明
├── prompts/
│   ├── README.md                         # Prompt 使用说明
│   └── kernel-optimization-flow.md       # CUDA 优化任务 prompt 模板
├── skills/                               # 三个核心 skills
│   ├── ncu-interpreter-skill/
│   ├── strategy-library-skill/
│   └── optimization-history-skill/
└── examples/                             # 示例任务工作空间
    └── matmul-optimization/
```

### 2.2 与 KDA 的对齐

| KDA 结构 | KernelForge-Optimizer 结构 | 说明 |
|----------|---------------------------|------|
| `docs/agent-flow.md` | `docs/agent-flow.md` | CUDA 优化工作流 |
| `prompts/basic-flow.md` | `prompts/kernel-optimization-flow.md` | CUDA 优化 prompt 模板 |
| `skills/KernelWiki/` | `skills/strategy-library-skill/` | 优化策略知识库 |
| `skills/ncu-report-skill/` | `skills/ncu-interpreter-skill/` | NCU 分析工具 |
| - | `skills/optimization-history-skill/` | 历史分析（新增） |

---

## 三、三个核心 Skills 设计

### 3.1 ncu-interpreter-skill

**目的**：将原始 NCU 指标转换为高层次性能诊断

**结构**（对齐 ncu-report-skill）：
```
skills/ncu-interpreter-skill/
├── SKILL.md                              # Skill 入口点
├── README.md                             # 使用说明
├── reference/                            # 参考文档
│   ├── 00-overview.md                    # NCU 解释器概述
│   ├── 01-derived-metrics.md             # 派生指标计算
│   ├── 02-memory-analysis.md             # 内存子系统分析
│   ├── 03-compute-analysis.md            # 计算子系统分析
│   ├── 04-roofline-model.md              # Roofline 模型分析
│   ├── 05-bottleneck-identification.md   # 瓶颈识别逻辑
│   ├── 06-access-patterns.md             # 访问模式识别
│   └── 07-gpu-specs.md                   # GPU 架构规格
├── helpers/                              # Python 工具
│   ├── analyze_ncu_report.py             # 解析 NCU 报告
│   ├── identify_bottleneck.py            # 瓶颈识别
│   ├── detect_access_pattern.py          # 访问模式检测
│   └── generate_diagnosis.py             # 生成诊断报告
└── data/
    └── gpu_specs.yaml                    # GPU 规格数据库
```

**代码量**：~300 行 Python + ~800 行 Markdown

**从现有代码迁移**：
- `agents/ncu_interpreter.py` (600 行) → helpers/ (~300 行) + reference/ (~800 行 MD)
- `utils/gpu_arch_detection.py` (400 行) → data/gpu_specs.yaml + helpers/detect_gpu.py

---

### 3.2 strategy-library-skill

**目的**：提供经过验证的 CUDA 优化策略和代码模板

**结构**（对齐 KernelWiki）：
```
skills/strategy-library-skill/
├── SKILL.md                              # Skill 入口点
├── README.md                             # 使用说明
├── index.md                              # 策略索引
├── wiki/                                 # 策略知识库
│   ├── strategies/                       # 9 个优化策略
│   │   ├── matmul-tiling.md              # 矩阵乘法分块
│   │   ├── reduction-warp-primitives.md  # Warp 级归约
│   │   ├── vectorized-memory.md          # 向量化访问
│   │   ├── kernel-fusion.md              # 算子融合
│   │   ├── tensor-core.md                # Tensor Core 优化
│   │   ├── bank-conflict-free.md         # 避免 Bank Conflict
│   │   ├── occupancy-tuning.md           # 占用率优化
│   │   ├── cooperative-groups.md         # 协作组
│   │   └── persistent-threads.md         # 持久线程
│   └── patterns/                         # 问题模式
│       ├── memory-bandwidth-bound.md     # 内存带宽瓶颈
│       ├── memory-latency-bound.md       # 内存延迟瓶颈
│       ├── compute-bound.md              # 计算瓶颈
│       └── low-occupancy.md              # 低占用率
├── scripts/                              # 查询工具
│   ├── query_strategy.py                 # 策略查询
│   ├── get_strategy.py                   # 获取策略详情
│   └── match_strategy.py                 # 策略匹配
└── data/
    ├── strategy_metadata.yaml            # 策略元数据
    └── applicability_rules.yaml          # 适用性规则
```

**代码量**：~200 行 Python + ~1000 行 Markdown

**从现有代码迁移**：
- `agents/strategy_templates.py` (900 行) → scripts/ (~200 行) + wiki/ (~1000 行 MD)
- `utils/operator_detection.py` (250 行) → scripts/ 或集成到匹配逻辑

**每个策略文档包含**：
- 原理说明
- 适用场景
- 代码模板（完整可编译）
- 参数选择规则
- 性能预期
- 相关 PR/论文引用

---

### 3.3 optimization-history-skill

**目的**：跟踪优化历史，分析趋势，推荐下一步策略

**结构**（轻量级工具型）：
```
skills/optimization-history-skill/
├── SKILL.md                              # Skill 入口点
├── README.md                             # 使用说明
├── reference/                            # 参考文档
│   ├── 00-overview.md                    # 历史管理概述
│   ├── 01-trend-analysis.md              # 趋势分析方法
│   ├── 02-bottleneck-shift.md            # 瓶颈转移检测
│   ├── 03-strategy-effectiveness.md      # 策略有效性分析
│   └── 04-recommendation-logic.md        # 推荐逻辑
├── helpers/                              # Python 工具
│   ├── record_round.py                   # 记录优化轮次
│   ├── analyze_trend.py                  # 趋势分析
│   ├── detect_shift.py                   # 瓶颈转移检测
│   ├── recommend_strategy.py             # 策略推荐
│   └── generate_report.py                # 生成历史报告
└── data/
    └── history_schema.yaml               # 历史记录格式
```

**代码量**：~400 行 Python + ~200 行 Markdown

**从现有代码迁移**：
- `agents/optimization_history.py` (400 行) → helpers/ (~400 行) + reference/ (~200 行 MD)

---

## 四、工作流设计

### 4.1 CUDA 优化工作流（`docs/agent-flow.md`）

**基于 KDA 的 Minimal Loop，针对 CUDA 优化定制**：

```
1. 定义任务契约 (Task Contract)
   ↓
2. Agent 检查工作空间
   ↓
3. Profile baseline (使用 NCU)
   ↓
4. Agent 写 docs/draft.md
   ↓
5. 转换为可执行计划 docs/plan.md
   ↓
6. 实现候选方案
   ↓
7. 验证正确性
   ↓
8. 测量性能 (NCU + benchmark)
   ↓
9. 记录证据 (candidates.jsonl, benchmark.csv, profile/)
   ↓
10. 决策 (保留/修改/拒绝)
   ↓
11. 重复直到满足晋升标准
```

### 4.2 任务契约 (Task Contract)

每个优化任务应明确：
- **Objective**: 优化目标
- **Baseline**: 当前实现和性能
- **Target metric**: 目标指标
- **Target value**: 目标值
- **Correctness requirements**: 正确性要求
- **Constraints**: 约束条件
- **Validation command**: 验证命令
- **Evaluation command**: 性能测试命令
- **Promotion criteria**: 晋升标准

### 4.3 证据记录 (Evidence Records)

在任务工作空间中使用简单文件：
- `docs/draft.md` - 初始计划草稿
- `docs/plan.md` - 可执行计划
- `kernels/` - 候选 kernel 实现
- `profile/` - NCU 报告
- `benchmark.csv` - 性能测试结果
- `candidates.jsonl` - 候选方案记录

---

## 五、任务工作空间设计

### 5.1 推荐的工作空间布局

```
task-workspace/                           # 独立的任务工作空间
├── docs/
│   ├── draft.md                          # 初始优化计划草稿
│   └── plan.md                           # 可执行优化计划
├── kernels/                              # Kernel 实现
│   ├── baseline.cu                       # Baseline kernel
│   ├── candidate_v1.cu                   # 候选方案 v1
│   ├── candidate_v2.cu                   # 候选方案 v2
│   └── best.cu                           # 最佳候选方案
├── profile/                              # NCU 报告
│   ├── baseline.ncu-rep                  # Baseline NCU 报告
│   ├── candidate_v1.ncu-rep              # 候选方案 v1 NCU 报告
│   └── candidate_v2.ncu-rep              # 候选方案 v2 NCU 报告
├── tests/                                # 测试文件
│   ├── test_correctness.py               # 正确性测试
│   └── benchmark.py                      # 性能测试
├── benchmark.csv                         # 性能测试结果表格
├── candidates.jsonl                      # 候选方案记录
└── README.md                             # 任务说明
```

### 5.2 文件格式

#### `candidates.jsonl` 格式
```jsonl
{"id": "baseline", "parent": null, "status": "baseline", "correctness": "pass", "performance": {"GFLOPS": 100}, "timestamp": "2026-06-01T10:00:00"}
{"id": "candidate_v1", "parent": "baseline", "status": "promoted", "strategy": "matmul-tiling", "correctness": "pass", "performance": {"GFLOPS": 250}, "speedup": 2.5, "timestamp": "2026-06-01T11:00:00"}
{"id": "candidate_v2", "parent": "candidate_v1", "status": "rejected", "strategy": "tensor-core", "correctness": "fail", "reason": "numerical precision error", "timestamp": "2026-06-01T12:00:00"}
```

#### `benchmark.csv` 格式
```csv
candidate,timestamp,GFLOPS,bandwidth_util,occupancy,speedup,status
baseline,2026-06-01T10:00:00,100,45%,50%,1.0x,baseline
candidate_v1,2026-06-01T11:00:00,250,78%,75%,2.5x,promoted
candidate_v2,2026-06-01T12:00:00,N/A,N/A,N/A,N/A,rejected
```

---

## 六、代码迁移策略

### 6.1 现有代码到 Skills 的映射

#### 从 `agents/ncu_interpreter.py` (600 行) → `ncu-interpreter-skill`

```
agents/ncu_interpreter.py
├── NCUInterpreter 类
│   ├── _compute_derived_metrics()      → helpers/analyze_ncu_report.py
│   ├── _analyze_memory()               → helpers/analyze_ncu_report.py
│   ├── _analyze_compute()              → helpers/analyze_ncu_report.py
│   ├── _analyze_roofline()             → helpers/analyze_ncu_report.py
│   ├── _identify_bottleneck()          → helpers/identify_bottleneck.py
│   └── _generate_issues()              → helpers/generate_diagnosis.py
└── PerformanceDiagnosis 数据结构       → data/diagnosis_schema.yaml

逻辑提炼到文档：
├── 带宽利用率计算公式                   → reference/01-derived-metrics.md
├── 内存子系统分析规则                   → reference/02-memory-analysis.md
├── 计算子系统分析规则                   → reference/03-compute-analysis.md
├── Roofline 模型原理                   → reference/04-roofline-model.md
├── 瓶颈识别逻辑                        → reference/05-bottleneck-identification.md
└── GPU 架构规格                        → reference/07-gpu-specs.md
```

**保留的代码**：核心计算逻辑（~300 行）  
**转为文档**：原理说明、规则描述（~300 行 → Markdown）

#### 从 `agents/strategy_templates.py` (900 行) → `strategy-library-skill`

```
agents/strategy_templates.py
├── StrategyTemplate 数据结构           → data/strategy_metadata.yaml
├── StrategyLibrary 类
│   ├── get_applicable_strategies()     → scripts/match_strategy.py
│   ├── select_parameters()             → scripts/match_strategy.py
│   └── instantiate_template()          → (移除，Agent 直接使用文档)
└── 9 个策略的代码模板和规则             → wiki/strategies/*.md
```

**保留的代码**：策略匹配逻辑（~200 行）  
**转为文档**：策略详情、代码模板（~700 行 → Markdown）

#### 从 `agents/optimization_history.py` (400 行) → `optimization-history-skill`

```
agents/optimization_history.py
├── OptimizationRound 数据结构          → data/history_schema.yaml
├── OptimizationHistory 类
│   ├── add_round()                     → helpers/record_round.py
│   ├── get_recent_trend()              → helpers/analyze_trend.py
│   ├── detect_bottleneck_shift()       → helpers/detect_shift.py
│   ├── analyze_strategy_effectiveness()→ helpers/recommend_strategy.py
│   ├── recommend_next_strategy()       → helpers/recommend_strategy.py
│   └── save()/load()                   → helpers/record_round.py
```

**保留的代码**：全部核心逻辑（~400 行）  
**转为文档**：方法说明（~100 行 → Markdown）

### 6.2 辅助工具的处理

| 现有文件 | 处理方式 |
|---------|---------|
| `utils/operator_detection.py` (250 行) | 集成到 strategy-library-skill/scripts/ |
| `utils/gpu_arch_detection.py` (400 行) | 提取到 ncu-interpreter-skill/data/gpu_specs.yaml |
| `prompts/enhanced_judge.py` | 删除（不再需要） |
| `prompts/enhanced_optimization.py` | 删除（不再需要） |
| `main_enhanced.py` (450 行) | 删除或移到 examples/ |
| `main_real_gpu.py` | 移到 examples/ 作为参考 |

### 6.3 迁移后的代码量估算

**原始代码**：~3500 行 Python

**迁移后**：
- **Skills 中的 Python 代码**：~900 行
  - ncu-interpreter-skill: ~300 行
  - strategy-library-skill: ~200 行
  - optimization-history-skill: ~400 行
- **Skills 中的 Markdown 文档**：~2000 行
  - ncu-interpreter-skill: ~800 行
  - strategy-library-skill: ~1000 行
  - optimization-history-skill: ~200 行
- **通用流程文档**：~500 行
  - docs/agent-flow.md
  - prompts/kernel-optimization-flow.md
  - README.md, CLAUDE.md

**总计**：~900 行 Python + ~2500 行 Markdown = ~3400 行

**代码减少**：~100 行（主要是删除了自动化主程序和 prompt 生成器）  
**结构优化**：代码更模块化，文档更清晰，完全对齐 KDA

---

## 七、实施计划

### 7.1 阶段划分

#### **阶段 1：基础架构搭建**（2-3 天）

**任务**：
1. 重组项目目录结构
2. 编写核心文档（README.md, CLAUDE.md, agent-flow.md, prompt 模板）
3. 创建示例任务工作空间

#### **阶段 2：ncu-interpreter-skill 实现**（3-4 天）

**任务**：
1. 创建 skill 目录结构
2. 编写 7 个参考文档
3. 实现 4 个 Python 工具
4. 准备 gpu_specs.yaml

#### **阶段 3：strategy-library-skill 实现**（4-5 天）

**任务**：
1. 创建 skill 目录结构
2. 编写 9 个策略文档 + 4 个问题模式文档
3. 实现 3 个查询工具
4. 准备元数据和规则文件

#### **阶段 4：optimization-history-skill 实现**（2-3 天）

**任务**：
1. 创建 skill 目录结构
2. 编写 4 个参考文档
3. 实现 5 个 Python 工具
4. 准备 history_schema.yaml

#### **阶段 5：清理和文档完善**（2-3 天）

**任务**：
1. 删除不再需要的文件
2. 更新 README.md
3. 编写完整示例
4. 编写测试

#### **阶段 6：验证和优化**（2-3 天）

**任务**：
1. 端到端测试
2. 性能对比
3. 文档优化

### 7.2 总体时间估算

**总计**：15-21 天（约 3-4 周）

**关键路径**：
1. 阶段 1（基础架构）→ 必须先完成
2. 阶段 2-4（三个 skills）→ 可以并行或顺序进行
3. 阶段 5-6（清理和验证）→ 最后完成

---

## 八、预期效果

### 8.1 技术层面

- ✅ 完全对齐 kernel-design-agents-main 的成熟架构
- ✅ 保留现有的技术深度和专业知识
- ✅ 提升灵活性和可扩展性
- ✅ 减少自动化代码，增强文档和指导

### 8.2 使用层面

- ✅ 更清晰的工作流程
- ✅ 更好的证据记录和可追溯性
- ✅ 更强的人机协作能力
- ✅ 易于添加新的 skills 和策略

### 8.3 面试展示层面

- ✅ 展示对成熟开源项目的学习和对齐能力
- ✅ 展示 Agent 工作流的理解
- ✅ 展示多智能体框架的设计思路
- ✅ 保留原有的技术深度和代码质量

---

## 九、风险和应对

### 9.1 潜在风险

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| 文档编写工作量大 | 延期 | 优先完成核心文档，逐步完善 |
| Skills 调用复杂度 | 使用困难 | 提供详细示例和使用指南 |
| 迁移过程中功能缺失 | 功能不完整 | 保留旧代码作为参考，逐步迁移 |
| Agent 理解能力不足 | 效果不佳 | 优化文档结构，提供更多示例 |

### 9.2 质量保证

- ✅ 每个 skill 包含完整的 README 和使用示例
- ✅ 每个策略文档包含可编译的代码模板
- ✅ 提供端到端的示例任务
- ✅ 编写单元测试和集成测试

---

## 十、总结

本设计文档详细描述了将 KernelForge-Optimizer 改造为基于 kernel-design-agents-main 的 Agent 中心化工作流的完整方案。

**核心改造**：
- 从自动化工具 → Agent 协作工作流
- 从代码驱动 → 文档 + 工具混合
- 从单体架构 → Skills 模块化架构
- 从黑盒优化 → 证据驱动决策

**保留优势**：
- NCU 解释能力
- 优化策略知识库
- 历史分析能力
- GPU 架构支持

**新增能力**：
- 完整的工作流指导
- 证据记录和可追溯性
- 人机协作决策
- 灵活的扩展性

**实施计划**：15-21 天，分 6 个阶段完成

---

**文档版本**：v1.0  
**最后更新**：2026-06-01  
**作者**：Claude (Opus 4.7)
