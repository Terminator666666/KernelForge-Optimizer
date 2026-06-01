
## 阶段 2：ncu-interpreter-skill 实现（3-4 天）

### Task 2.1: 创建 ncu-interpreter-skill 目录结构

**Files:**
- Create: `skills/ncu-interpreter-skill/SKILL.md`
- Create: `skills/ncu-interpreter-skill/README.md`
- Create: `skills/ncu-interpreter-skill/reference/` (7 个文档)
- Create: `skills/ncu-interpreter-skill/helpers/` (4 个 Python 脚本)
- Create: `skills/ncu-interpreter-skill/data/gpu_specs.yaml`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p skills/ncu-interpreter-skill/{reference,helpers,data}
touch skills/ncu-interpreter-skill/SKILL.md
touch skills/ncu-interpreter-skill/README.md
```

- [ ] **Step 2: 创建占位文件**

```bash
cd skills/ncu-interpreter-skill/reference
touch {00-overview,01-derived-metrics,02-memory-analysis,03-compute-analysis,04-roofline-model,05-bottleneck-identification,06-access-patterns,07-gpu-specs}.md

cd ../helpers
touch analyze_ncu_report.py identify_bottleneck.py detect_access_pattern.py generate_diagnosis.py

cd ../data
touch gpu_specs.yaml
```

- [ ] **Step 3: 提交**

```bash
git add skills/ncu-interpreter-skill/
git commit -m "feat: create ncu-interpreter-skill directory structure"
```

---

### Task 2.2: 编写 SKILL.md 和 README.md

**Files:**
- Modify: `skills/ncu-interpreter-skill/SKILL.md`
- Modify: `skills/ncu-interpreter-skill/README.md`

**SKILL.md 内容要点**：
- Skill 名称和描述
- 何时使用此 skill
- 允许的工具
- 使用示例

**README.md 内容要点**：
- Skill 概述
- 安装说明
- 使用指南
- 文件结构说明

- [ ] **Step 1: 编写 SKILL.md**（参考 ncu-report-skill 的格式）
- [ ] **Step 2: 编写 README.md**
- [ ] **Step 3: 提交**

```bash
git add skills/ncu-interpreter-skill/{SKILL.md,README.md}
git commit -m "docs(ncu-interpreter): add SKILL.md and README.md"
```

---

### Task 2.3: 编写参考文档（7 个文档）

**Files:**
- Modify: `skills/ncu-interpreter-skill/reference/*.md`

**从现有代码提炼**：`agents/ncu_interpreter.py` → reference 文档

**文档列表**：
1. `00-overview.md` - NCU 解释器概述
2. `01-derived-metrics.md` - 派生指标计算（带宽利用率、算术强度）
3. `02-memory-analysis.md` - 内存子系统分析规则
4. `03-compute-analysis.md` - 计算子系统分析规则
5. `04-roofline-model.md` - Roofline 模型原理和实现
6. `05-bottleneck-identification.md` - 瓶颈识别逻辑
7. `06-access-patterns.md` - 访问模式识别（coalesced/strided/random）
8. `07-gpu-specs.md` - GPU 架构规格说明

- [ ] **Step 1-7: 逐个编写参考文档**（每个文档一个 commit）
- [ ] **Step 8: 验证所有文档**

```bash
ls -la skills/ncu-interpreter-skill/reference/
```

---

### Task 2.4: 实现 Python 工具（4 个脚本）

**Files:**
- Modify: `skills/ncu-interpreter-skill/helpers/*.py`

**从现有代码迁移**：`agents/ncu_interpreter.py` → helpers/

**脚本列表**：
1. `analyze_ncu_report.py` - 解析 NCU 报告，计算派生指标
2. `identify_bottleneck.py` - 瓶颈识别
3. `detect_access_pattern.py` - 访问模式检测
4. `generate_diagnosis.py` - 生成结构化诊断报告

- [ ] **Step 1: 实现 analyze_ncu_report.py**（从 NCUInterpreter 类提取核心逻辑）
- [ ] **Step 2: 实现 identify_bottleneck.py**
- [ ] **Step 3: 实现 detect_access_pattern.py**
- [ ] **Step 4: 实现 generate_diagnosis.py**
- [ ] **Step 5: 添加单元测试**
- [ ] **Step 6: 提交**

```bash
git add skills/ncu-interpreter-skill/helpers/
git commit -m "feat(ncu-interpreter): implement Python analysis tools"
```

---

### Task 2.5: 准备 GPU 规格数据

**Files:**
- Modify: `skills/ncu-interpreter-skill/data/gpu_specs.yaml`

**从现有代码提取**：`utils/gpu_arch_detection.py` → `gpu_specs.yaml`

**YAML 格式示例**：
```yaml
gpus:
  rtx_4090:
    name: "NVIDIA GeForce RTX 4090"
    architecture: "Ada Lovelace"
    compute_capability: "8.9"
    sm_count: 128
    memory_bandwidth_gbps: 1008
    peak_fp32_tflops: 82.6
    ...
```

- [ ] **Step 1: 编写 gpu_specs.yaml**
- [ ] **Step 2: 验证 YAML 格式**
- [ ] **Step 3: 提交**

```bash
git add skills/ncu-interpreter-skill/data/gpu_specs.yaml
git commit -m "data(ncu-interpreter): add GPU specifications database"
```

---

## 阶段 2 完成检查点

- [ ] **验证 skill 结构完整**

```bash
tree skills/ncu-interpreter-skill/
```

Expected: 所有目录和文件都存在

- [ ] **验证文档质量**

```bash
wc -l skills/ncu-interpreter-skill/reference/*.md
```

Expected: 每个文档至少 50-100 行

- [ ] **验证 Python 代码**

```bash
python -m py_compile skills/ncu-interpreter-skill/helpers/*.py
```

Expected: 无语法错误

---

## 阶段 3：strategy-library-skill 实现（4-5 天）

### Task 3.1: 创建 strategy-library-skill 目录结构

**Files:**
- Create: `skills/strategy-library-skill/SKILL.md`
- Create: `skills/strategy-library-skill/README.md`
- Create: `skills/strategy-library-skill/index.md`
- Create: `skills/strategy-library-skill/wiki/strategies/` (9 个策略文档)
- Create: `skills/strategy-library-skill/wiki/patterns/` (4 个问题模式文档)
- Create: `skills/strategy-library-skill/scripts/` (3 个查询脚本)
- Create: `skills/strategy-library-skill/data/`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p skills/strategy-library-skill/{wiki/{strategies,patterns},scripts,data}
touch skills/strategy-library-skill/{SKILL.md,README.md,index.md}
```

- [ ] **Step 2: 提交**

```bash
git add skills/strategy-library-skill/
git commit -m "feat: create strategy-library-skill directory structure"
```

---

### Task 3.2: 编写策略文档（9 个策略）

**Files:**
- Create: `skills/strategy-library-skill/wiki/strategies/*.md`

**从现有代码提炼**：`agents/strategy_templates.py` → wiki/strategies/

**策略列表**：
1. `matmul-tiling.md` - 矩阵乘法分块优化
2. `reduction-warp-primitives.md` - Warp 级归约
3. `vectorized-memory.md` - 向量化内存访问
4. `kernel-fusion.md` - 算子融合
5. `tensor-core.md` - Tensor Core 优化
6. `bank-conflict-free.md` - 避免 Bank Conflict
7. `occupancy-tuning.md` - 占用率优化
8. `cooperative-groups.md` - 协作组
9. `persistent-threads.md` - 持久线程

**每个策略文档包含**：
- 原理说明
- 适用场景
- 代码模板（完整可编译）
- 参数选择规则
- 性能预期
- 相关 PR/论文引用

- [ ] **Step 1-9: 逐个编写策略文档**（每个策略一个 commit）
- [ ] **Step 10: 验证所有策略文档**

---

### Task 3.3: 编写问题模式文档（4 个模式）

**Files:**
- Create: `skills/strategy-library-skill/wiki/patterns/*.md`

**模式列表**：
1. `memory-bandwidth-bound.md` - 内存带宽瓶颈 → 策略推荐
2. `memory-latency-bound.md` - 内存延迟瓶颈 → 策略推荐
3. `compute-bound.md` - 计算瓶颈 → 策略推荐
4. `low-occupancy.md` - 低占用率 → 策略推荐

- [ ] **Step 1-4: 编写问题模式文档**
- [ ] **Step 5: 提交**

---

### Task 3.4: 实现查询工具（3 个脚本）

**Files:**
- Create: `skills/strategy-library-skill/scripts/*.py`

**从现有代码迁移**：`agents/strategy_templates.py` → scripts/

**脚本列表**：
1. `query_strategy.py` - 策略查询（按瓶颈、算子类型、GPU）
2. `get_strategy.py` - 获取特定策略详情
3. `match_strategy.py` - 策略匹配和推荐

- [ ] **Step 1-3: 实现查询脚本**
- [ ] **Step 4: 添加单元测试**
- [ ] **Step 5: 提交**

---

### Task 3.5: 准备元数据和规则

**Files:**
- Create: `skills/strategy-library-skill/data/strategy_metadata.yaml`
- Create: `skills/strategy-library-skill/data/applicability_rules.yaml`

- [ ] **Step 1: 编写 strategy_metadata.yaml**
- [ ] **Step 2: 编写 applicability_rules.yaml**
- [ ] **Step 3: 提交**

---

## 阶段 3 完成检查点

- [ ] **验证 skill 结构完整**
- [ ] **验证策略文档质量**（每个至少 200 行，包含完整代码模板）
- [ ] **验证查询工具功能**

---

## 阶段 4：optimization-history-skill 实现（2-3 天）

### Task 4.1: 创建 optimization-history-skill 目录结构

**Files:**
- Create: `skills/optimization-history-skill/SKILL.md`
- Create: `skills/optimization-history-skill/README.md`
- Create: `skills/optimization-history-skill/reference/` (4 个文档)
- Create: `skills/optimization-history-skill/helpers/` (5 个脚本)
- Create: `skills/optimization-history-skill/data/history_schema.yaml`

- [ ] **Step 1: 创建目录结构**
- [ ] **Step 2: 提交**

---

### Task 4.2: 编写参考文档（4 个文档）

**Files:**
- Create: `skills/optimization-history-skill/reference/*.md`

**文档列表**：
1. `00-overview.md` - 历史管理概述
2. `01-trend-analysis.md` - 趋势分析方法
3. `02-bottleneck-shift.md` - 瓶颈转移检测
4. `03-strategy-effectiveness.md` - 策略有效性分析
5. `04-recommendation-logic.md` - 推荐逻辑

- [ ] **Step 1-4: 编写参考文档**
- [ ] **Step 5: 提交**

---

### Task 4.3: 实现 Python 工具（5 个脚本）

**Files:**
- Create: `skills/optimization-history-skill/helpers/*.py`

**从现有代码迁移**：`agents/optimization_history.py` → helpers/

**脚本列表**：
1. `record_round.py` - 记录优化轮次
2. `analyze_trend.py` - 趋势分析
3. `detect_shift.py` - 瓶颈转移检测
4. `recommend_strategy.py` - 策略推荐
5. `generate_report.py` - 生成历史报告

- [ ] **Step 1-5: 实现工具脚本**
- [ ] **Step 6: 添加单元测试**
- [ ] **Step 7: 提交**

---

### Task 4.4: 准备历史记录格式

**Files:**
- Create: `skills/optimization-history-skill/data/history_schema.yaml`

- [ ] **Step 1: 编写 history_schema.yaml**
- [ ] **Step 2: 提交**

---

## 阶段 4 完成检查点

- [ ] **验证 skill 结构完整**
- [ ] **验证 Python 工具功能**
- [ ] **验证历史记录格式**

---

## 阶段 5：清理和文档完善（2-3 天）

### Task 5.1: 创建示例任务工作空间

**Files:**
- Create: `examples/matmul-optimization/` (完整示例)

**目录结构**：
```
examples/matmul-optimization/
├── docs/
│   ├── draft.md
│   └── plan.md
├── kernels/
│   ├── baseline.cu
│   ├── candidate_v1.cu
│   └── best.cu
├── profile/
│   ├── baseline.ncu-rep
│   └── candidate_v1.ncu-rep
├── tests/
│   ├── test_correctness.py
│   └── benchmark.py
├── benchmark.csv
├── candidates.jsonl
└── README.md
```

- [ ] **Step 1: 创建目录结构**
- [ ] **Step 2: 编写示例文件**
- [ ] **Step 3: 提交**

---

### Task 5.2: 删除/移动旧代码

**Files:**
- Delete: `agents/` (已迁移到 skills)
- Delete: `utils/` (部分集成到 skills)
- Delete: `prompts/enhanced_*.py`
- Move: `main_enhanced.py` → `examples/legacy/` (可选保留)

- [ ] **Step 1: 备份旧代码**

```bash
mkdir -p backup
cp -r agents/ utils/ prompts/enhanced_*.py backup/
```

- [ ] **Step 2: 删除旧代码**

```bash
git rm -r agents/ utils/
git rm prompts/enhanced_*.py
```

- [ ] **Step 3: 移动 main_enhanced.py**

```bash
mkdir -p examples/legacy
git mv main_enhanced.py examples/legacy/
```

- [ ] **Step 4: 提交**

```bash
git commit -m "chore: remove old code after migration to skills"
```

---

### Task 5.3: 更新 requirements.txt

**Files:**
- Modify: `requirements.txt`

**新的依赖**：
```
# Core dependencies
pyyaml>=6.0
numpy>=1.21.0

# Optional: for NCU report parsing
# pandas>=1.3.0

# Optional: for CUDA compilation
# pycuda>=2021.1
```

- [ ] **Step 1: 更新 requirements.txt**
- [ ] **Step 2: 测试安装**

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: 提交**

---

### Task 5.4: 编写测试

**Files:**
- Create: `tests/test_ncu_interpreter_skill.py`
- Create: `tests/test_strategy_library_skill.py`
- Create: `tests/test_optimization_history_skill.py`

- [ ] **Step 1: 编写单元测试**
- [ ] **Step 2: 运行测试**

```bash
pytest tests/ -v
```

- [ ] **Step 3: 提交**

---

## 阶段 5 完成检查点

- [ ] **验证示例完整**
- [ ] **验证旧代码已删除**
- [ ] **验证测试通过**

---

## 阶段 6：验证和优化（2-3 天）

### Task 6.1: 端到端测试

**目标**：使用新工作流完成一个完整的优化任务

- [ ] **Step 1: 创建测试任务工作空间**

```bash
mkdir -p test-workspace
cd test-workspace
```

- [ ] **Step 2: 定义任务契约**（创建 README.md）
- [ ] **Step 3: 启动 Agent 会话**（使用 prompt 模板）
- [ ] **Step 4: 遵循完整工作流**
  - Profile baseline
  - 写 draft.md
  - 转为 plan.md
  - 实现候选方案
  - 验证 + 测量 + 记录
- [ ] **Step 5: 记录问题和改进点**

---

### Task 6.2: 文档优化

**基于测试反馈优化文档**

- [ ] **Step 1: 修复文档中的错误**
- [ ] **Step 2: 补充缺失的说明**
- [ ] **Step 3: 添加更多示例**
- [ ] **Step 4: 提交**

---

### Task 6.3: 性能对比

**对比新旧架构**

- [ ] **Step 1: 记录旧架构的优化效果**（如果有历史数据）
- [ ] **Step 2: 记录新架构的优化效果**
- [ ] **Step 3: 编写对比报告**
- [ ] **Step 4: 更新 README.md**（添加性能对比）

---

## 阶段 6 完成检查点

- [ ] **端到端测试通过**
- [ ] **文档质量良好**
- [ ] **性能对比完成**

---

## 最终验证清单

### 结构验证

- [ ] 所有目录结构符合设计
- [ ] 所有文件都已创建
- [ ] Git 历史清晰

### 文档验证

- [ ] README.md 完整且清晰
- [ ] CLAUDE.md 已更新
- [ ] 所有 skills 都有 SKILL.md 和 README.md
- [ ] 参考文档完整（ncu-interpreter: 7 个，strategy-library: 13 个，optimization-history: 4 个）

### 代码验证

- [ ] 所有 Python 脚本无语法错误
- [ ] 单元测试通过
- [ ] 代码风格一致

### 功能验证

- [ ] ncu-interpreter-skill 可以解析 NCU 报告
- [ ] strategy-library-skill 可以查询策略
- [ ] optimization-history-skill 可以记录和分析历史
- [ ] 示例任务工作空间完整

### 清理验证

- [ ] 旧代码已删除或移动
- [ ] 无冗余文件
- [ ] requirements.txt 正确

---

## 实施建议

### 执行顺序

1. **严格按阶段顺序执行**：阶段 1 → 2 → 3 → 4 → 5 → 6
2. **每个阶段完成后进行检查点验证**
3. **遇到问题及时调整，不要跳过步骤**

### 时间分配

- 阶段 1：2-3 天（基础架构）
- 阶段 2：3-4 天（ncu-interpreter-skill）
- 阶段 3：4-5 天（strategy-library-skill，最复杂）
- 阶段 4：2-3 天（optimization-history-skill）
- 阶段 5：2-3 天（清理和文档）
- 阶段 6：2-3 天（验证和优化）

**总计**：15-21 天

### 并行化建议

如果有多人协作：
- 阶段 2、3、4 可以并行（三个 skills 独立）
- 阶段 1 必须先完成
- 阶段 5、6 必须在 2、3、4 完成后进行

---

## 执行选项

计划完成并保存到 `docs/superpowers/plans/2026-06-01-kda-alignment-implementation.md`。

**两种执行方式**：

**1. Subagent-Driven（推荐）** - 每个任务派发一个新的 subagent，任务间进行审查，快速迭代

**2. Inline Execution** - 在当前会话中使用 executing-plans 执行任务，批量执行并设置检查点

你希望使用哪种方式？
