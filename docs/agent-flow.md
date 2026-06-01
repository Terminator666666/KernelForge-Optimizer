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
