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
