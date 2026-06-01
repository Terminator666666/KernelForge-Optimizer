# Fused MoE (简化版) 优化任务草稿

## 任务目标

优化 Fused MoE (简化版) 算子，在 RTX 5070 上实现显著的性能提升。

## 算子描述

混合专家模型优化 - Gating + Expert Batching

## Baseline 实现

**复杂度**: 分离的 Gating 和 Expert 计算

**特点**:
- 多次内存访问
- 未充分利用并行性
- 未使用共享内存优化

## 优化策略

1. 融合 Gating 和 Expert 选择
2. Expert batching
3. 共享内存优化
4. 动态并行

## 预期性能提升

**目标加速比**: 2-4x

## 实现计划

### 阶段 1: Baseline 实现
- [ ] 实现简单的并行版本（公平 baseline）
- [ ] 编译和功能测试
- [ ] 性能基准测试

### 阶段 2: 优化实现
- [ ] 应用优化策略
- [ ] 编译和功能测试
- [ ] 性能对比测试

### 阶段 3: 分析和验证
- [ ] NCU profiling 分析
- [ ] 精度验证（误差 < 0.01%）
- [ ] 记录候选方案

### 阶段 4: 文档和提交
- [ ] 更新 CLAUDE.md
- [ ] Git 提交
- [ ] 推送到 GitHub

## 验收标准

1. ✅ 精度验证通过（误差 < 0.01%）
2. ✅ 加速比达到预期范围
3. ✅ 公平的 baseline 对比（并行 vs 并行）
4. ✅ 完整的 agentic workflow（draft + candidates + evidence）
5. ✅ NCU profiling 数据
6. ✅ Git 提交历史清晰

## 参考资料

- CUDA Programming Guide
- NVIDIA Nsight Compute Documentation
- 相关学术论文

---

**创建时间**: 2026-06-01 20:39:25
**状态**: 待实现
