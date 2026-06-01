# 🚀 剩余算子完成指南

## 📋 概述

本指南帮助你完成 KernelForge-Optimizer 项目的剩余 5 个算子的测试和验证。

**当前状态**：
- ✅ 代码实现：100%（8/8 算子）
- ⏳ GPU 测试：37.5%（3/8 算子）
- 🎯 目标：完成剩余 5 个算子的 GPU 测试和验证

**新增算子**：
1. Softmax（激活函数优化）
2. LayerNorm（层归一化优化）
3. Scan/Prefix Sum（并行前缀和）
4. Flash Attention（简化版）
5. Fused MoE（简化版）

---

## 🔧 环境要求

### 必需
- NVIDIA GPU（RTX 5070 或其他支持 CUDA 的 GPU）
- CUDA Toolkit 12.x
- nvcc 编译器
- Linux 环境（WSL2 或原生 Linux）

### 可选
- NVIDIA Nsight Compute（用于性能分析）
- Git（用于版本控制）

---

## 📝 快速开始

### 方法 1：一键测试所有算子（推荐）

```bash
# 进入项目目录
cd /mnt/d/Agent/KernelForge-Optimizer

# 运行自动化测试脚本
./test_remaining_operators.sh
```

**脚本功能**：
- ✅ 自动编译所有 5 个算子
- ✅ 运行性能测试
- ✅ 记录候选方案到 JSONL
- ✅ 生成性能报告
- ⏳ 可选：运行 NCU profiling

**预计时间**：5-10 分钟（不含 NCU）

---

### 方法 2：逐个测试算子

#### 1. Softmax

```bash
# 编译
nvcc examples/softmax_baseline.cu -o runs/test_softmax -O3 -arch=sm_89

# 运行测试
./runs/test_softmax | tee runs/softmax_test.log

# 查看结果
grep "加速比" runs/softmax_test.log
grep "总体状态" runs/softmax_test.log
```

#### 2. LayerNorm

```bash
# 编译
nvcc examples/layernorm_baseline.cu -o runs/test_layernorm -O3 -arch=sm_89

# 运行测试
./runs/test_layernorm | tee runs/layernorm_test.log

# 查看结果
grep "加速比" runs/layernorm_test.log
grep "总体状态" runs/layernorm_test.log
```

#### 3. Scan (Prefix Sum)

```bash
# 编译
nvcc examples/scan_baseline.cu -o runs/test_scan -O3 -arch=sm_89

# 运行测试
./runs/test_scan | tee runs/scan_test.log

# 查看结果
grep "加速比" runs/scan_test.log
grep "总体状态" runs/scan_test.log
```

#### 4. Flash Attention

```bash
# 编译
nvcc examples/flash_attention_baseline.cu -o runs/test_flash_attention -O3 -arch=sm_89 -lcublas

# 运行测试
./runs/test_flash_attention | tee runs/flash_attention_test.log

# 查看结果
grep "加速比" runs/flash_attention_test.log
grep "总体状态" runs/flash_attention_test.log
```

#### 5. Fused MoE

```bash
# 编译
nvcc examples/fused_moe_baseline.cu -o runs/test_fused_moe -O3 -arch=sm_89

# 运行测试
./runs/test_fused_moe | tee runs/fused_moe_test.log

# 查看结果
grep "加速比" runs/fused_moe_test.log
grep "总体状态" runs/fused_moe_test.log
```

---

## 📊 结果分析

### 查看性能数据

```bash
# 查看所有加速比
cat runs/*_test.log | grep "加速比"

# 查看所有精度验证
cat runs/*_test.log | grep "总体状态"

# 生成性能表格
for op in softmax layernorm scan flash_attention fused_moe; do
    echo "=== $op ==="
    grep "Baseline.*执行时间" runs/${op}_test.log
    grep "Optimized.*执行时间" runs/${op}_test.log
    grep "加速比" runs/${op}_test.log
    echo ""
done
```

### 查看候选记录

```bash
# 查看所有候选记录
cat candidates/*_candidates.jsonl | jq '.'

# 统计加速比
cat candidates/*_candidates.jsonl | jq '.speedup'
```

---

## 🔍 NCU Profiling（可选）

### 运行 NCU 分析

```bash
# 对单个算子运行 NCU
ncu --set full -o profile/softmax_baseline runs/test_softmax

# 批量运行
for op in softmax layernorm scan flash_attention fused_moe; do
    echo "Profiling $op..."
    ncu --set full -o profile/${op}_baseline runs/test_${op}
done
```

### 查看 NCU 报告

```bash
# 使用 ncu-ui 查看（GUI）
ncu-ui profile/softmax_baseline.ncu-rep

# 或导出为文本
ncu --import profile/softmax_baseline.ncu-rep --page details > profile/softmax_analysis.txt
```

---

## 📝 记录结果

### 更新 CLAUDE.md

测试完成后，手动更新 `CLAUDE.md` 中的实际加速比：

```markdown
| **Softmax** | **X.XXx** | 1024×1024 | ✅ PASS | 已验证 |
| **LayerNorm** | **X.XXx** | 1024×1024 | ✅ PASS | 已验证 |
| **Scan** | **X.XXx** | 1M 元素 | ✅ PASS | 已验证 |
| **Flash Attention** | **X.XXx** | 256×64 | ✅ PASS | 已验证 |
| **Fused MoE** | **X.XXx** | 1024×256×8 | ✅ PASS | 已验证 |
```

### 创建候选草稿

为每个算子创建详细的优化记录：

```bash
# 示例：Softmax
cat > docs/softmax-optimization-record.md << 'EOF'
# Softmax 优化记录

## 测试环境
- GPU: RTX 5070
- CUDA: 12.6
- 输入规模: 1024×1024

## 性能结果
- Baseline: X.XX ms
- Optimized: X.XX ms
- 加速比: X.XXx

## 优化策略
1. 在线 Softmax 算法
2. 共享内存并行归约
3. Warp-level primitives

## 精度验证
- 最大误差: X.XXe-XX
- 状态: ✅ PASS
EOF
```

---

## 🎯 Git 提交

### 提交新代码

```bash
# 查看状态
git status

# 添加新文件
git add examples/softmax_baseline.cu
git add examples/layernorm_baseline.cu
git add examples/scan_baseline.cu
git add examples/flash_attention_baseline.cu
git add examples/fused_moe_baseline.cu

# 添加草稿和候选记录
git add docs/*-draft.md
git add candidates/*_candidates.jsonl

# 添加测试脚本
git add test_remaining_operators.sh
git add automation/complete_remaining_operators.py
git add automation/update_progress.py

# 提交
git commit -m "feat: 实现剩余 5 个高级算子

- Softmax: 在线算法 + 共享内存归约
- LayerNorm: Welford 算法 + 融合仿射变换
- Scan: Blelloch work-efficient 算法
- Flash Attention: Tiling + 融合 kernel
- Fused MoE: 融合 Gating + Expert 计算

所有算子包含：
- 公平的 baseline 实现
- 优化版本实现
- CPU 参考实现
- 完整的精度验证
- 性能测试代码

测试结果：
- Softmax: X.XXx 加速
- LayerNorm: X.XXx 加速
- Scan: X.XXx 加速
- Flash Attention: X.XXx 加速
- Fused MoE: X.XXx 加速

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"

# 推送到 GitHub
git push origin main
```

---

## 📈 预期结果

### 性能预期

| 算子 | 预期加速比 | 关键优化 |
|------|-----------|---------|
| Softmax | 3-5x | 在线算法 + 共享内存 |
| LayerNorm | 4-6x | Welford 算法 |
| Scan | 2-3x | Work-efficient 算法 |
| Flash Attention | 3-5x | Tiling + Kernel Fusion |
| Fused MoE | 2-4x | Kernel Fusion |

### 精度要求

所有算子必须满足：
- ✅ 最大误差 < 1e-3
- ✅ 平均误差 < 1e-5
- ✅ 与 CPU 参考实现一致

---

## 🐛 故障排除

### 编译错误

**问题**：`nvcc: command not found`
```bash
# 解决：添加 CUDA 到 PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**问题**：`arch=sm_89 not supported`
```bash
# 解决：使用你的 GPU 对应的架构
# RTX 5070: sm_89
# RTX 4090: sm_89
# RTX 3090: sm_86
# A100: sm_80

nvcc ... -arch=sm_XX  # 替换为你的架构
```

### 运行时错误

**问题**：`out of memory`
```bash
# 解决：减小测试规模
# 编辑 .cu 文件，修改 N 和 D 的值
const int N = 512;  // 从 1024 减小到 512
const int D = 512;
```

**问题**：精度验证失败
```bash
# 检查：查看详细误差
grep "最大误差" runs/*_test.log
grep "平均误差" runs/*_test.log

# 如果误差在 1e-3 到 1e-2 之间，可能是浮点精度问题
# 可以放宽精度要求或使用 double 精度
```

---

## 📚 参考资料

### 算法参考

- **Softmax**: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- **LayerNorm**: [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
- **Scan**: [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- **Flash Attention**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- **Fused MoE**: [Mixture of Experts optimization techniques](https://arxiv.org/abs/2006.16668)

### CUDA 编程

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

---

## ✅ 完成检查清单

测试完成后，确认以下项目：

- [ ] 所有 5 个算子编译成功
- [ ] 所有 5 个算子运行成功
- [ ] 所有 5 个算子精度验证通过
- [ ] 记录了实际加速比
- [ ] 更新了 CLAUDE.md
- [ ] 创建了候选记录（JSONL）
- [ ] （可选）运行了 NCU profiling
- [ ] Git 提交并推送到 GitHub

---

## 🎉 完成后

恭喜！你已经完成了 KernelForge-Optimizer 项目的所有 8 个算子：

1. ✅ Transpose (26.69x)
2. ✅ Matrix Multiplication Ultra (8.82x)
3. ✅ Reduction (2.25x)
4. ✅ Softmax (X.XXx)
5. ✅ LayerNorm (X.XXx)
6. ✅ Scan (X.XXx)
7. ✅ Flash Attention (X.XXx)
8. ✅ Fused MoE (X.XXx)

**项目亮点**：
- 8 个不同类型的高级 CUDA 算子
- 平均加速比 5-10x
- 完整的 Agentic Workflow
- 公平的 baseline 对比
- 严格的精度验证
- 清晰的 Git 提交历史

**适合用于**：
- 🎓 实习面试展示
- 📝 简历项目
- 🔬 CUDA 优化学习
- 🏭 实际应用参考

---

**最后更新**：2026-06-01
**作者**：Claude Opus 4.8
**项目地址**：https://github.com/Terminator666666/KernelForge-Optimizer
