# KernelForge-Optimizer 完整配置和运行指南

## 系统要求

### 硬件
- ✅ NVIDIA GPU (你的 RTX 5070 已支持)
- 至少 8GB 显存（推荐）

### 软件
- ✅ Windows 11
- ✅ CUDA Toolkit (包含 nvcc 和 ncu)
- Python 3.8+
- Git

## 第一步：验证 CUDA 环境

打开命令行，运行以下命令验证 CUDA 工具链：

```bash
# 检查 CUDA 编译器
nvcc --version

# 检查 Nsight Compute profiler
ncu --version

# 检查 GPU 信息
nvidia-smi
```

**预期输出：**
- `nvcc --version` 应显示 CUDA 版本（如 12.x）
- `ncu --version` 应显示 Nsight Compute 版本
- `nvidia-smi` 应显示你的 RTX 5070

如果任何命令失败，请从 [NVIDIA 官网](https://developer.nvidia.com/cuda-toolkit) 下载安装 CUDA Toolkit。

## 第二步：配置 API Key

API key 已配置在 `.env` 文件中：

```bash
# 查看配置
cat .env
```

应该看到：
```
DEEPSEEK_API_KEY=sk-eea014b8ac6c48bbad877630e4ecaa35
```

## 第三步：安装 Python 依赖

```bash
cd D:\Agent\KernelForge-Optimizer

# 安装基础依赖
pip install torch numpy openai python-dotenv

# 可选：安装可视化依赖
pip install matplotlib pandas
```

## 第四步：快速测试

### 测试 1：验证环境配置

```bash
python test_real_gpu.py
```

这个脚本会：
1. ✅ 检查 CUDA 工具（nvcc, ncu）
2. ✅ 检测 GPU 架构（应识别为 RTX 5070, Blackwell）
3. ✅ 验证 API key 配置
4. ✅ 测试 DeepSeek API 连接

**预期输出：**
```
=== Environment Check ===
✓ nvcc found: CUDA 12.x
✓ ncu found: Nsight Compute 2024.x
✓ GPU detected: NVIDIA GeForce RTX 5070
✓ Architecture: Blackwell (SM 10.0)
✓ Memory: 12 GB
✓ API key configured
✓ DeepSeek API connection successful
```

### 测试 2：运行简单矩阵乘法优化

```bash
python main_real_gpu.py \
  --kernel_path ../CudaForge-main/CudaForge-main/KernelBench/level1/1_Square_matrix_multiplication_.py \
  --max_rounds 3 \
  --server_type deepseek \
  --model_name deepseek-chat
```

这会：
1. 从 CudaForge 的 KernelBench 加载矩阵乘法算子
2. 编译并运行 NCU profiling
3. 使用 NCU 解释器分析性能瓶颈
4. 从策略模板库选择优化方案
5. 调用 DeepSeek API 生成优化代码
6. 验证优化效果
7. 迭代 3 轮

**预期输出：**
```
=== Round 1 ===
Compiling kernel...
Running NCU profiling...
Analyzing performance...
  Bottleneck: Memory Bandwidth (85% utilization)
  Access Pattern: Strided (inefficient)
  Recommendation: Apply shared memory tiling

Selecting strategy...
  Matched template: matmul_shared_memory_tiling
  Parameters: tile_size=32, block_size=(16,16)

Generating optimized code...
  Using DeepSeek API...
  Generated 156 lines

Verifying...
  Speedup: 2.3x ✓

=== Round 2 ===
...
```

## 第五步：完整优化流程

### 优化单个算子

```bash
# 矩阵乘法
python main_real_gpu.py \
  --kernel_path ../CudaForge-main/CudaForge-main/KernelBench/level1/1_Square_matrix_multiplication_.py \
  --max_rounds 5

# Reduction 操作
python main_real_gpu.py \
  --kernel_path ../CudaForge-main/CudaForge-main/KernelBench/level1/5_Sum_reduction_.py \
  --max_rounds 5

# 激活函数
python main_real_gpu.py \
  --kernel_path ../CudaForge-main/CudaForge-main/KernelBench/level1/3_ReLU_activation_.py \
  --max_rounds 3
```

### 批量优化（面试演示推荐）

```bash
# 优化多个算子，生成对比报告
python batch_optimize.py \
  --kernel_dir ../CudaForge-main/CudaForge-main/KernelBench/level1 \
  --output_dir results \
  --max_rounds 5
```

## 第六步：查看结果

优化结果保存在 `results/` 目录：

```
results/
├── matmul_optimization_report.json    # 详细优化记录
├── matmul_final.cu                    # 最终优化代码
├── matmul_performance.png             # 性能对比图
└── summary.md                         # 汇总报告
```

查看汇总报告：
```bash
cat results/summary.md
```

## 常见问题

### Q1: `nvcc: command not found`
**解决：** CUDA Toolkit 未安装或未添加到 PATH
```bash
# 添加到 PATH（临时）
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
```

### Q2: `ncu: command not found`
**解决：** Nsight Compute 未安装
- 重新运行 CUDA Toolkit 安装程序
- 确保勾选 "Nsight Compute" 组件

### Q3: API 调用失败
**解决：** 检查网络连接和 API key
```bash
# 测试 API 连接
curl https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer sk-eea014b8ac6c48bbad877630e4ecaa35"
```

### Q4: 编译错误 `sm_100 not supported`
**解决：** CUDA Toolkit 版本过旧，不支持 Blackwell 架构
- 升级到 CUDA 12.6+ 
- 或在代码中降级到 `sm_89` (Ada Lovelace)

### Q5: 优化效果不明显
**可能原因：**
- 初始代码已经很优化
- 算子规模太小，无法体现优化效果
- 尝试更大的输入尺寸：`--input_size 2048`

## 面试演示建议

### 演示流程（15-20分钟）

1. **环境展示（2分钟）**
   ```bash
   nvidia-smi  # 展示 RTX 5070
   python test_real_gpu.py  # 验证环境
   ```

2. **核心能力讲解（3分钟）**
   - NCU 指标解释器：将原始性能数据转换为可操作的诊断
   - 策略模板库：9个经过验证的优化模式
   - 优化历史管理：智能迭代和策略推荐

3. **实时优化演示（8分钟）**
   ```bash
   # 选择一个有明显优化空间的算子
   python main_real_gpu.py \
     --kernel_path ../CudaForge-main/CudaForge-main/KernelBench/level1/1_Square_matrix_multiplication_.py \
     --max_rounds 3 \
     --verbose
   ```
   
   **讲解要点：**
   - Round 1: 识别内存带宽瓶颈 → 应用 shared memory tiling → 2-3x 加速
   - Round 2: 检测到瓶颈转移到 compute → 应用 vectorized load → 1.5x 加速
   - Round 3: 进一步优化占用率 → 调整 block size → 1.2x 加速
   - 总加速比：~4-5x

4. **结果分析（3分钟）**
   ```bash
   cat results/matmul_optimization_report.json
   ```
   
   展示：
   - 每轮的性能指标变化
   - 瓶颈类型的转移
   - 策略选择的依据
   - 最终加速比

5. **技术亮点总结（2分钟）**
   - 确定性算法 + LLM 的混合方案
   - 基于知识库的策略选择（而非每次重新发明）
   - 迭代优化中的智能决策

### 准备工作

提前运行一次完整流程，确保：
- ✅ 所有依赖正常
- ✅ API 调用稳定
- ✅ 优化效果明显（选择合适的测试算子）
- ✅ 准备好结果截图（以防现场网络问题）

### 备选方案

如果现场无法访问 GPU：
```bash
# 使用演示模式（模拟数据）
python demo_optimization_flow.py
```

## 下一步

- 📖 阅读 [EXAMPLES.md](EXAMPLES.md) 了解更多使用场景
- 🔧 查看 [agents/](agents/) 目录了解核心实现
- 📊 运行 `batch_optimize.py` 生成完整的性能对比报告
- 🎯 尝试优化你自己的 CUDA kernel

## 技术支持

遇到问题？
1. 查看 [EXAMPLES.md](EXAMPLES.md) 中的常见场景
2. 检查 `logs/` 目录中的详细日志
3. 参考 CudaForge 原项目文档

祝面试顺利！🚀
