# KernelForge-Optimizer 使用指南

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保安装了 NVIDIA Nsight Compute
# 下载地址: https://developer.nvidia.com/nsight-compute
```

### 2. 基本使用

#### 方式一：命令行模式

```bash
python main_enhanced.py \
    --kernel examples/matmul_naive.cu \
    --kernel-name matmul_naive \
    --operator-type matmul \
    --max-iterations 5 \
    --output optimized_matmul.cu
```

#### 方式二：Python API

```python
from main_enhanced import KernelForgeOptimizer

# 初始化优化器（自动检测 GPU）
optimizer = KernelForgeOptimizer()

# 读取 kernel 代码
with open('examples/matmul_naive.cu', 'r') as f:
    kernel_code = f.read()

# 运行优化
result = optimizer.optimize_kernel(
    kernel_code=kernel_code,
    kernel_name='matmul_naive',
    operator_type='matmul'
)

# 查看结果
print(f"总加速比: {result['total_speedup']:.2f}x")
print(f"最佳时间: {result['best_time_ms']:.3f} ms")
print(f"优化轮数: {result['iterations']}")
```

### 3. 独立使用核心模块

#### NCU 指标解释器

```python
from agents import create_interpreter_for_gpu

# 创建解释器
interpreter = create_interpreter_for_gpu('RTX 5070')

# 解释 NCU 指标
ncu_metrics = {
    'dram__bytes.sum': 1e9,
    'duration': 1e6,
    'sm__warps_active.avg.pct_of_peak_sustained_active': 60.0,
    # ... 更多指标
}

diagnosis = interpreter.interpret(ncu_metrics)

# 查看诊断结果
print(f"主要瓶颈: {diagnosis.bottleneck}")
print(f"带宽利用率: {diagnosis.memory_bandwidth_util:.1f}%")
print(f"占用率: {diagnosis.achieved_occupancy:.1f}%")
print(f"算术强度: {diagnosis.arithmetic_intensity:.2f} FLOPs/byte")

# 查看优化建议
for issue in diagnosis.issues[:3]:
    print(f"- {issue['description']}")
    print(f"  建议: {issue['suggestion']}")
```

#### 策略模板库

```python
from agents import create_strategy_library, OperatorType, BottleneckType

# 创建策略库
library = create_strategy_library()

# 获取适用的策略
strategies = library.get_applicable_strategies(
    operator_type=OperatorType.MATMUL,
    bottleneck=BottleneckType.MEMORY_BANDWIDTH,
    gpu_compute_capability=10.0  # RTX 5070
)

# 查看推荐策略
for strategy in strategies[:3]:
    print(f"\n策略: {strategy.name}")
    print(f"描述: {strategy.description}")
    print(f"预期加速: {strategy.expected_speedup[0]:.1f}x - {strategy.expected_speedup[1]:.1f}x")
    print(f"参数: {list(strategy.parameter_rules.keys())}")
```

#### 优化历史管理

```python
from agents import create_optimization_history, OptimizationRound
from datetime import datetime

# 创建历史管理器
history = create_optimization_history()

# 记录优化轮次
round_data = OptimizationRound(
    round_id=1,
    timestamp=datetime.now().isoformat(),
    kernel_code="",
    operator_type="matmul",
    gpu_name="RTX 5070",
    bottleneck="memory_bandwidth",
    bottleneck_confidence=0.85,
    bandwidth_util=75.0,
    occupancy=60.0,
    arithmetic_intensity=15.0,
    strategy_name="matmul_tiling",
    strategy_params={'TILE_SIZE': 32},
    execution_time_ms=2.5,
    speedup=1.8,
    compilation_success=True,
    runtime_success=True
)

history.add_round(round_data)

# 获取推荐策略
available_strategies = ['matmul_tiling', 'vectorized_memory', 'tensor_core']
recommended, reasoning = history.recommend_next_strategy(
    current_bottleneck='memory_bandwidth',
    available_strategies=available_strategies
)

print(f"推荐策略: {recommended}")
print(f"理由: {reasoning}")

# 生成报告
report = history.generate_report()
print(report)
```

## 高级用法

### 集成到 CudaForge

在 CudaForge 的 `main.py` 中集成增强模块：

```python
# 在 CudaForge main.py 开头添加
from agents import create_interpreter_for_gpu, create_strategy_library
from prompts import build_enhanced_judge_prompt, build_enhanced_optimization_prompt

# 初始化增强模块
ncu_interpreter = create_interpreter_for_gpu(gpu_name)
strategy_library = create_strategy_library()

# 在优化循环中替换原有的 Judge 调用
# 原来:
# judge_prompt = build_judge_prompt(ncu_metrics, current_code)

# 现在:
diagnosis = ncu_interpreter.interpret(ncu_metrics)
judge_prompt = build_enhanced_judge_prompt(
    diagnosis, operator_type, current_code
)

# 替换 Generator 调用
# 原来:
# generator_prompt = build_optimization_prompt(current_code, optimization_goal)

# 现在:
strategies = strategy_library.get_applicable_strategies(
    operator_type, bottleneck_type, compute_capability
)
selected_strategy = strategies[0]  # 或使用历史推荐

generator_prompt = build_enhanced_optimization_prompt(
    diagnosis, selected_strategy, operator_type, current_code, gpu_specs
)
```

### 自定义配置

```python
config = {
    'max_iterations': 15,              # 最大优化轮数
    'min_speedup_threshold': 1.1,      # 最小加速比阈值（10%）
    'enable_history': True,            # 启用历史推荐
    'save_history': True,              # 保存历史到文件
    'history_file': 'my_history.json', # 历史文件路径
    'verbose': True                    # 详细日志
}

optimizer = KernelForgeOptimizer(gpu_name='RTX 5070', config=config)
```

### 添加自定义策略模板

```python
from agents.strategy_templates import StrategyTemplate, OperatorType, BottleneckType

# 定义自定义策略
custom_strategy = StrategyTemplate(
    name='my_custom_optimization',
    description='My custom optimization technique',
    operator_types=[OperatorType.MATMUL],
    bottleneck_types=[BottleneckType.MEMORY_BANDWIDTH],
    min_gpu_compute_capability=8.0,
    expected_speedup=(1.5, 3.0),
    memory_reduction=40.0,
    code_template="""
    // Your custom CUDA code template here
    __global__ void my_kernel(...) {
        // Implementation
    }
    """,
    parameter_rules={
        'MY_PARAM': {
            'type': 'int',
            'candidates': [16, 32, 64],
            'default': 32
        }
    },
    notes='Custom optimization notes'
)

# 添加到策略库
library = create_strategy_library()
library.templates['my_custom_optimization'] = custom_strategy
```

## 常见问题

### Q1: 如何查看详细的优化过程？

设置 `verbose=True` 并查看控制台输出，或者读取保存的历史文件：

```python
from agents import OptimizationHistory

history = OptimizationHistory.load_from_file('optimization_history.json')
print(history.generate_report())
```

### Q2: 如何针对特定 GPU 优化？

```python
# 指定 GPU 型号
optimizer = KernelForgeOptimizer(gpu_name='RTX 5070')

# 或者使用自定义 GPU 规格
from utils import GPUArchitecture

custom_specs = {
    'peak_bandwidth_gbps': 672,
    'peak_tflops_fp32': 38,
    'sm_count': 48,
    # ... 其他规格
}

from agents import NCUInterpreter
interpreter = NCUInterpreter(custom_specs)
```

### Q3: 如何处理编译错误？

优化器会自动记录编译错误并尝试其他策略。查看历史记录：

```python
for round_data in history.rounds:
    if not round_data.compilation_success:
        print(f"Round {round_data.round_id} 编译失败:")
        print(f"  策略: {round_data.strategy_name}")
        print(f"  错误: {round_data.error_message}")
```

### Q4: 如何评估优化效果？

```python
summary = history.get_summary()
print(f"总轮数: {summary['total_rounds']}")
print(f"成功率: {summary['success_rate']:.1f}%")
print(f"总加速比: {summary['total_speedup']:.2f}x")

# 查看策略有效性
effectiveness = history.analyze_strategy_effectiveness()
for eff in effectiveness:
    print(f"{eff.strategy_name}: {eff.avg_speedup:.2f}x 平均加速")
```

## 性能调优建议

### 针对不同算子类型

**矩阵乘法 (MatMul)**
- 优先尝试：共享内存 tiling → Tensor Core → 向量化
- 关键参数：TILE_SIZE (16/32/64)
- 预期加速：3-10x

**归约操作 (Reduction)**
- 优先尝试：Warp primitives → 树形归约 → 占用率优化
- 关键参数：BLOCK_SIZE (256/512)
- 预期加速：3-8x

**逐元素操作 (Elementwise)**
- 优先尝试：Kernel fusion → 向量化 → 内存合并
- 关键参数：VECTOR_SIZE (2/4)
- 预期加速：2-4x

### 针对不同瓶颈

**内存带宽瓶颈**
- 使用共享内存减少全局内存访问
- 向量化内存访问（float4）
- Kernel fusion 减少中间结果写入

**内存延迟瓶颈**
- 提高占用率隐藏延迟
- 改善内存访问模式（coalescing）
- 使用 L1/L2 缓存

**计算瓶颈**
- 使用 Tensor Cores（FP16/TF32）
- 优化指令混合
- 减少寄存器压力

**占用率瓶颈**
- 减少寄存器使用（__launch_bounds__）
- 减少共享内存使用
- 调整 block size

## 更多资源

- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)
- [CudaForge 原始项目](https://github.com/CudaForge/CudaForge)
- [Roofline 模型论文](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
