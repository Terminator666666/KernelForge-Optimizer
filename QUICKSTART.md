# KernelForge-Optimizer 快速启动指南

## 当前状态说明

✅ **已完成**：所有核心代码、文档、测试框架
⚠️ **需要补充**：LLM API 集成、NCU profiling 集成

目前项目可以：
1. ✅ 展示代码质量和设计思路（面试演示）
2. ✅ 运行单元测试验证核心逻辑
3. ✅ 独立使用各个模块（NCU 解释器、策略库、历史管理）
4. ⚠️ 端到端优化需要补充 LLM 和 NCU 集成

## 方式一：运行单元测试（推荐，无需额外环境）

这是最快速验证项目功能的方式。

```bash
# 1. 安装依赖
cd KernelForge-Optimizer
pip install pytest numpy pandas

# 2. 运行所有测试
pytest tests/ -v

# 3. 运行特定测试
pytest tests/test_ncu_interpreter.py -v
pytest tests/test_strategy_templates.py -v

# 4. 查看测试覆盖率
pip install pytest-cov
pytest --cov=agents --cov=utils tests/
```

**测试内容**：
- NCU 解释器的瓶颈识别
- 策略模板的匹配和选择
- GPU 架构检测
- 参数选择逻辑

## 方式二：独立使用核心模块（演示用）

### 2.1 使用 NCU 解释器

```python
# demo_ncu_interpreter.py
from agents import create_interpreter_for_gpu

# 创建解释器（自动检测你的 RTX 5070）
interpreter = create_interpreter_for_gpu('RTX 5070')

# 模拟 NCU 指标（实际使用时从 NCU 获取）
ncu_metrics = {
    'dram__bytes.sum': 1e9,  # 1 GB 内存传输
    'duration': 1e6,  # 1 ms (纳秒)
    'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum': 1e6,
    'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum': 1e6,
    'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 1e6,
    'sm__warps_active.avg.pct_of_peak_sustained_active': 60.0,
    'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed': 70.0,
}

# 解释指标
diagnosis = interpreter.interpret(ncu_metrics)

# 查看诊断结果
print(f"主要瓶颈: {diagnosis.bottleneck}")
print(f"置信度: {diagnosis.bottleneck_confidence:.0%}")
print(f"带宽利用率: {diagnosis.memory_bandwidth_util:.1f}%")
print(f"占用率: {diagnosis.achieved_occupancy:.1f}%")
print(f"算术强度: {diagnosis.arithmetic_intensity:.2f} FLOPs/byte")
print(f"Roofline 区域: {diagnosis.roofline_region}")

print("\n优先级问题:")
for i, issue in enumerate(diagnosis.issues[:3], 1):
    print(f"{i}. [{issue['severity']}] {issue['description']}")
    if issue['suggestion']:
        print(f"   建议: {issue['suggestion']}")
```

运行：
```bash
cd KernelForge-Optimizer
python demo_ncu_interpreter.py
```

### 2.2 使用策略模板库

```python
# demo_strategy_library.py
from agents import create_strategy_library, OperatorType, BottleneckType

# 创建策略库
library = create_strategy_library()

# 获取适用的策略
strategies = library.get_applicable_strategies(
    operator_type=OperatorType.MATMUL,
    bottleneck=BottleneckType.MEMORY_BANDWIDTH,
    gpu_compute_capability=10.0  # RTX 5070
)

print(f"找到 {len(strategies)} 个适用策略:\n")

for i, strategy in enumerate(strategies[:3], 1):
    print(f"{i}. {strategy.name}")
    print(f"   描述: {strategy.description}")
    print(f"   预期加速: {strategy.expected_speedup[0]:.1f}x - {strategy.expected_speedup[1]:.1f}x")
    print(f"   参数: {list(strategy.parameter_rules.keys())}")
    print()

# 查看具体策略的代码模板
print("=" * 80)
print("matmul_tiling 策略代码模板:")
print("=" * 80)
template = library.templates['matmul_tiling']
print(template.code_template[:500] + "...")
```

运行：
```bash
python demo_strategy_library.py
```

### 2.3 使用优化历史管理

```python
# demo_optimization_history.py
from agents import create_optimization_history, OptimizationRound
from datetime import datetime

# 创建历史管理器
history = create_optimization_history()

# 模拟几轮优化
rounds = [
    OptimizationRound(
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
    ),
    OptimizationRound(
        round_id=2,
        timestamp=datetime.now().isoformat(),
        kernel_code="",
        operator_type="matmul",
        gpu_name="RTX 5070",
        bottleneck="memory_bandwidth",
        bottleneck_confidence=0.75,
        bandwidth_util=85.0,
        occupancy=65.0,
        arithmetic_intensity=20.0,
        strategy_name="vectorized_memory",
        strategy_params={'VECTOR_SIZE': 4},
        execution_time_ms=1.8,
        speedup=1.4,
        compilation_success=True,
        runtime_success=True
    ),
]

for round_data in rounds:
    history.add_round(round_data)

# 获取推荐
available_strategies = ['matmul_tiling', 'vectorized_memory', 'tensor_core']
recommended, reasoning = history.recommend_next_strategy(
    current_bottleneck='memory_bandwidth',
    available_strategies=available_strategies
)

print(f"推荐策略: {recommended}")
print(f"理由: {reasoning}\n")

# 生成报告
print(history.generate_report())
```

运行：
```bash
python demo_optimization_history.py
```

## 方式三：完整端到端运行（需要额外集成）

要运行完整的优化流程，需要补充两个集成：

### 3.1 补充 LLM API 集成

在 `main_enhanced.py` 中找到 `_call_llm` 方法，添加实际的 LLM 调用：

```python
def _call_llm(self, prompt: str) -> str:
    """调用 LLM 生成代码"""
    
    # 选项 1: 使用 OpenAI API
    import openai
    openai.api_key = "your-api-key"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content
    
    # 选项 2: 使用 DeepSeek API
    # import requests
    # response = requests.post(
    #     "https://api.deepseek.com/v1/chat/completions",
    #     headers={"Authorization": f"Bearer {api_key}"},
    #     json={"model": "deepseek-coder", "messages": [{"role": "user", "content": prompt}]}
    # )
    # return response.json()['choices'][0]['message']['content']
    
    # 选项 3: 使用 CudaForge 的 query_server
    # from CudaForge-main.agents.query_server import query_model
    # return query_model(prompt, model="gpt-4")
```

### 3.2 补充 NCU Profiling 集成

在 `main_enhanced.py` 中找到 `_profile_kernel` 方法：

```python
def _profile_kernel(self, kernel_code: str, kernel_name: str) -> Dict[str, float]:
    """使用 NCU 分析 kernel"""
    
    # 选项 1: 使用 CudaForge 的 run_ncu.py
    import sys
    sys.path.insert(0, '../CudaForge-main')
    from run_ncu import profile_kernel
    
    metrics = profile_kernel(kernel_code, kernel_name)
    return metrics
    
    # 选项 2: 直接调用 NCU 命令行
    # import subprocess
    # import json
    # 
    # # 编译 kernel
    # compile_kernel(kernel_code, 'temp_kernel.cu')
    # 
    # # 运行 NCU
    # result = subprocess.run([
    #     'ncu', '--metrics', 'dram__bytes.sum,duration,...',
    #     '--csv', './temp_kernel'
    # ], capture_output=True, text=True)
    # 
    # # 解析输出
    # metrics = parse_ncu_output(result.stdout)
    # return metrics
```

### 3.3 运行完整优化

```bash
# 补充集成后运行
python main_enhanced.py \
    --kernel examples/matmul_naive.cu \
    --kernel-name matmul_naive \
    --operator-type matmul \
    --max-iterations 5 \
    --output optimized_matmul.cu
```

## 方式四：集成到 CudaForge（推荐用于实际优化）

如果你已经有 CudaForge 环境：

```python
# 在 CudaForge-main/main.py 中添加
import sys
sys.path.insert(0, '../KernelForge-Optimizer')

from agents import create_interpreter_for_gpu, create_strategy_library
from prompts import build_enhanced_judge_prompt, build_enhanced_optimization_prompt

# 初始化增强模块
ncu_interpreter = create_interpreter_for_gpu(gpu_name)
strategy_library = create_strategy_library()

# 在优化循环中替换原有调用
# ... (参考 docs/usage_guide.md 中的集成说明)
```

## 快速演示脚本（面试用）

创建一个完整的演示脚本：

```bash
# demo_all.sh
echo "=== KernelForge-Optimizer 演示 ==="
echo ""

echo "1. 运行单元测试..."
pytest tests/test_ncu_interpreter.py::TestNCUInterpreter::test_memory_bound_detection -v
echo ""

echo "2. 演示 NCU 解释器..."
python demo_ncu_interpreter.py
echo ""

echo "3. 演示策略库..."
python demo_strategy_library.py
echo ""

echo "4. 演示历史管理..."
python demo_optimization_history.py
echo ""

echo "=== 演示完成 ==="
```

## 常见问题

**Q: 为什么不能直接运行完整优化？**
A: 需要 LLM API（OpenAI/DeepSeek）和 CUDA 环境。核心逻辑已完成，只需补充这两个外部依赖的集成代码（约 50 行）。

**Q: 如何验证代码质量？**
A: 运行单元测试：`pytest tests/ -v`

**Q: 如何展示给面试官？**
A: 
1. 展示项目结构和文档
2. 运行单元测试展示核心功能
3. 运行演示脚本展示各模块
4. 讲解代码设计和技术细节

**Q: 需要 GPU 才能运行吗？**
A: 不需要。单元测试和演示脚本都是纯 Python 逻辑，可以在任何机器上运行。只有实际优化 CUDA kernel 时才需要 GPU。
