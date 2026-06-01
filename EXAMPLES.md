# Usage Examples

## Example 1: Quick Test with Matrix Multiplication

```bash
# Set your API key
export DEEPSEEK_API_KEY="your-api-key-here"

# Run quick test
python test_real_gpu.py --kernel-path ../CudaForge-main/KernelBench/level1/1_Square_matrix_multiplication_.py
```

**Expected Output:**
```
=== Testing Kernel: 1_Square_matrix_multiplication_ ===
Operator Type: matmul
GPU: NVIDIA GeForce RTX 5070 (Blackwell, SM 10.0)

Running NCU profiling...
✓ Profiling completed

Performance Metrics:
- Execution Time: 2.45 ms
- Memory Bandwidth: 245 GB/s (36.5% of peak)
- Compute Throughput: 8.2 TFLOPS (21.6% of peak)
- Occupancy: 45.2%

NCU Diagnosis:
Primary Bottleneck: Memory Bandwidth (score: 0.82)
- Issue: Low bandwidth utilization (36.5%)
- Issue: Uncoalesced memory access detected
- Recommendation: Use shared memory tiling
- Recommendation: Increase block size to improve occupancy

Applicable Strategies:
1. matmul_shared_memory_tiling (confidence: 0.95)
   - Use 2D tiling with shared memory
   - Recommended tile size: 32x32
   - Expected speedup: 3-5x

2. vectorized_load_store (confidence: 0.75)
   - Use float4 for memory access
   - Expected speedup: 1.5-2x
```

## Example 2: Full Optimization Run

```bash
python main_real_gpu.py \
    --kernel-path ../CudaForge-main/KernelBench/level1/1_Square_matrix_multiplication_.py \
    --max-iterations 5 \
    --llm-backend deepseek \
    --output-dir ./results/matmul_opt
```

**Expected Output:**
```
=== KernelForge-Optimizer: Real GPU Testing ===
Kernel: 1_Square_matrix_multiplication_
GPU: NVIDIA GeForce RTX 5070
Max Iterations: 5

[Iteration 1/5]
Strategy: matmul_shared_memory_tiling
- Generating optimized code with LLM...
- Compiling kernel...
- Running NCU profiling...
Result: 0.82 ms (2.99x speedup) ✓

[Iteration 2/5]
Strategy: vectorized_load_store
- Generating optimized code with LLM...
- Compiling kernel...
- Running NCU profiling...
Result: 0.65 ms (3.77x speedup) ✓

[Iteration 3/5]
Strategy: increase_occupancy
- Generating optimized code with LLM...
- Compiling kernel...
- Running NCU profiling...
Result: 0.58 ms (4.22x speedup) ✓

[Iteration 4/5]
Trend: Improving (3 consecutive improvements)
Strategy: bank_conflict_elimination
- Generating optimized code with LLM...
- Compiling kernel...
- Running NCU profiling...
Result: 0.55 ms (4.45x speedup) ✓

[Iteration 5/5]
Trend: Stagnant (improvement < 5%)
Stopping: Converged to optimal solution

=== Final Results ===
Best Performance: 0.55 ms
Total Speedup: 4.45x
Iterations: 4
Successful Strategies:
  1. matmul_shared_memory_tiling: 2.99x
  2. vectorized_load_store: 1.26x (cumulative: 3.77x)
  3. increase_occupancy: 1.12x (cumulative: 4.22x)
  4. bank_conflict_elimination: 1.05x (cumulative: 4.45x)

Optimized kernel saved to: ./results/matmul_opt/optimized_kernel.cu
Performance report saved to: ./results/matmul_opt/report.json
```

## Example 3: Python API Usage

```python
from main_real_gpu import RealGPUOptimizer, OptimizationConfig

# Configure optimization
config = OptimizationConfig(
    kernel_path="../CudaForge-main/KernelBench/level1/1_Square_matrix_multiplication_.py",
    max_iterations=10,
    use_template_library=True,
    llm_backend="deepseek",
    output_dir="./results/my_optimization",
    verbose=True
)

# Run optimization
optimizer = RealGPUOptimizer(config)
results = optimizer.optimize()

# Access results
print(f"Baseline time: {results['baseline_time']:.3f} ms")
print(f"Optimized time: {results['best_time']:.3f} ms")
print(f"Speedup: {results['speedup']:.2f}x")
print(f"Improvement: {results['improvement_percent']:+.1f}%")
print(f"Iterations: {results['iterations']}")

# Get optimization history
for i, round_info in enumerate(results['history'], 1):
    print(f"\nRound {i}:")
    print(f"  Strategy: {round_info['strategy']}")
    print(f"  Time: {round_info['time']:.3f} ms")
    print(f"  Speedup: {round_info['speedup']:.2f}x")
    print(f"  Bottleneck: {round_info['bottleneck']}")
```

## Example 4: Testing Multiple Kernels

```python
import os
from pathlib import Path
from main_real_gpu import RealGPUOptimizer, OptimizationConfig

# Get all level1 kernels from KernelBench
kernelbench_dir = Path("../CudaForge-main/KernelBench/level1")
kernel_files = list(kernelbench_dir.glob("*.py"))

results_summary = []

for kernel_file in kernel_files:
    print(f"\n{'='*60}")
    print(f"Optimizing: {kernel_file.name}")
    print('='*60)
    
    config = OptimizationConfig(
        kernel_path=str(kernel_file),
        max_iterations=5,
        use_template_library=True,
        llm_backend="deepseek",
        output_dir=f"./results/{kernel_file.stem}",
        verbose=False
    )
    
    try:
        optimizer = RealGPUOptimizer(config)
        results = optimizer.optimize()
        
        results_summary.append({
            'kernel': kernel_file.name,
            'speedup': results['speedup'],
            'iterations': results['iterations'],
            'baseline_time': results['baseline_time'],
            'best_time': results['best_time']
        })
        
        print(f"✓ Success: {results['speedup']:.2f}x speedup")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        results_summary.append({
            'kernel': kernel_file.name,
            'speedup': 1.0,
            'error': str(e)
        })

# Print summary
print("\n" + "="*60)
print("OPTIMIZATION SUMMARY")
print("="*60)
for result in sorted(results_summary, key=lambda x: x.get('speedup', 0), reverse=True):
    if 'error' in result:
        print(f"{result['kernel']:40s} FAILED: {result['error']}")
    else:
        print(f"{result['kernel']:40s} {result['speedup']:6.2f}x  ({result['baseline_time']:.3f}ms → {result['best_time']:.3f}ms)")

avg_speedup = sum(r.get('speedup', 1.0) for r in results_summary) / len(results_summary)
print(f"\nAverage Speedup: {avg_speedup:.2f}x")
```

## Example 5: Custom Strategy Selection

```python
from main_real_gpu import RealGPUOptimizer, OptimizationConfig
from agents.strategy_templates import StrategyTemplateLibrary

# Create custom strategy library
strategy_lib = StrategyTemplateLibrary()

# Get all available strategies
all_strategies = strategy_lib.get_all_templates()
print("Available strategies:")
for name, template in all_strategies.items():
    print(f"  - {name}: {template.description}")

# Configure optimizer with specific strategies
config = OptimizationConfig(
    kernel_path="../CudaForge-main/KernelBench/level1/1_Square_matrix_multiplication_.py",
    max_iterations=3,
    use_template_library=True,
    llm_backend="deepseek",
    output_dir="./results/custom_strategy",
    
    # Force specific strategy order
    strategy_order=[
        "matmul_shared_memory_tiling",
        "vectorized_load_store",
        "bank_conflict_elimination"
    ]
)

optimizer = RealGPUOptimizer(config)
results = optimizer.optimize()
```

## Example 6: Demo Mode (No GPU Required)

```bash
# Run interactive demo
python demo_all.py
```

**Interactive Menu:**
```
=== KernelForge-Optimizer Demo ===

Choose a demo:
1. NCU Interpreter - See how raw metrics are analyzed
2. Strategy Library - Browse optimization templates
3. Optimization History - Track iteration progress
4. Full Workflow - Complete optimization simulation
5. Exit

Enter choice (1-5): 1

=== NCU Interpreter Demo ===

Simulating NCU metrics for matrix multiplication...

Raw Metrics:
- Memory Bandwidth: 245 GB/s
- Compute Throughput: 8.2 TFLOPS
- Occupancy: 45.2%
- L2 Cache Hit Rate: 12.3%
- Global Load Efficiency: 68.5%

Diagnosis:
Primary Bottleneck: Memory Bandwidth (score: 0.82)
Secondary Bottleneck: Occupancy (score: 0.65)

Issues Detected:
- Low bandwidth utilization (36.5% of peak)
- Uncoalesced memory access pattern
- Low occupancy limiting parallelism

Recommendations:
1. Use shared memory tiling to reduce global memory traffic
2. Increase block size to improve occupancy
3. Ensure coalesced memory access patterns

Press Enter to continue...
```

## Example 7: Integration with CudaForge

```python
# In your CudaForge workflow, replace the optimization loop:

from agents.ncu_interpreter import NCUInterpreter
from agents.strategy_templates import StrategyTemplateLibrary
from agents.optimization_history import OptimizationHistory
from prompts.enhanced_optimization import build_enhanced_optimization_prompt

# Initialize components
ncu_interpreter = NCUInterpreter(gpu_name="RTX 5070")
strategy_lib = StrategyTemplateLibrary()
opt_history = OptimizationHistory()

# In optimization loop:
for iteration in range(max_iterations):
    # 1. Profile with NCU
    ncu_metrics = run_ncu_profiling(kernel_code)
    
    # 2. Interpret metrics
    diagnosis = ncu_interpreter.interpret(ncu_metrics)
    
    # 3. Select strategy
    if iteration == 0:
        # First iteration: use template matching
        strategies = strategy_lib.get_applicable_strategies(
            operator_type=operator_type,
            bottleneck_type=diagnosis.primary_bottleneck,
            compute_capability=gpu_compute_capability
        )
        selected_strategy = strategies[0] if strategies else None
    else:
        # Later iterations: use history-based recommendation
        recommendation = opt_history.recommend_next_strategy(
            current_diagnosis=diagnosis
        )
        selected_strategy = recommendation['strategy']
    
    # 4. Build enhanced prompt
    prompt = build_enhanced_optimization_prompt(
        diagnosis=diagnosis,
        strategy=selected_strategy,
        current_code=kernel_code,
        operator_type=operator_type
    )
    
    # 5. Generate optimized code with LLM
    optimized_code = llm_generate(prompt)
    
    # 6. Test and profile
    new_metrics = compile_and_profile(optimized_code)
    
    # 7. Record in history
    opt_history.add_round(
        strategy=selected_strategy.name,
        diagnosis=diagnosis,
        metrics=new_metrics,
        code=optimized_code
    )
    
    # 8. Check convergence
    if opt_history.should_stop():
        break
```

## Tips for Best Results

1. **Start with template-based optimization** - Use `use_template_library=True` for first few iterations
2. **Monitor bottleneck shifts** - Check if bottleneck changes from memory to compute or vice versa
3. **Use appropriate iteration count** - 5-10 iterations usually sufficient, more may overfit
4. **Save optimization history** - Useful for understanding what worked and what didn't
5. **Test on representative inputs** - Make sure test inputs match production workload size
6. **Verify correctness** - Always check that optimized kernel produces correct results
7. **Profile in isolation** - Close other GPU applications during profiling for accurate metrics

## Troubleshooting

**Issue: NCU profiling fails**
```bash
# Check NCU installation
ncu --version

# Run with sudo if permission denied
sudo ncu --version

# Check GPU is accessible
nvidia-smi
```

**Issue: LLM API errors**
```bash
# Verify API key is set
echo $DEEPSEEK_API_KEY

# Test API connection
python -c "from agents.llm_client import test_connection; test_connection()"
```

**Issue: Compilation errors**
```bash
# Check CUDA toolkit
nvcc --version

# Verify compute capability matches GPU
python -c "import torch; print(torch.cuda.get_device_capability())"
```
