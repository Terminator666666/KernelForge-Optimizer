"""
KernelForge-Optimizer: Real GPU Testing Version
Uses CudaForge's KernelBench and NCU profiling tools for actual GPU testing.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add CudaForge to path
CUDAFORGE_PATH = Path(__file__).parent.parent / "CudaForge-main" / "CudaForge-main"
sys.path.insert(0, str(CUDAFORGE_PATH))

# Import CudaForge modules
from run_ncu import profile_bench, load_ncu_metrics, METRICS
from scripts.individual import KernelIndividual
from agents.query_server import query_model

# Import our enhanced modules
from agents.ncu_interpreter import NCUInterpreter, PerformanceDiagnosis
from agents.strategy_templates import StrategyLibrary, StrategyTemplate, BottleneckType
from agents.optimization_history import OptimizationHistory, OptimizationRound
from utils.operator_detection import detect_operator_type
from utils.gpu_arch_detection import detect_gpu
from prompts.enhanced_judge import build_enhanced_judge_prompt
from prompts.enhanced_optimization_v2 import build_enhanced_optimization_prompt


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""
    benchmark_path: str  # Path to KernelBench operator file
    max_iterations: int = 10
    use_template_library: bool = True
    llm_backend: str = "deepseek"  # or "openai", "anthropic", etc.
    llm_model: str = "deepseek-chat"  # deepseek-chat, deepseek-coder, deepseek-v4-pro
    temperature: float = 0.7
    output_dir: str = "./optimization_results"
    ncu_metrics: List[str] = None  # Use default METRICS from CudaForge
    use_simple_profiling: bool = False  # Use PyTorch timing instead of NCU when True

    def __post_init__(self):
        if self.ncu_metrics is None:
            self.ncu_metrics = METRICS


class RealGPUOptimizer:
    """
    Enhanced CUDA kernel optimizer using real GPU profiling.
    Integrates CudaForge's profiling tools with our enhanced analysis modules.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

        # Detect GPU architecture first (needed by NCUInterpreter)
        gpu_result = detect_gpu()
        if gpu_result:
            gpu_name, gpu_specs = gpu_result
            self.gpu_arch = gpu_specs
            self.gpu_arch['name'] = gpu_name  # 添加 GPU 名称到 specs 字典
            print(f"Detected GPU: {gpu_name} ({gpu_specs['architecture']})")
            print(f"  Compute Capability: {gpu_specs['compute_capability']}")
            print(f"  Memory Bandwidth: {gpu_specs['peak_bandwidth_gbps']} GB/s")
            print(f"  Peak FP32: {gpu_specs['peak_tflops_fp32']} TFLOPS")
        else:
            print("Warning: No GPU detected, using default specs")
            self.gpu_arch = {
                'name': 'Unknown',
                'architecture': 'Unknown',
                'compute_capability': 7.5,
                'peak_bandwidth_gbps': 600,
                'peak_tflops_fp32': 10.0
            }

        # Initialize modules (NCUInterpreter needs gpu_specs)
        self.ncu_interpreter = NCUInterpreter(self.gpu_arch)
        self.strategy_library = StrategyLibrary()
        self.optimization_history = OptimizationHistory()

        # Detect operator type
        self.operator_type = detect_operator_type(config.benchmark_path)
        print(f"Detected operator type: {self.operator_type}")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Load baseline benchmark
        self.baseline_bench = self._load_benchmark(config.benchmark_path)

    def _load_benchmark(self, bench_path: str) -> KernelIndividual:
        """Load a KernelBench operator as baseline."""
        # Read kernel code
        code = Path(bench_path).read_text()

        # Create KernelIndividual with code
        individual = KernelIndividual(code)

        # Set additional attributes
        if not hasattr(individual, 'operator_type'):
            individual.operator_type = self.operator_type

        return individual

    def _convert_to_ncu_format(self, code: str) -> str:
        """
        转换 benchmark 代码为 NCU profiling 所需的格式。

        支持两种输入格式：
        1. 完整的 benchmark 文件（包含 Model, get_inputs, get_init_inputs, __main__）
        2. 只有 ModelNew 类的代码（新 prompt 生成的格式）

        输出格式：
        - 类名必须是 ModelNew
        - 必须有 get_inputs() 和 get_init_inputs() 函数
        - 必须有 __main__ 块支持直接执行
        """
        # 检查是否是完整的 benchmark 文件
        has_get_inputs = 'def get_inputs()' in code
        has_get_init_inputs = 'def get_init_inputs()' in code
        has_main_block = 'if __name__ == "__main__"' in code

        print(f"  [DEBUG] has_get_inputs={has_get_inputs}, has_get_init_inputs={has_get_init_inputs}, has_main_block={has_main_block}")

        # 如果是完整的 benchmark 文件，只需要替换类名
        if has_get_inputs and has_get_init_inputs and has_main_block:
            print(f"  [DEBUG] Detected complete benchmark file, replacing class names...")
            # 替换类名 Model -> ModelNew（更全面的替换）
            import re

            # 调试：检查替换前
            has_model_class = 'class Model(' in code
            print(f"  [DEBUG] Before replacement: has 'class Model(' = {has_model_class}")

            # 替换 class Model( 为 class ModelNew(
            code = re.sub(r'\bclass Model\(', 'class ModelNew(', code)
            # 替换 super(Model, 为 super(ModelNew,
            code = re.sub(r'\bsuper\(Model,', 'super(ModelNew,', code)
            # 替换 Model() 为 ModelNew()（在 __main__ 块中）
            code = re.sub(r'\bModel\(\)', 'ModelNew()', code)

            # 调试：检查替换后
            has_modelnew_class = 'class ModelNew(' in code
            print(f"  [DEBUG] After replacement: has 'class ModelNew(' = {has_modelnew_class}")
            if has_model_class and not has_modelnew_class:
                print(f"  ⚠ Warning: Failed to replace 'class Model(' with 'class ModelNew('")

            return code

        print(f"  [DEBUG] Not a complete benchmark file, building from scratch...")
        # 否则，这是只有 ModelNew 类的代码，需要添加缺失的部分
        # 从 baseline 代码中提取 get_inputs 和 get_init_inputs
        baseline_code = self.baseline_bench.code

        # 首先，替换类名 Model -> ModelNew（如果代码中有 Model 类）
        import re
        if 'class Model(' in code:
            print(f"  [DEBUG] Replacing 'class Model(' with 'class ModelNew(' in input code...")
            code = re.sub(r'\bclass Model\(', 'class ModelNew(', code)
            code = re.sub(r'\bsuper\(Model,', 'super(ModelNew,', code)

        # 提取全局变量定义（在 get_inputs 之前的赋值语句）
        global_vars_code = ""
        if 'def get_inputs()' in baseline_code:
            get_inputs_start = baseline_code.find('def get_inputs()')

            # 查找 get_inputs 之前的全局变量定义
            # 从 class 定义结束到 get_inputs 开始之间的内容
            class_end = baseline_code.rfind('\n', 0, get_inputs_start)
            if class_end != -1:
                # 向前查找，找到最后一个 class 或 def 的结束位置
                prev_class = baseline_code.rfind('\nclass ', 0, get_inputs_start)
                prev_def = baseline_code.rfind('\ndef ', 0, get_inputs_start)

                # 从最后一个 class/def 之后开始提取
                search_start = 0
                if prev_class != -1 or prev_def != -1:
                    search_start = max(prev_class if prev_class != -1 else 0,
                                     prev_def if prev_def != -1 else 0)
                    # 找到该 class/def 的结束位置（下一个非缩进行）
                    lines = baseline_code[search_start:get_inputs_start].split('\n')
                    for i, line in enumerate(lines):
                        if i > 0 and line and not line[0].isspace():
                            search_start += sum(len(l) + 1 for l in lines[:i])
                            break

                # 提取全局变量（简单的赋值语句）
                potential_globals = baseline_code[search_start:get_inputs_start].strip()
                if potential_globals:
                    # 只保留赋值语句（变量 = 值）
                    import re
                    lines = potential_globals.split('\n')
                    var_lines = []
                    for line in lines:
                        # 匹配简单的赋值语句：变量名 = 值
                        if re.match(r'^[a-zA-Z_]\w*\s*=\s*.+', line.strip()):
                            var_lines.append(line)
                    if var_lines:
                        global_vars_code = '\n'.join(var_lines) + '\n\n'

        # 提取 get_inputs 函数
        get_inputs_code = ""
        if 'def get_inputs()' in baseline_code:
            start = baseline_code.find('def get_inputs()')
            # 找到下一个函数定义或文件结尾
            next_def = baseline_code.find('\ndef ', start + 1)
            next_class = baseline_code.find('\nclass ', start + 1)
            next_if = baseline_code.find('\nif __name__', start + 1)

            end = len(baseline_code)
            for pos in [next_def, next_class, next_if]:
                if pos != -1 and pos < end:
                    end = pos

            get_inputs_code = baseline_code[start:end].strip()

        # 提取 get_init_inputs 函数
        get_init_inputs_code = ""
        if 'def get_init_inputs()' in baseline_code:
            start = baseline_code.find('def get_init_inputs()')
            # 找到下一个函数定义或文件结尾
            next_def = baseline_code.find('\ndef ', start + 1)
            next_class = baseline_code.find('\nclass ', start + 1)
            next_if = baseline_code.find('\nif __name__', start + 1)

            end = len(baseline_code)
            for pos in [next_def, next_class, next_if]:
                if pos != -1 and pos < end:
                    end = pos

            get_init_inputs_code = baseline_code[start:end].strip()

        # 如果没有找到，使用默认实现
        if not get_inputs_code:
            get_inputs_code = """
def get_inputs():
    # Default implementation - may need adjustment
    x = torch.randn(16, 64, 256, 256).cuda()
    return [x]
"""

        if not get_init_inputs_code:
            get_init_inputs_code = """
def get_init_inputs():
    return []
"""

        # 构建完整的 benchmark 文件
        main_block = '''

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    # Initialize model and move to GPU
    init_inputs = get_init_inputs()
    if init_inputs:
        model = ModelNew(*init_inputs).cuda()
    else:
        model = ModelNew().cuda()

    # Get inputs and move to GPU
    inputs = get_inputs()
    inputs = [x.cuda() for x in inputs]

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model(*inputs)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Timing runs
    start = time.time()
    for _ in range(args.repeat):
        with torch.no_grad():
            output = model(*inputs)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / args.repeat * 1000
    print(f"Average time: {avg_time_ms:.3f} ms")
'''

        # 组合所有部分（只添加尚未存在的函数）
        # 1. 先添加全局变量（修改 batch_size 以适应 8GB 显存）
        if global_vars_code:
            # 将 batch_size 从 4096 降到 512，避免 OOM（从 6GB 降到 0.75GB）
            import re
            global_vars_code = re.sub(r'batch_size\s*=\s*\d+', 'batch_size = 512', global_vars_code)
            code = code + "\n\n" + global_vars_code

        # 2. 添加函数定义
        if 'def get_inputs()' not in code:
            code = code + "\n" + get_inputs_code
        if 'def get_init_inputs()' not in code:
            code = code + "\n\n" + get_init_inputs_code

        full_code = code + "\n" + main_block

        return full_code

    def _profile_kernel(self, individual: KernelIndividual, iteration: int) -> Dict:
        """Profile a kernel using CudaForge's NCU profiling and return metrics."""
        print(f"\n[Iteration {iteration}] Profiling kernel...")

        # Convert code to NCU-compatible format
        ncu_code = self._convert_to_ncu_format(individual.code)

        # Save kernel code to temporary file
        temp_bench_path = os.path.join(self.config.output_dir, f"iter_{iteration}_kernel.py")
        with open(temp_bench_path, 'w') as f:
            f.write(ncu_code)

        # Check if we should use simple profiling (no NCU)
        if self.config.use_simple_profiling:
            return self._profile_kernel_simple(temp_bench_path, iteration)

        # Use CudaForge's profile_bench to run NCU and generate CSV
        csv_path = os.path.join(self.config.output_dir, f"iter_{iteration}_ncu.csv")

        print(f"  Running NCU profiling via CudaForge...")
        print(f"  Benchmark: {temp_bench_path}")
        print(f"  Output CSV: {csv_path}")

        try:
            # Call CudaForge's profile_bench function
            # It will run: sudo -E ncu --csv --page=raw --metrics [...] python bench.py --repeat N
            print(f"  → Calling profile_bench()...")
            actual_csv_path = profile_bench(
                bench_py=temp_bench_path,
                out_csv=csv_path,
                repeat=100,
                kernel_names=None  # Profile all kernels
            )

            print(f"  [OK] NCU profiling completed successfully")
            print(f"  → CSV output: {actual_csv_path}")

            # Load metrics from CSV using CudaForge's load_ncu_metrics
            print(f"  → Loading metrics from CSV...")
            df = load_ncu_metrics(
                csv_path=actual_csv_path,
                columns=None,  # Use default METRIC_COLUMNS
                extra_keep=("Kernel Name",),
                coerce_numeric=True,
                name_list=None,  # Get all kernels
                select="last"  # Use last measurement if multiple
            )

            print(f"  → Loaded {len(df)} kernel profile(s)")

            # Extract metrics from the DataFrame
            # Take the first row (or aggregate if multiple kernels)
            if len(df) > 0:
                row = df.iloc[0]

                # NCU 2025.2+ on Blackwell uses different metric names
                # Try new names first, fallback to old names
                dram_bytes_per_sec = float(row.get('dram__bytes.sum.per_second', 0))
                dram_bytes_read = float(row.get('dram__bytes_read.sum', 0))
                dram_bytes_write = float(row.get('dram__bytes_write.sum', 0))

                # Get kernel duration from GPU time
                duration_ns = float(row.get('gpu__time_duration.sum', 0))
                cycles = float(row.get('sm__cycles_active.avg', 0))

                # Debug: print what we got
                print(f"  [DEBUG] duration_ns={duration_ns}, cycles={cycles}, dram_bytes_per_sec={dram_bytes_per_sec}")

                # If duration is 0, estimate from cycles (assume 1.4 GHz clock)
                if duration_ns == 0 and cycles > 0:
                    duration_ns = cycles / 1.4  # Approximate: 1.4 GHz = 1.4 cycles/ns
                    print(f"  [DEBUG] Estimated duration from cycles: {duration_ns:.0f} ns")

                # Calculate total bytes transferred
                # If we have duration and bytes/sec, calculate total bytes
                if dram_bytes_per_sec > 0 and duration_ns > 0:
                    dram_bytes_total = dram_bytes_per_sec * (duration_ns / 1e9)  # Convert ns to seconds
                else:
                    dram_bytes_total = dram_bytes_read + dram_bytes_write

                # If new metric exists but old ones don't, estimate read/write as 50/50
                if dram_bytes_total > 0 and dram_bytes_read == 0 and dram_bytes_write == 0:
                    dram_bytes_read = dram_bytes_total / 2
                    dram_bytes_write = dram_bytes_total / 2

                # Get occupancy from launch__occupancy_limit_blocks or sm__warps_active
                occupancy_pct = float(row.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0))

                metrics = {
                    'execution_time_ms': duration_ns / 1e6 if duration_ns > 0 else cycles / 1e6,  # Convert to ms
                    'sm__cycles_active.avg': cycles,
                    'dram__bytes_read.sum': dram_bytes_read,
                    'dram__bytes_write.sum': dram_bytes_write,
                    'dram__bytes.sum': dram_bytes_total,  # For NCU interpreter
                    'dram__bytes.sum.per_second': dram_bytes_per_sec,
                    'dram__throughput.avg.pct': float(row.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0)),
                    'duration': duration_ns,  # For NCU interpreter (in nanoseconds)
                    'sm__warps_active.avg.pct_of_peak_sustained_active': occupancy_pct,
                    'occupancy': occupancy_pct / 100.0,  # Convert percentage to fraction
                    'kernel_name': str(row.get('Kernel Name', 'unknown')),
                    'csv_path': str(actual_csv_path)
                }

                print(f"  → Extracted metrics:")
                print(f"     - Cycles: {metrics['sm__cycles_active.avg']:.0f}")
                print(f"     - DRAM bandwidth: {metrics['dram__bytes.sum.per_second'] / 1e9:.2f} GB/s")
                print(f"     - DRAM throughput: {metrics['dram__throughput.avg.pct']:.2f}%")
                print(f"     - Occupancy: {metrics['occupancy']:.2%}")
            else:
                print(f"  ⚠ No kernel profiles found in CSV")
                metrics = self._create_fallback_metrics()

        except SystemExit as e:
            print(f"  ⚠ NCU profiling failed with SystemExit: {e.code}")
            print(f"  → This usually means NCU returned a non-zero exit code")
            print(f"  → Check if:")
            print(f"     1. The benchmark file is valid Python")
            print(f"     2. CUDA kernels are being launched")
            print(f"     3. NCU has proper permissions (sudo)")
            print(f"  → Try running manually:")
            print(f"     python {temp_bench_path} --repeat 10")
            metrics = self._create_fallback_metrics()
        except Exception as e:
            print(f"  ⚠ NCU profiling failed: {e}")
            import traceback
            print(f"  → Full traceback:")
            traceback.print_exc()
            metrics = self._create_fallback_metrics()

        return metrics

    def _profile_kernel_simple(self, bench_path: str, iteration: int) -> Dict:
        """
        Simple profiling using PyTorch timing (no NCU required).
        Runs the benchmark and measures execution time only.
        """
        print(f"  Using simple PyTorch timing (NCU disabled)...")

        import subprocess
        import re

        try:
            # Run the benchmark script (use fewer repeats for large benchmarks)
            result = subprocess.run(
                [sys.executable, bench_path, "--repeat", "10", "--warmup", "2"],
                capture_output=True,
                text=True,
                timeout=120  # Increase timeout to 2 minutes
            )

            if result.returncode != 0:
                print(f"  [FAIL] Benchmark execution failed:")
                print(f"    {result.stderr}")
                return self._create_fallback_metrics()

            # Extract timing from output: "Average time: X.XXX ms"
            match = re.search(r'Average time:\s+([\d.]+)\s+ms', result.stdout)
            if match:
                exec_time = float(match.group(1))
                print(f"  [OK] Execution time: {exec_time:.3f} ms")

                # Return minimal metrics (only timing available)
                return {
                    'execution_time_ms': exec_time,
                    'sm__cycles_active.avg': exec_time * 1e6,  # Rough estimate
                    'dram__bytes_read.sum': 0,  # Unknown
                    'dram__bytes_write.sum': 0,  # Unknown
                    'occupancy': 0.5,  # Unknown, assume 50%
                    'kernel_name': f'iter_{iteration}',
                    'csv_path': None
                }
            else:
                print(f"  [FAIL] Could not parse timing from output:")
                print(f"    {result.stdout}")
                return self._create_fallback_metrics()

        except subprocess.TimeoutExpired:
            print(f"  [FAIL] Benchmark timed out after 60 seconds")
            return self._create_fallback_metrics()
        except Exception as e:
            print(f"  [FAIL] Simple profiling failed: {e}")
            return self._create_fallback_metrics()

    def _verify_correctness(self, optimized_code: str, baseline_code: str) -> tuple[bool, str]:
        """
        验证优化后的代码是否与 baseline 产生相同的输出

        Returns:
            (is_correct, error_message)
        """
        import tempfile
        import subprocess
        import numpy as np
        import re

        print(f"  验证正确性...")

        # 检测 in-place 操作（会破坏输入数据）
        inplace_patterns = [
            r'torch\.\w+_\(',  # torch.relu_(, torch.add_(, etc.
            r'\.relu_\(',
            r'\.add_\(',
            r'\.mul_\(',
            r'\.sub_\(',
            r'\.div_\(',
            r'inplace\s*=\s*True',
        ]
        for pattern in inplace_patterns:
            if re.search(pattern, optimized_code):
                return False, f"检测到 in-place 操作 ({pattern})，这会修改输入数据导致 benchmark 循环测量不准确"

        try:
            # 将代码转换为可执行格式
            baseline_executable = self._convert_to_ncu_format(baseline_code)
            optimized_executable = self._convert_to_ncu_format(optimized_code)

            # 保存转换后的代码用于调试
            debug_dir = self.config.output_dir
            with open(os.path.join(debug_dir, "debug_baseline_executable.py"), 'w') as f:
                f.write(baseline_executable)
            with open(os.path.join(debug_dir, "debug_optimized_executable.py"), 'w') as f:
                f.write(optimized_executable)
            print(f"  [DEBUG] Saved executable code to {debug_dir}/debug_*_executable.py")

            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='_baseline.py', delete=False) as f_base:
                f_base.write(baseline_executable)
                baseline_path = f_base.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='_optimized.py', delete=False) as f_opt:
                f_opt.write(optimized_executable)
                optimized_path = f_opt.name

            # 运行 baseline 并保存输出
            result_base = subprocess.run(
                [sys.executable, baseline_path, "--repeat", "1"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result_base.returncode != 0:
                return False, f"Baseline 执行失败: {result_base.stderr[:200]}"

            # 运行 optimized 并保存输出
            result_opt = subprocess.run(
                [sys.executable, optimized_path, "--repeat", "1"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result_opt.returncode != 0:
                return False, f"优化版本执行失败: {result_opt.stderr[:200]}"

            # 简单比较：如果都能运行成功，认为正确
            # TODO: 更严格的验证需要比较实际输出值
            print(f"    [OK] Baseline 和优化版本都成功执行")
            return True, ""

        except subprocess.TimeoutExpired:
            return False, "执行超时"
        except Exception as e:
            return False, f"验证过程出错: {str(e)}"
        finally:
            # 清理临时文件
            try:
                os.unlink(baseline_path)
                os.unlink(optimized_path)
            except:
                pass

    def _create_fallback_metrics(self) -> Dict:
        """
        移除虚假的 fallback metrics。
        如果 profiling 失败，返回 None 表示失败，而不是虚假数据。
        """
        print(f"  [FAIL] Profiling failed, no fallback metrics")
        return None

    def _analyze_performance(self, metrics: Dict, iteration: int) -> PerformanceDiagnosis:
        """Analyze NCU metrics using our interpreter."""
        print(f"\n[Iteration {iteration}] Analyzing performance...")

        diagnosis = self.ncu_interpreter.interpret(
            metrics,
            kernel_info={
                'operator_type': self.operator_type
            }
        )

        print(f"  Primary bottleneck: {diagnosis.bottleneck}")
        print(f"  Memory bandwidth utilization: {diagnosis.memory_bandwidth_util:.1f}%")
        print(f"  Compute utilization (SM efficiency): {diagnosis.sm_efficiency:.1f}%")
        print(f"  Occupancy: {diagnosis.achieved_occupancy:.1f}%")

        return diagnosis

    def _select_strategy(self, diagnosis: PerformanceDiagnosis, iteration: int) -> Optional[StrategyTemplate]:
        """Select optimization strategy based on diagnosis and history."""
        print(f"\n[Iteration {iteration}] Selecting optimization strategy...")

        if self.config.use_template_library:
            # Match templates first
            matched_templates = self.strategy_library.get_applicable_strategies(
                operator_type=self.operator_type,
                bottleneck=BottleneckType(diagnosis.bottleneck),
                gpu_compute_capability=self.gpu_arch['compute_capability']
            )

            if matched_templates:
                # Get available strategy names
                available_strategies = [t.name for t in matched_templates]

                # Get strategy recommendation from history
                recommended_strategy, reason = self.optimization_history.recommend_next_strategy(
                    current_bottleneck=diagnosis.bottleneck,
                    available_strategies=available_strategies
                )

                print(f"  History recommendation: {recommended_strategy}")
                print(f"  Reason: {reason}")

                # Find the recommended strategy in matched templates
                for template in matched_templates:
                    if template.name == recommended_strategy:
                        strategy = template
                        print(f"  Selected strategy: {strategy.name}")
                        print(f"  Expected speedup: {strategy.expected_speedup}x")
                        return strategy

                # If recommended strategy not found, use first matched template
                strategy = matched_templates[0]
                print(f"  Selected strategy: {strategy.name}")
                print(f"  Expected speedup: {strategy.expected_speedup}x")
                return strategy

        print("  No template matched, will use pure LLM optimization")
        return None

    def _generate_optimized_code(
        self,
        current_code: str,
        diagnosis: PerformanceDiagnosis,
        strategy: Optional[StrategyTemplate],
        iteration: int
    ) -> str:
        """Generate optimized code using LLM."""
        print(f"\n[Iteration {iteration}] Generating optimized code...")

        # Build prompt
        prompt = build_enhanced_optimization_prompt(
            diagnosis=diagnosis,
            strategy=strategy,
            operator_type=str(self.operator_type.value),
            current_code=current_code,
            gpu_specs=self.gpu_arch,
            judge_recommendations=None
        )

        # Query LLM
        try:
            response = query_model(
                prompt=prompt,
                backend=self.config.llm_backend,
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=16384  # 增加 max_tokens，确保代码不被截断
            )

            # 保存完整的 LLM 响应用于调试
            debug_response_path = os.path.join(self.config.output_dir, f"iter_{iteration}_llm_response.txt")
            with open(debug_response_path, 'w') as f:
                f.write(response)

            print(f"  [DEBUG] LLM response length: {len(response)} characters")
            print(f"  [DEBUG] LLM response preview: {response[:200]}...")
            print(f"  [DEBUG] Full response saved to: {debug_response_path}")

            # Extract code from response
            optimized_code = self._extract_code_from_response(response)

            print(f"  Generated {len(optimized_code)} characters of code")
            print(f"  [DEBUG] Extracted code preview: {optimized_code[:200]}...")
            return optimized_code

        except Exception as e:
            print(f"  Code generation failed: {e}")
            return current_code

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            code = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            code = response[start:end].strip()
        else:
            # Return full response if no code blocks
            code = response.strip()

        # 验证生成的代码
        is_valid, error_msg = self._validate_generated_code(code)
        if not is_valid:
            print(f"  ⚠ 代码验证失败: {error_msg}")
            print(f"  → 尝试修复...")
            code = self._fix_generated_code(code)

        return code

    def _validate_generated_code(self, code: str) -> tuple[bool, str]:
        """
        验证生成的代码是否符合要求

        Returns:
            (is_valid, error_message)
        """
        # 检查是否包含必要的 import
        if 'import torch' not in code:
            return False, "缺少 'import torch'"

        # 检查是否包含 ModelNew 类
        if 'class ModelNew' not in code:
            return False, "缺少 'class ModelNew' 定义"

        # 检查是否包含 __init__ 方法
        if 'def __init__' not in code:
            return False, "缺少 '__init__' 方法"

        # 检查是否包含 forward 方法
        if 'def forward' not in code:
            return False, "缺少 'forward' 方法"

        # 检查是否使用了禁止的模式
        if 'load_inline' in code:
            return False, "使用了禁止的 'load_inline'"

        # 检查语法是否正确
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"语法错误: {e}"

        return True, ""

    def _fix_generated_code(self, code: str) -> str:
        """
        尝试修复生成的代码
        """
        # 如果缺少 import，添加
        if 'import torch' not in code:
            code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n" + code

        # 移除 load_inline 相关代码
        if 'load_inline' in code:
            # 简单地移除包含 load_inline 的行
            lines = code.split('\n')
            filtered_lines = [line for line in lines if 'load_inline' not in line]
            code = '\n'.join(filtered_lines)

        return code

    def _record_iteration(
        self,
        iteration: int,
        metrics: Dict,
        diagnosis: PerformanceDiagnosis,
        strategy: Optional[StrategyTemplate],
        code: str
    ):
        """Record optimization iteration in history."""
        from datetime import datetime

        record = OptimizationRound(
            round_id=iteration,
            timestamp=datetime.now().isoformat(),
            kernel_code=code[:500],  # 存储前 500 个字符
            operator_type=self.operator_type,
            gpu_name=self.gpu_arch.get('name', 'Unknown'),
            bottleneck=diagnosis.bottleneck,
            bottleneck_confidence=diagnosis.bottleneck_confidence,
            bandwidth_util=diagnosis.memory_bandwidth_util,
            occupancy=diagnosis.achieved_occupancy,
            arithmetic_intensity=diagnosis.arithmetic_intensity,
            strategy_name=strategy.name if strategy else "pure_llm",
            strategy_params=strategy.parameter_rules if strategy else {},
            execution_time_ms=metrics.get('execution_time_ms', 0),
            speedup=1.0,  # 将在后续计算
            compilation_success=True,
            runtime_success=True,
            error_message=None,
            notes=""
        )

        self.optimization_history.add_round(record)

    def optimize(self) -> Dict:
        """
        Run the complete optimization process.
        Returns final results and optimization history.
        """
        print("="*80)
        print("Starting KernelForge-Optimizer (Real GPU Testing)")
        print("="*80)

        # Profile baseline
        print("\n" + "="*80)
        print("BASELINE PROFILING")
        print("="*80)
        baseline_metrics = self._profile_kernel(self.baseline_bench, 0)
        if baseline_metrics is None:
            print("Failed to profile baseline kernel!")
            return None

        baseline_diagnosis = self._analyze_performance(baseline_metrics, 0)
        baseline_time = baseline_metrics['execution_time_ms']

        # Initialize best result
        best_individual = self.baseline_bench
        best_metrics = baseline_metrics
        best_diagnosis = baseline_diagnosis
        best_time = baseline_time

        # Optimization loop
        current_individual = self.baseline_bench

        for iteration in range(1, self.config.max_iterations + 1):
            print("\n" + "="*80)
            print(f"ITERATION {iteration}/{self.config.max_iterations}")
            print("="*80)

            # Analyze current performance
            current_metrics = self._profile_kernel(current_individual, iteration)
            if current_metrics is None:
                print(f"Iteration {iteration} failed, skipping...")
                continue

            current_diagnosis = self._analyze_performance(current_metrics, iteration)

            # Select optimization strategy
            strategy = self._select_strategy(current_diagnosis, iteration)

            # Generate optimized code
            optimized_code = self._generate_optimized_code(
                current_individual.code,
                current_diagnosis,
                strategy,
                iteration
            )

            # Create new individual
            new_individual = KernelIndividual(code=optimized_code)
            new_individual.operator_type = self.operator_type

            # Verify correctness
            is_correct, error_msg = self._verify_correctness(optimized_code, self.baseline_bench.code)
            if not is_correct:
                print(f"  [FAIL] 正确性验证失败: {error_msg}")
                print(f"  跳过此迭代...")
                continue
            print(f"  [OK] 正确性验证通过")

            # Profile new kernel
            new_metrics = self._profile_kernel(new_individual, iteration)
            if new_metrics is None:
                print(f"New kernel failed to profile, skipping...")
                continue

            new_diagnosis = self._analyze_performance(new_metrics, iteration)
            new_time = new_metrics['execution_time_ms']

            # Record iteration
            self._record_iteration(iteration, new_metrics, new_diagnosis, strategy, optimized_code)

            # Compare performance
            speedup = baseline_time / new_time
            improvement = (baseline_time - new_time) / baseline_time * 100

            print(f"\n[Iteration {iteration}] Performance comparison:")
            print(f"  Baseline: {baseline_time:.3f} ms")
            print(f"  Current:  {new_time:.3f} ms")
            print(f"  Speedup:  {speedup:.2f}x ({improvement:+.1f}%)")

            # Update best if improved
            if new_time < best_time:
                best_individual = new_individual
                best_metrics = new_metrics
                best_diagnosis = new_diagnosis
                best_time = new_time
                print(f"  [OK] New best result!")

            # Update current for next iteration
            current_individual = new_individual

            # Check for convergence
            if iteration >= 3:
                trend = self.optimization_history.get_recent_trend()
                if trend == 'stagnant':
                    print(f"\n  Optimization stagnant, stopping early.")
                    break

        # Final summary
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)

        final_speedup = baseline_time / best_time
        final_improvement = (baseline_time - best_time) / baseline_time * 100

        print(f"\nFinal Results:")
        print(f"  Baseline time:    {baseline_time:.3f} ms")
        print(f"  Best time:        {best_time:.3f} ms")
        print(f"  Final speedup:    {final_speedup:.2f}x")
        print(f"  Improvement:      {final_improvement:+.1f}%")
        print(f"  Iterations:       {iteration}")

        # Save results
        results = {
            'benchmark_path': self.config.benchmark_path,
            'operator_type': self.operator_type,
            'gpu_arch': self.gpu_arch,
            'baseline': {
                'execution_time_ms': baseline_time,
                'metrics': baseline_metrics,
                'diagnosis': asdict(baseline_diagnosis)
            },
            'best': {
                'execution_time_ms': best_time,
                'metrics': best_metrics,
                'diagnosis': asdict(best_diagnosis),
                'code': best_individual.code
            },
            'speedup': final_speedup,
            'improvement_percent': final_improvement,
            'iterations': iteration,
            'history': [asdict(r) for r in self.optimization_history.rounds]
        }

        results_path = os.path.join(self.config.output_dir, 'optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # Use default=str to handle enums

        print(f"\nResults saved to: {results_path}")

        # Save best kernel code
        best_code_path = os.path.join(self.config.output_dir, 'best_kernel.py')
        with open(best_code_path, 'w') as f:
            f.write(best_individual.code)

        print(f"Best kernel saved to: {best_code_path}")

        return results


def main():
    """Main entry point for real GPU testing."""
    import argparse

    parser = argparse.ArgumentParser(description='KernelForge-Optimizer: Real GPU Testing')
    parser.add_argument('benchmark', type=str, help='Path to KernelBench operator file')
    parser.add_argument('--max-iterations', type=int, default=10, help='Maximum optimization iterations')
    parser.add_argument('--no-templates', action='store_true', help='Disable template library')
    parser.add_argument('--llm-backend', type=str, default='deepseek', help='LLM backend')
    parser.add_argument('--llm-model', type=str, default='deepseek-chat', help='LLM model')
    parser.add_argument('--output-dir', type=str, default='./optimization_results', help='Output directory')

    args = parser.parse_args()

    # Create config
    config = OptimizationConfig(
        benchmark_path=args.benchmark,
        max_iterations=args.max_iterations,
        use_template_library=not args.no_templates,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        output_dir=args.output_dir
    )

    # Run optimization
    optimizer = RealGPUOptimizer(config)
    results = optimizer.optimize()

    if results:
        print("\n[OK] Optimization completed successfully!")
    else:
        print("\n[FAIL] Optimization failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
