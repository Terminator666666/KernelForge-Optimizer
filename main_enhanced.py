"""
KernelForge-Optimizer Main Program

Enhanced version of CudaForge with three core improvements:
1. NCU Interpreter - Translates raw metrics to high-level diagnostics
2. Strategy Templates - Provides verified optimization patterns
3. Optimization History - Tracks iterations and recommends strategies

This program can run in two modes:
- Integrated mode: Enhances CudaForge optimization loop
- Standalone mode: Independent optimization tool for demos
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    create_interpreter_for_gpu,
    create_strategy_library,
    create_optimization_history,
    OperatorType,
    BottleneckType,
    OptimizationRound
)
from utils import (
    detect_operator_type,
    detect_gpu,
    get_gpu_specs
)
from prompts import (
    build_enhanced_judge_prompt,
    build_enhanced_optimization_prompt
)


class KernelForgeOptimizer:
    """Main optimizer class integrating all enhancements."""

    def __init__(self, gpu_name: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize optimizer.

        Args:
            gpu_name: GPU model name (auto-detect if None)
            config: Configuration dictionary
        """
        self.config = config or self._default_config()

        # Detect or set GPU
        if gpu_name is None:
            detected = detect_gpu()
            if detected:
                self.gpu_name, self.gpu_specs = detected
            else:
                raise RuntimeError("Could not detect GPU. Please specify gpu_name manually.")
        else:
            self.gpu_name = gpu_name
            self.gpu_specs = get_gpu_specs(gpu_name)
            if not self.gpu_specs:
                raise ValueError(f"Unknown GPU: {gpu_name}")

        # Initialize core components
        self.ncu_interpreter = create_interpreter_for_gpu(self.gpu_name)
        self.strategy_library = create_strategy_library()
        self.optimization_history = create_optimization_history()

        print(f"[KernelForge-Optimizer] Initialized for {self.gpu_name}")
        print(f"  Architecture: {self.gpu_specs['architecture']}")
        print(f"  Compute Capability: {self.gpu_specs['compute_capability']}")
        print(f"  SM Count: {self.gpu_specs['sm_count']}")

    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'max_iterations': 10,
            'min_speedup_threshold': 1.05,  # Stop if speedup < 5%
            'enable_history': True,
            'save_history': True,
            'history_file': 'optimization_history.json',
            'verbose': True
        }

    def optimize_kernel(self,
                       kernel_code: str,
                       kernel_name: str,
                       operator_type: Optional[str] = None,
                       ncu_metrics: Optional[Dict] = None,
                       test_function: Optional[callable] = None) -> Dict:
        """
        Optimize a CUDA kernel using enhanced workflow.

        Args:
            kernel_code: Initial kernel source code
            kernel_name: Kernel function name
            operator_type: Operator type (auto-detect if None)
            ncu_metrics: NCU profiling metrics (will profile if None)
            test_function: Function to test kernel correctness

        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*80}")
        print(f"Optimizing kernel: {kernel_name}")
        print(f"{'='*80}\n")

        # Step 1: Detect operator type
        if operator_type is None:
            op_type = detect_operator_type(kernel_code, filename=kernel_name)
            print(f"[Step 1] Detected operator type: {op_type.value}")
        else:
            op_type = OperatorType(operator_type)
            print(f"[Step 1] Using specified operator type: {op_type.value}")

        # Step 2: Initial profiling (if metrics not provided)
        if ncu_metrics is None:
            print(f"[Step 2] Profiling initial kernel with NCU...")
            ncu_metrics = self._profile_kernel(kernel_code, kernel_name)

        # Step 3: Interpret NCU metrics
        print(f"[Step 3] Interpreting performance metrics...")
        diagnosis = self.ncu_interpreter.interpret(ncu_metrics)
        print(f"  Primary bottleneck: {diagnosis.bottleneck} ({diagnosis.bottleneck_confidence:.0%} confidence)")
        print(f"  Bandwidth utilization: {diagnosis.memory_bandwidth_util:.1f}%")
        print(f"  Occupancy: {diagnosis.achieved_occupancy:.1f}%")
        print(f"  Arithmetic intensity: {diagnosis.arithmetic_intensity:.2f} FLOPs/byte")

        # Optimization loop
        current_code = kernel_code
        current_time_ms = ncu_metrics.get('duration', 0) * 1e-6  # Convert to ms
        best_code = current_code
        best_time_ms = current_time_ms

        for iteration in range(self.config['max_iterations']):
            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1}/{self.config['max_iterations']}")
            print(f"{'='*80}\n")

            # Step 4: Select optimization strategy
            print(f"[Step 4] Selecting optimization strategy...")
            strategy, reasoning = self._select_strategy(
                op_type,
                BottleneckType(diagnosis.bottleneck),
                diagnosis
            )
            print(f"  Selected: {strategy.name}")
            print(f"  Reasoning: {reasoning}")

            # Step 5: Generate optimized code
            print(f"[Step 5] Generating optimized kernel...")
            optimized_code = self._generate_optimized_code(
                current_code,
                op_type,
                diagnosis,
                strategy
            )

            # Step 6: Compile and test
            print(f"[Step 6] Compiling and testing...")
            compile_success, compile_error = self._compile_kernel(optimized_code, kernel_name)

            if not compile_success:
                print(f"  ✗ Compilation failed: {compile_error}")
                self._record_round(
                    iteration + 1, op_type, diagnosis, strategy, {},
                    current_time_ms, 1.0, False, False, compile_error
                )
                continue

            # Step 7: Profile optimized kernel
            print(f"[Step 7] Profiling optimized kernel...")
            new_metrics = self._profile_kernel(optimized_code, kernel_name)
            new_time_ms = new_metrics.get('duration', 0) * 1e-6

            # Calculate speedup
            speedup = current_time_ms / new_time_ms if new_time_ms > 0 else 0

            print(f"  Time: {current_time_ms:.3f} ms → {new_time_ms:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")

            # Test correctness if test function provided
            runtime_success = True
            error_message = None
            if test_function:
                try:
                    test_function(optimized_code)
                    print(f"  ✓ Correctness test passed")
                except Exception as e:
                    runtime_success = False
                    error_message = str(e)
                    print(f"  ✗ Correctness test failed: {error_message}")

            # Record this round
            self._record_round(
                iteration + 1, op_type, diagnosis, strategy, new_metrics,
                new_time_ms, speedup, compile_success, runtime_success, error_message
            )

            # Update best if improved
            if runtime_success and new_time_ms < best_time_ms:
                best_code = optimized_code
                best_time_ms = new_time_ms
                print(f"  ✓ New best: {best_time_ms:.3f} ms")

            # Check stopping criteria
            if speedup < self.config['min_speedup_threshold']:
                print(f"\n  Stopping: speedup below threshold ({speedup:.2f}x < {self.config['min_speedup_threshold']:.2f}x)")
                break

            # Update for next iteration
            current_code = optimized_code
            current_time_ms = new_time_ms
            diagnosis = self.ncu_interpreter.interpret(new_metrics)

        # Final summary
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*80}\n")

        summary = self.optimization_history.get_summary()
        total_speedup = summary['total_speedup']

        print(f"Total iterations: {summary['total_rounds']}")
        print(f"Successful iterations: {summary['successful_rounds']}")
        print(f"Best time: {best_time_ms:.3f} ms")
        print(f"Total speedup: {total_speedup:.2f}x")
        print(f"Strategies tried: {', '.join(summary['strategies_tried'])}")

        # Save history if enabled
        if self.config['save_history']:
            history_file = self.config['history_file']
            self.optimization_history.save_to_file(history_file)
            print(f"\nHistory saved to: {history_file}")

        return {
            'best_code': best_code,
            'best_time_ms': best_time_ms,
            'total_speedup': total_speedup,
            'iterations': summary['total_rounds'],
            'summary': summary
        }

    def _select_strategy(self,
                        op_type: OperatorType,
                        bottleneck: BottleneckType,
                        diagnosis) -> Tuple:
        """Select optimization strategy based on context."""

        # Get applicable strategies
        applicable = self.strategy_library.get_applicable_strategies(
            op_type,
            bottleneck,
            self.gpu_specs['compute_capability']
        )

        if not applicable:
            raise RuntimeError(f"No applicable strategies for {op_type.value} with {bottleneck.value} bottleneck")

        # Use history to recommend strategy
        if self.config['enable_history'] and len(self.optimization_history.rounds) > 0:
            strategy_names = [s.name for s in applicable]
            recommended_name, reasoning = self.optimization_history.recommend_next_strategy(
                bottleneck.value,
                strategy_names
            )

            # Find the strategy object
            for strategy in applicable:
                if strategy.name == recommended_name:
                    return (strategy, reasoning)

        # Default: use first applicable strategy
        return (applicable[0], "First applicable strategy for this bottleneck")

    def _generate_optimized_code(self,
                                current_code: str,
                                op_type: OperatorType,
                                diagnosis,
                                strategy) -> str:
        """Generate optimized code using LLM."""

        # Build enhanced prompt
        prompt = build_enhanced_optimization_prompt(
            diagnosis,
            strategy,
            op_type.value,
            current_code,
            self.gpu_specs
        )

        # Call LLM (placeholder - integrate with actual LLM service)
        # In practice, this would call OpenAI, DeepSeek, or other LLM API
        optimized_code = self._call_llm(prompt)

        return optimized_code

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM service to generate code.

        This is a placeholder. In practice, integrate with:
        - OpenAI API
        - DeepSeek API
        - Anthropic API
        - Local vLLM server
        """
        # TODO: Implement actual LLM integration
        raise NotImplementedError("LLM integration not implemented. Use CudaForge's query_server.py")

    def _profile_kernel(self, kernel_code: str, kernel_name: str) -> Dict[str, float]:
        """
        Profile kernel with NCU.

        This is a placeholder. In practice, integrate with:
        - CudaForge's run_ncu.py
        - Direct NCU command line
        """
        # TODO: Implement actual NCU profiling
        raise NotImplementedError("NCU profiling not implemented. Use CudaForge's run_ncu.py")

    def _compile_kernel(self, kernel_code: str, kernel_name: str) -> Tuple[bool, Optional[str]]:
        """
        Compile kernel and check for errors.

        Returns:
            Tuple of (success, error_message)
        """
        # TODO: Implement actual compilation
        raise NotImplementedError("Kernel compilation not implemented")

    def _record_round(self,
                     round_id: int,
                     op_type: OperatorType,
                     diagnosis,
                     strategy,
                     metrics: Dict,
                     time_ms: float,
                     speedup: float,
                     compile_success: bool,
                     runtime_success: bool,
                     error_message: Optional[str]):
        """Record optimization round in history."""

        round_data = OptimizationRound(
            round_id=round_id,
            timestamp=datetime.now().isoformat(),
            kernel_code="",  # Not saved to reduce memory
            operator_type=op_type.value,
            gpu_name=self.gpu_name,
            bottleneck=diagnosis.bottleneck,
            bottleneck_confidence=diagnosis.bottleneck_confidence,
            bandwidth_util=diagnosis.memory_bandwidth_util,
            occupancy=diagnosis.achieved_occupancy,
            arithmetic_intensity=diagnosis.arithmetic_intensity,
            strategy_name=strategy.name,
            strategy_params={},  # TODO: Extract actual parameters used
            execution_time_ms=time_ms,
            speedup=speedup,
            compilation_success=compile_success,
            runtime_success=runtime_success,
            error_message=error_message
        )

        self.optimization_history.add_round(round_data)


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description='KernelForge-Optimizer: Enhanced CUDA Kernel Optimization')
    parser.add_argument('--kernel', type=str, required=True, help='Path to kernel source file')
    parser.add_argument('--kernel-name', type=str, required=True, help='Kernel function name')
    parser.add_argument('--operator-type', type=str, help='Operator type (auto-detect if not specified)')
    parser.add_argument('--gpu', type=str, help='GPU model name (auto-detect if not specified)')
    parser.add_argument('--max-iterations', type=int, default=10, help='Maximum optimization iterations')
    parser.add_argument('--output', type=str, help='Output file for optimized kernel')
    parser.add_argument('--history', type=str, default='optimization_history.json', help='History file path')

    args = parser.parse_args()

    # Read kernel code
    with open(args.kernel, 'r') as f:
        kernel_code = f.read()

    # Create optimizer
    config = {
        'max_iterations': args.max_iterations,
        'history_file': args.history,
    }

    optimizer = KernelForgeOptimizer(gpu_name=args.gpu, config=config)

    # Run optimization
    result = optimizer.optimize_kernel(
        kernel_code,
        args.kernel_name,
        operator_type=args.operator_type
    )

    # Save optimized kernel if output specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result['best_code'])
        print(f"\nOptimized kernel saved to: {args.output}")

    # Print final report
    print("\n" + optimizer.optimization_history.generate_report())


if __name__ == '__main__':
    main()
