#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KernelForge 优化演示 - 完整的端到端优化流程演示

功能：
1. 代码理解和分析
2. 模拟 NCU profiling
3. 瓶颈诊断
4. 策略推荐
5. 代码生成
6. 多轮迭代
7. 生成完整报告
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ncu_interpreter import NCUInterpreter, create_interpreter_for_gpu
from agents.strategy_templates import StrategyLibrary, OperatorType, BottleneckType, create_strategy_library
from agents.optimization_history import OptimizationHistory, OptimizationRound


class KernelOptimizationDemo:
    """Kernel 优化演示 - 展示完整的优化流程"""

    def __init__(self, kernel_name: str, operator_type: str, gpu_name: str = "RTX 5070"):
        self.kernel_name = kernel_name
        self.operator_type = operator_type
        self.gpu_name = gpu_name

        # 初始化组件
        self.interpreter = create_interpreter_for_gpu(gpu_name)
        self.strategy_library = create_strategy_library()
        self.history = OptimizationHistory()

        # 创建工作目录
        self.workspace = Path("demo-optimization-results") / operator_type / kernel_name
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run_optimization_demo(self, max_rounds: int = 3):
        """运行优化演示"""
        print("=" * 80)
        print(f"🚀 KernelForge 优化演示")
        print("=" * 80)
        print(f"Kernel: {self.kernel_name}")
        print(f"算子类型: {self.operator_type}")
        print(f"GPU: {self.gpu_name}")
        print(f"最大轮次: {max_rounds}")
        print("=" * 80)

        # Round 0: Baseline
        print("\n" + "=" * 80)
        print("Round 0: Baseline")
        print("=" * 80)

        baseline_metrics = self._get_baseline_metrics()
        baseline_diagnosis = self.interpreter.interpret(baseline_metrics)

        print("\n[NCU Metrics]")
        print(f"  Duration: {baseline_metrics['duration']/1e6:.3f} ms")
        print(f"  Memory Bandwidth: {baseline_metrics['dram__bytes.sum']/1e9:.2f} GB")

        print("\n[诊断结果]")
        self._print_diagnosis(baseline_diagnosis)

        # 记录 baseline
        baseline_round = OptimizationRound(
            round_id=0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            kernel_code="// Baseline code",
            operator_type=self.operator_type,
            gpu_name=self.gpu_name,
            bottleneck=baseline_diagnosis.bottleneck,
            bottleneck_confidence=baseline_diagnosis.bottleneck_confidence,
            bandwidth_util=baseline_diagnosis.memory_bandwidth_util,
            occupancy=baseline_diagnosis.achieved_occupancy,
            arithmetic_intensity=baseline_diagnosis.arithmetic_intensity,
            strategy_name="baseline",
            strategy_params={},
            execution_time_ms=baseline_metrics['duration'] / 1e6,
            speedup=1.0,
            compilation_success=True,
            runtime_success=True,
            notes="Baseline implementation"
        )
        self.history.add_round(baseline_round)

        # 迭代优化
        current_diagnosis = baseline_diagnosis
        baseline_time = baseline_metrics['duration']

        for round_num in range(1, max_rounds + 1):
            print("\n" + "=" * 80)
            print(f"Round {round_num}: 优化迭代")
            print("=" * 80)

            # 选择策略
            print(f"\n[步骤 1] 选择优化策略...")
            strategy = self._select_strategy(current_diagnosis)
            if not strategy:
                print("✅ 没有更多可用策略，优化完成")
                break

            print(f"\n推荐策略: {strategy.name}")
            print(f"  描述: {strategy.description}")
            print(f"  预期加速比: {strategy.expected_speedup[0]:.1f}x - {strategy.expected_speedup[1]:.1f}x")
            print(f"  适用瓶颈: {[b.value for b in strategy.bottleneck_types]}")

            # 模拟优化效果
            print(f"\n[步骤 2] 应用优化策略...")
            optimized_metrics = self._apply_strategy(baseline_metrics, strategy, round_num)
            optimized_diagnosis = self.interpreter.interpret(optimized_metrics)

            print("\n[优化后的诊断结果]")
            self._print_diagnosis(optimized_diagnosis)

            # 计算加速比
            optimized_time = optimized_metrics['duration']
            speedup = baseline_time / optimized_time
            cumulative_speedup = baseline_time / optimized_time

            print(f"\n⚡ 性能提升:")
            print(f"  本轮加速比: {speedup:.2f}x")
            print(f"  累计加速比: {cumulative_speedup:.2f}x")
            print(f"  时间: {baseline_time/1e6:.3f} ms → {optimized_time/1e6:.3f} ms")

            # 检测瓶颈转移
            if current_diagnosis.bottleneck != optimized_diagnosis.bottleneck:
                print(f"\n🔄 瓶颈转移检测:")
                print(f"  从: {current_diagnosis.bottleneck}")
                print(f"  到: {optimized_diagnosis.bottleneck}")

            # 记录轮次
            round_data = OptimizationRound(
                round_id=round_num,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                kernel_code=f"// Optimized with {strategy.name}",
                operator_type=self.operator_type,
                gpu_name=self.gpu_name,
                bottleneck=optimized_diagnosis.bottleneck,
                bottleneck_confidence=optimized_diagnosis.bottleneck_confidence,
                bandwidth_util=optimized_diagnosis.memory_bandwidth_util,
                occupancy=optimized_diagnosis.achieved_occupancy,
                arithmetic_intensity=optimized_diagnosis.arithmetic_intensity,
                strategy_name=strategy.name,
                strategy_params={},
                execution_time_ms=optimized_time / 1e6,
                speedup=cumulative_speedup,
                compilation_success=True,
                runtime_success=True,
                notes=f"Applied {strategy.name}"
            )
            self.history.add_round(round_data)

            # 更新当前状态
            current_diagnosis = optimized_diagnosis
            baseline_time = optimized_time

        # 生成报告
        self._generate_final_report()

    def _get_baseline_metrics(self) -> Dict:
        """获取 baseline 性能指标"""
        if self.operator_type == "matmul":
            return {
                'dram__bytes.sum': 5e9,  # 5 GB
                'duration': 45.2e6,  # 45.2 ms
                'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum': 1e8,
                'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum': 1e8,
                'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 5e7,
                'sm__warps_active.avg.pct_of_peak_sustained_active': 62.0,
                'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed': 70.0,
                'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct': 45.0,
            }
        else:
            return {
                'dram__bytes.sum': 1e9,
                'duration': 10e6,
                'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum': 1e7,
                'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum': 1e7,
                'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 1e7,
                'sm__warps_active.avg.pct_of_peak_sustained_active': 50.0,
                'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed': 60.0,
            }

    def _select_strategy(self, diagnosis):
        """选择优化策略"""
        operator_type = OperatorType[self.operator_type.upper()]
        bottleneck_type = BottleneckType[diagnosis.bottleneck.upper()]

        strategies = self.strategy_library.get_applicable_strategies(
            operator_type=operator_type,
            bottleneck=bottleneck_type,
            gpu_compute_capability=10.0  # RTX 5070
        )

        if not strategies:
            return None

        # 选择第一个未尝试的策略
        tried_strategies = {r.strategy_name for r in self.history.rounds}
        for strategy in strategies:
            if strategy.name not in tried_strategies:
                return strategy

        return None

    def _apply_strategy(self, baseline_metrics: Dict, strategy, round_num: int) -> Dict:
        """模拟应用策略后的性能"""
        metrics = baseline_metrics.copy()

        # 根据策略模拟性能提升
        speedup_factor = (strategy.expected_speedup[0] + strategy.expected_speedup[1]) / 2

        # 调整时间
        metrics['duration'] = metrics['duration'] / speedup_factor

        # 根据策略类型调整其他指标
        if strategy.name == "matmul_tiling":
            # 共享内存分块 - 提高带宽利用率
            metrics['smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct'] = 85.0
            metrics['dram__bytes.sum'] = metrics['dram__bytes.sum'] * 0.5  # 减少内存访问

        elif strategy.name == "vectorized_memory":
            # 向量化 - 进一步提高带宽
            metrics['smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct'] = 95.0
            metrics['sm__warps_active.avg.pct_of_peak_sustained_active'] = 75.0

        elif strategy.name == "tensor_core":
            # Tensor Core - 大幅提升计算性能
            metrics['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum'] *= 10
            metrics['sm__warps_active.avg.pct_of_peak_sustained_active'] = 85.0
            metrics['smsp__cycles_active.avg.pct_of_peak_sustained_elapsed'] = 90.0

        return metrics

    def _print_diagnosis(self, diagnosis):
        """打印诊断结果"""
        print(f"  瓶颈: {diagnosis.bottleneck} (置信度: {diagnosis.bottleneck_confidence:.1%})")
        print(f"  带宽利用率: {diagnosis.memory_bandwidth_util:.1f}%")
        print(f"  占用率: {diagnosis.achieved_occupancy:.1f}%")
        print(f"  算术强度: {diagnosis.arithmetic_intensity:.2f} FLOPs/byte")
        print(f"  Roofline 区域: {diagnosis.roofline_region}")
        print(f"  内存访问模式: {diagnosis.memory_access_pattern}")

        if diagnosis.issues:
            print(f"\n  主要问题:")
            for issue in diagnosis.issues[:3]:
                print(f"    [{issue['severity'].upper()}] {issue['description']}")

    def _generate_final_report(self):
        """生成最终报告"""
        print("\n" + "=" * 80)
        print("📊 优化总结报告")
        print("=" * 80)

        baseline = self.history.rounds[0]
        best = self.history.best_round

        print(f"\nKernel: {self.kernel_name}")
        print(f"算子类型: {self.operator_type}")
        print(f"GPU: {self.gpu_name}")
        print(f"优化轮次: {len(self.history.rounds) - 1}")

        print(f"\n📈 性能对比:")
        print(f"  {'轮次':<6} {'策略':<25} {'时间(ms)':<12} {'加速比':<10} {'瓶颈':<20}")
        print(f"  {'-'*6} {'-'*25} {'-'*12} {'-'*10} {'-'*20}")

        for r in self.history.rounds:
            print(f"  {r.round_id:<6} {r.strategy_name:<25} {r.execution_time_ms:<12.3f} {r.speedup:<10.2f}x {r.bottleneck:<20}")

        print(f"\n🎯 最终结果:")
        print(f"  初始时间: {baseline.execution_time_ms:.3f} ms")
        print(f"  最佳时间: {best.execution_time_ms:.3f} ms")
        print(f"  总加速比: {best.speedup:.2f}x")
        print(f"  性能提升: {(1 - best.execution_time_ms/baseline.execution_time_ms)*100:.1f}%")

        print(f"\n🔍 优化历程:")
        for i, r in enumerate(self.history.rounds[1:], 1):
            print(f"  Round {i}: {r.strategy_name}")
            print(f"    - 加速比: {r.speedup:.2f}x")
            print(f"    - 瓶颈: {r.bottleneck}")

        # 保存报告
        report = {
            "kernel": self.kernel_name,
            "operator_type": self.operator_type,
            "gpu": self.gpu_name,
            "total_rounds": len(self.history.rounds) - 1,
            "baseline_time_ms": baseline.execution_time_ms,
            "best_time_ms": best.execution_time_ms,
            "total_speedup": best.speedup,
            "improvement_percent": (1 - best.execution_time_ms/baseline.execution_time_ms)*100,
            "rounds": [
                {
                    "round": r.round_id,
                    "strategy": r.strategy_name,
                    "time_ms": r.execution_time_ms,
                    "speedup": r.speedup,
                    "bottleneck": r.bottleneck,
                }
                for r in self.history.rounds
            ]
        }

        report_path = self.workspace / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n💾 报告已保存: {report_path}")

        # 保存历史
        history_path = self.workspace / "optimization_history.json"
        with open(history_path, 'w') as f:
            json.dump([asdict(r) for r in self.history.rounds], f, indent=2, ensure_ascii=False)

        print(f"💾 历史已保存: {history_path}")


def main():
    """主函数 - 演示多个算子的优化"""
    print("🚀 KernelForge-Optimizer 完整演示")
    print("=" * 80)
    print("展示完整的端到端优化流程：")
    print("  1. 代码理解和分析")
    print("  2. NCU profiling 和瓶颈诊断")
    print("  3. 智能策略推荐")
    print("  4. 多轮迭代优化")
    print("  5. 瓶颈转移检测")
    print("  6. 完整报告生成")
    print("=" * 80)

    # 演示 1: MatMul 优化
    print("\n\n" + "🎯" * 40)
    print("演示 1: 矩阵乘法 (MatMul) 优化")
    print("🎯" * 40)

    demo1 = KernelOptimizationDemo(
        kernel_name="matmul_2048x2048",
        operator_type="matmul",
        gpu_name="RTX 5070"
    )
    demo1.run_optimization_demo(max_rounds=3)

    print("\n\n✅ 演示完成！")
    print(f"结果保存在: demo-optimization-results/")


if __name__ == "__main__":
    main()
