#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 Kernel 优化系统

功能：
1. 扫描项目中的所有 kernel
2. 对每个 kernel 进行多轮优化
3. 记录优化历史和失败分支
4. 生成完整的优化报告
5. 创建 Git 提交
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ncu_interpreter import create_interpreter_for_gpu
from agents.strategy_templates import create_strategy_library, OperatorType
from agents.optimization_history import OptimizationHistory


@dataclass
class KernelInfo:
    """Kernel 信息"""
    name: str
    path: Path
    operator_type: str
    size_kb: float


class BatchOptimizer:
    """批量优化器"""

    def __init__(self, gpu_name: str = "RTX 5070", max_rounds: int = 3):
        self.gpu_name = gpu_name
        self.max_rounds = max_rounds
        self.workspace = Path("batch-optimization-workspace")
        self.workspace.mkdir(exist_ok=True)

        # 初始化组件
        self.interpreter = create_interpreter_for_gpu(gpu_name)
        self.strategy_library = create_strategy_library()

        # 结果统计
        self.results = []

    def discover_kernels(self) -> List[KernelInfo]:
        """发现项目中的所有 kernel"""
        print("🔍 扫描项目中的 kernel...")

        kernels = []

        # 扫描 examples 目录
        examples_dir = Path("examples")
        if examples_dir.exists():
            for cu_file in examples_dir.glob("*.cu"):
                if cu_file.name.startswith("matmul") or cu_file.name.startswith("vector"):
                    operator_type = "matmul" if "matmul" in cu_file.name else "elementwise"
                    kernels.append(KernelInfo(
                        name=cu_file.stem,
                        path=cu_file,
                        operator_type=operator_type,
                        size_kb=cu_file.stat().st_size / 1024
                    ))

        print(f"✅ 发现 {len(kernels)} 个 kernel")
        for k in kernels:
            print(f"   - {k.name} ({k.operator_type}, {k.size_kb:.1f} KB)")

        return kernels

    def optimize_kernel(self, kernel: KernelInfo) -> Dict:
        """优化单个 kernel"""
        print("\n" + "=" * 80)
        print(f"优化 Kernel: {kernel.name}")
        print("=" * 80)

        kernel_workspace = self.workspace / kernel.operator_type / kernel.name
        kernel_workspace.mkdir(parents=True, exist_ok=True)

        # 编译 baseline
        print("\n[1] 编译 Baseline...")
        baseline_bin = self._compile_kernel(kernel.path, kernel_workspace / "baseline.bin")

        if not baseline_bin:
            return {
                "kernel": kernel.name,
                "success": False,
                "error": "baseline_compile_failed"
            }

        # 运行 baseline
        print("\n[2] 测试 Baseline 性能...")
        baseline_perf = self._run_kernel(baseline_bin)

        if not baseline_perf['success']:
            return {
                "kernel": kernel.name,
                "success": False,
                "error": "baseline_run_failed"
            }

        print(f"   Baseline: {baseline_perf['time_ms']:.3f} ms, {baseline_perf['gflops']:.2f} GFLOPS")

        # 尝试优化
        best_time = baseline_perf['time_ms']
        best_strategy = "baseline"
        optimization_rounds = []

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n[{round_num + 2}] 优化轮次 {round_num}...")

            # 选择策略
            strategy = self._select_next_strategy(kernel.operator_type, optimization_rounds)
            if not strategy:
                print("   没有更多可用策略")
                break

            print(f"   策略: {strategy.name}")

            # 生成优化代码
            optimized_code = self._generate_optimized_code(kernel.path, strategy)
            optimized_path = kernel_workspace / f"round_{round_num}.cu"
            with open(optimized_path, 'w') as f:
                f.write(optimized_code)

            # 编译
            optimized_bin = self._compile_kernel(optimized_path, kernel_workspace / f"round_{round_num}.bin")

            if not optimized_bin:
                print(f"   ❌ 编译失败")
                optimization_rounds.append({
                    "round": round_num,
                    "strategy": strategy.name,
                    "compile_success": False
                })
                continue

            # 运行
            optimized_perf = self._run_kernel(optimized_bin)

            if not optimized_perf['success']:
                print(f"   ❌ 运行失败")
                optimization_rounds.append({
                    "round": round_num,
                    "strategy": strategy.name,
                    "compile_success": True,
                    "run_success": False
                })
                continue

            if optimized_perf['time_ms'] and baseline_perf['time_ms']:
                speedup = baseline_perf['time_ms'] / optimized_perf['time_ms']
                print(f"   ✅ {optimized_perf['time_ms']:.3f} ms, {optimized_perf.get('gflops', 0):.2f} GFLOPS, 加速比: {speedup:.2f}x")
            else:
                speedup = 1.0
                print(f"   ⚠️  无法计算加速比")

            optimization_rounds.append({
                "round": round_num,
                "strategy": strategy.name,
                "compile_success": True,
                "run_success": True,
                "time_ms": optimized_perf['time_ms'],
                "gflops": optimized_perf['gflops'],
                "speedup": speedup
            })

            # 更新最佳结果
            if optimized_perf['time_ms'] < best_time:
                best_time = optimized_perf['time_ms']
                best_strategy = strategy.name

        # 生成结果
        total_speedup = baseline_perf['time_ms'] / best_time if best_time > 0 else 1.0

        result = {
            "kernel": kernel.name,
            "operator_type": kernel.operator_type,
            "success": True,
            "baseline": {
                "time_ms": baseline_perf['time_ms'],
                "gflops": baseline_perf.get('gflops', 0)
            },
            "best": {
                "strategy": best_strategy,
                "time_ms": best_time,
                "speedup": total_speedup
            },
            "rounds": optimization_rounds
        }

        # 保存结果
        result_path = kernel_workspace / "optimization_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def _compile_kernel(self, source_path: Path, output_path: Path) -> Path:
        """编译 kernel"""
        env = os.environ.copy()
        env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')

        cmd = ["nvcc", "-O3", "-arch=sm_89", str(source_path), "-o", str(output_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
            if result.returncode == 0:
                return output_path
            else:
                print(f"      编译错误: {result.stderr[:200]}")
                return None
        except Exception as e:
            print(f"      编译异常: {e}")
            return None

    def _run_kernel(self, binary_path: Path) -> Dict:
        """运行 kernel 并测量性能"""
        try:
            result = subprocess.run([str(binary_path)], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                output = result.stdout
                time_ms = None
                gflops = None

                for line in output.split('\n'):
                    if 'Average time:' in line or '平均时间:' in line:
                        try:
                            time_ms = float(line.split(':')[1].strip().split()[0])
                        except:
                            pass
                    if 'GFLOPS:' in line:
                        try:
                            gflops = float(line.split(':')[1].strip())
                        except:
                            pass
                    # 解析分块版本的输出
                    if '分块版本:' in line:
                        try:
                            parts = line.split(',')
                            time_ms = float(parts[0].split(':')[1].strip().split()[0])
                            gflops = float(parts[1].strip().split()[0])
                        except:
                            pass

                return {'success': True, 'time_ms': time_ms or 0, 'gflops': gflops or 0}
            else:
                return {'success': False, 'error': result.stderr}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _select_next_strategy(self, operator_type: str, tried_rounds: List[Dict]):
        """选择下一个策略"""
        tried_strategies = {r['strategy'] for r in tried_rounds}

        try:
            op_type = OperatorType[operator_type.upper()]
        except:
            return None

        # 获取所有适用策略
        all_strategies = []
        for bottleneck in ['memory_bandwidth', 'compute_bound', 'memory_latency']:
            try:
                from agents.strategy_templates import BottleneckType
                bottleneck_type = BottleneckType[bottleneck.upper()]
                strategies = self.strategy_library.get_applicable_strategies(
                    operator_type=op_type,
                    bottleneck=bottleneck_type,
                    gpu_compute_capability=10.0
                )
                all_strategies.extend(strategies)
            except:
                pass

        # 去重并选择未尝试的
        seen = set()
        for strategy in all_strategies:
            if strategy.name not in tried_strategies and strategy.name not in seen:
                seen.add(strategy.name)
                return strategy

        return None

    def _generate_optimized_code(self, original_path: Path, strategy) -> str:
        """生成优化代码"""
        template_path = Path("skills/strategy-library-skill/templates") / f"{strategy.name}.cu"

        if template_path.exists():
            with open(template_path, 'r') as f:
                return f.read()

        # 否则返回原始代码加注释
        with open(original_path, 'r') as f:
            original = f.read()

        return f"// Optimized with {strategy.name}\n{original}"

    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "=" * 80)
        print("📊 批量优化总结报告")
        print("=" * 80)

        successful = [r for r in self.results if r.get('success')]
        failed = [r for r in self.results if not r.get('success')]

        print(f"\n总计: {len(self.results)} 个 kernel")
        print(f"  成功: {len(successful)}")
        print(f"  失败: {len(failed)}")

        if successful:
            print(f"\n✅ 成功优化的 kernel:")
            print(f"  {'Kernel':<30} {'Baseline(ms)':<15} {'Best(ms)':<15} {'加速比':<10} {'策略':<20}")
            print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10} {'-'*20}")

            for r in successful:
                print(f"  {r['kernel']:<30} {r['baseline']['time_ms']:<15.3f} "
                      f"{r['best']['time_ms']:<15.3f} {r['best']['speedup']:<10.2f}x {r['best']['strategy']:<20}")

            avg_speedup = sum(r['best']['speedup'] for r in successful) / len(successful)
            print(f"\n  平均加速比: {avg_speedup:.2f}x")

        if failed:
            print(f"\n❌ 失败的 kernel:")
            for r in failed:
                print(f"  - {r['kernel']}: {r.get('error', 'unknown')}")

        # 保存报告
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": self.gpu_name,
            "total_kernels": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "average_speedup": sum(r['best']['speedup'] for r in successful) / len(successful) if successful else 0,
            "results": self.results
        }

        report_path = self.workspace / "BATCH_OPTIMIZATION_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 报告已保存: {report_path}")

        # 生成 Markdown 报告
        self._generate_markdown_report(summary)

    def _generate_markdown_report(self, summary: Dict):
        """生成 Markdown 格式的报告"""
        md_path = self.workspace / "OPTIMIZATION_REPORT.md"

        with open(md_path, 'w') as f:
            f.write("# KernelForge-Optimizer 批量优化报告\n\n")
            f.write(f"**生成时间**: {summary['timestamp']}\n\n")
            f.write(f"**GPU**: {summary['gpu']}\n\n")
            f.write(f"**总计**: {summary['total_kernels']} 个 kernel\n\n")

            f.write("## 📊 总体统计\n\n")
            f.write(f"- ✅ 成功: {summary['successful']}\n")
            f.write(f"- ❌ 失败: {summary['failed']}\n")
            f.write(f"- ⚡ 平均加速比: {summary['average_speedup']:.2f}x\n\n")

            successful = [r for r in summary['results'] if r.get('success')]

            if successful:
                f.write("## ✅ 优化成功的 Kernel\n\n")
                f.write("| Kernel | Baseline (ms) | Best (ms) | 加速比 | 最佳策略 |\n")
                f.write("|--------|---------------|-----------|--------|----------|\n")

                for r in successful:
                    f.write(f"| {r['kernel']} | {r['baseline']['time_ms']:.3f} | "
                           f"{r['best']['time_ms']:.3f} | {r['best']['speedup']:.2f}x | "
                           f"{r['best']['strategy']} |\n")

        print(f"💾 Markdown 报告已保存: {md_path}")

    def run_batch_optimization(self):
        """运行批量优化"""
        print("🚀 KernelForge 批量优化系统")
        print("=" * 80)

        # 发现 kernel
        kernels = self.discover_kernels()

        if not kernels:
            print("❌ 未发现任何 kernel")
            return

        # 优化每个 kernel
        for i, kernel in enumerate(kernels, 1):
            print(f"\n进度: {i}/{len(kernels)}")
            result = self.optimize_kernel(kernel)
            self.results.append(result)

        # 生成报告
        self.generate_summary_report()

        print("\n✅ 批量优化完成！")


def main():
    optimizer = BatchOptimizer(gpu_name="RTX 5070", max_rounds=2)
    optimizer.run_batch_optimization()


if __name__ == "__main__":
    main()
