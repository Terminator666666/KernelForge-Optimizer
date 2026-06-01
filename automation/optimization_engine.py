#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KernelForge 优化引擎 - 完整的端到端优化流程

功能：
1. 代码理解和分析
2. NCU profiling
3. 瓶颈诊断
4. 策略推荐
5. 代码生成
6. 编译测试
7. 性能验证
8. 多轮迭代
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ncu_interpreter import NCUInterpreter, create_interpreter_for_gpu
from agents.strategy_templates import StrategyLibrary, OperatorType, BottleneckType, create_strategy_library
from agents.optimization_history import OptimizationHistory, OptimizationRound


@dataclass
class OptimizationConfig:
    """优化配置"""
    kernel_path: str
    operator_type: str
    gpu_name: str = "RTX 5070"
    max_rounds: int = 5
    workspace_dir: str = "kernels-workspace"
    use_llm: bool = False  # 是否使用 LLM（需要 API 密钥）
    llm_model: str = "claude-opus-4"
    compile_timeout: int = 60
    profile_timeout: int = 120


class OptimizationEngine:
    """优化引擎 - 管理完整的优化流程"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.workspace = Path(config.workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.interpreter = create_interpreter_for_gpu(config.gpu_name)
        self.strategy_library = create_strategy_library()
        self.history = OptimizationHistory()

        # 创建 kernel 工作目录
        kernel_name = Path(config.kernel_path).stem
        self.kernel_workspace = self.workspace / config.operator_type / kernel_name
        self.kernel_workspace.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.candidates_dir = self.kernel_workspace / "candidates"
        self.profile_dir = self.kernel_workspace / "profile"
        self.logs_dir = self.kernel_workspace / "logs"

        for d in [self.candidates_dir, self.profile_dir, self.logs_dir]:
            d.mkdir(exist_ok=True)

    def run_optimization(self) -> Dict:
        """运行完整的优化流程"""
        print("=" * 80)
        print("KernelForge 优化引擎启动")
        print("=" * 80)
        print(f"Kernel: {self.config.kernel_path}")
        print(f"算子类型: {self.config.operator_type}")
        print(f"GPU: {self.config.gpu_name}")
        print(f"最大轮次: {self.config.max_rounds}")
        print(f"工作目录: {self.kernel_workspace}")
        print("=" * 80)

        # 读取原始代码
        with open(self.config.kernel_path, 'r') as f:
            baseline_code = f.read()

        # 保存 baseline
        baseline_path = self.kernel_workspace / "baseline.cu"
        with open(baseline_path, 'w') as f:
            f.write(baseline_code)

        print("\n[步骤 1] 分析 Baseline 代码...")
        baseline_analysis = self._analyze_code(baseline_code, "baseline")

        print("\n[步骤 2] 编译 Baseline...")
        baseline_binary = self._compile_kernel(baseline_path, "baseline")
        if not baseline_binary:
            print("❌ Baseline 编译失败，无法继续")
            return {"success": False, "error": "baseline_compile_failed"}

        print("\n[步骤 3] Profile Baseline...")
        baseline_metrics = self._profile_kernel(baseline_binary, "baseline")
        if not baseline_metrics:
            print("⚠️  无法获取 NCU 数据，使用模拟数据继续")
            baseline_metrics = self._get_simulated_metrics()

        print("\n[步骤 4] 诊断 Baseline 性能...")
        baseline_diagnosis = self.interpreter.interpret(baseline_metrics)
        self._print_diagnosis(baseline_diagnosis)

        # 记录 baseline
        baseline_round = OptimizationRound(
            round_id=0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            kernel_code=baseline_code,
            operator_type=self.config.operator_type,
            gpu_name=self.config.gpu_name,
            bottleneck=baseline_diagnosis.bottleneck,
            bottleneck_confidence=baseline_diagnosis.bottleneck_confidence,
            bandwidth_util=baseline_diagnosis.memory_bandwidth_util,
            occupancy=baseline_diagnosis.achieved_occupancy,
            arithmetic_intensity=baseline_diagnosis.arithmetic_intensity,
            strategy_name="baseline",
            strategy_params={},
            execution_time_ms=baseline_metrics.get('duration', 0) / 1e6,
            speedup=1.0,
            compilation_success=True,
            runtime_success=True,
            notes="Baseline implementation"
        )
        self.history.add_round(baseline_round)

        # 迭代优化
        current_code = baseline_code
        current_diagnosis = baseline_diagnosis

        for round_num in range(1, self.config.max_rounds + 1):
            print("\n" + "=" * 80)
            print(f"优化轮次 {round_num}/{self.config.max_rounds}")
            print("=" * 80)

            # 选择策略
            print(f"\n[步骤 {round_num}.1] 选择优化策略...")
            strategy = self._select_strategy(current_diagnosis, round_num)
            if not strategy:
                print("✅ 没有更多可用策略，优化完成")
                break

            print(f"选择策略: {strategy.name}")
            print(f"预期加速比: {strategy.expected_speedup[0]:.1f}x - {strategy.expected_speedup[1]:.1f}x")

            # 生成优化代码
            print(f"\n[步骤 {round_num}.2] 生成优化代码...")
            optimized_code = self._generate_optimized_code(
                current_code, strategy, current_diagnosis
            )

            # 保存候选代码
            candidate_path = self.candidates_dir / f"round_{round_num}.cu"
            with open(candidate_path, 'w') as f:
                f.write(optimized_code)

            # 编译
            print(f"\n[步骤 {round_num}.3] 编译优化代码...")
            optimized_binary = self._compile_kernel(candidate_path, f"round_{round_num}")
            if not optimized_binary:
                print(f"❌ 编译失败，记录失败分支")
                self._record_failure(round_num, strategy, "compilation_failed", optimized_code)
                continue

            # Profile
            print(f"\n[步骤 {round_num}.4] Profile 优化代码...")
            optimized_metrics = self._profile_kernel(optimized_binary, f"round_{round_num}")
            if not optimized_metrics:
                print(f"⚠️  无法获取 NCU 数据")
                optimized_metrics = self._get_simulated_metrics()

            # 诊断
            print(f"\n[步骤 {round_num}.5] 诊断优化效果...")
            optimized_diagnosis = self.interpreter.interpret(optimized_metrics)
            self._print_diagnosis(optimized_diagnosis)

            # 计算加速比
            baseline_time = baseline_metrics.get('duration', 1e6)
            optimized_time = optimized_metrics.get('duration', 1e6)
            speedup = baseline_time / optimized_time

            print(f"\n⚡ 加速比: {speedup:.2f}x")

            # 记录轮次
            round_data = OptimizationRound(
                round_id=round_num,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                kernel_code=optimized_code,
                operator_type=self.config.operator_type,
                gpu_name=self.config.gpu_name,
                bottleneck=optimized_diagnosis.bottleneck,
                bottleneck_confidence=optimized_diagnosis.bottleneck_confidence,
                bandwidth_util=optimized_diagnosis.memory_bandwidth_util,
                occupancy=optimized_diagnosis.achieved_occupancy,
                arithmetic_intensity=optimized_diagnosis.arithmetic_intensity,
                strategy_name=strategy.name,
                strategy_params={},
                execution_time_ms=optimized_time / 1e6,
                speedup=speedup,
                compilation_success=True,
                runtime_success=True,
                notes=f"Applied {strategy.name}"
            )
            self.history.add_round(round_data)

            # 更新当前状态
            if speedup > 1.0:
                current_code = optimized_code
                current_diagnosis = optimized_diagnosis
                print(f"✅ 性能提升，采用此版本")
            else:
                print(f"⚠️  性能未提升，保持原版本")

        # 生成报告
        print("\n" + "=" * 80)
        print("优化完成，生成报告...")
        print("=" * 80)

        report = self._generate_report()

        # 保存历史
        history_path = self.kernel_workspace / "optimization_history.json"
        with open(history_path, 'w') as f:
            json.dump([asdict(r) for r in self.history.rounds], f, indent=2)

        # 保存报告
        report_path = self.kernel_workspace / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self._print_report(report)

        return report

    def _analyze_code(self, code: str, name: str) -> Dict:
        """分析代码"""
        # 简单的代码分析
        analysis = {
            "name": name,
            "lines": len(code.split('\n')),
            "has_shared_memory": "__shared__" in code,
            "has_syncthreads": "__syncthreads" in code,
            "has_vectorization": "float4" in code or "float2" in code,
            "has_tensor_core": "wmma" in code or "mma" in code,
        }
        return analysis

    def _compile_kernel(self, kernel_path: Path, name: str) -> Optional[Path]:
        """编译 kernel"""
        binary_path = self.kernel_workspace / f"{name}.bin"

        # 设置 CUDA 环境变量
        env = os.environ.copy()
        env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')

        # 编译命令
        cmd = [
            "nvcc",
            "-O3",
            "-arch=sm_89",  # RTX 5070
            str(kernel_path),
            "-o", str(binary_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.compile_timeout,
                env=env
            )

            if result.returncode == 0:
                print(f"✅ 编译成功: {binary_path}")
                return binary_path
            else:
                print(f"❌ 编译失败:")
                print(result.stderr)

                # 保存错误日志
                log_path = self.logs_dir / f"{name}_compile_error.log"
                with open(log_path, 'w') as f:
                    f.write(result.stderr)

                return None

        except subprocess.TimeoutExpired:
            print(f"❌ 编译超时")
            return None
        except Exception as e:
            print(f"❌ 编译异常: {e}")
            return None

    def _profile_kernel(self, binary_path: Path, name: str) -> Optional[Dict]:
        """使用 NCU profile kernel"""
        ncu_report = self.profile_dir / f"{name}.ncu-rep"

        # NCU 命令
        cmd = [
            "ncu",
            "--set", "full",
            "--export", str(ncu_report),
            str(binary_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.profile_timeout
            )

            if result.returncode == 0 and ncu_report.exists():
                print(f"✅ Profiling 成功: {ncu_report}")
                # 解析 NCU 报告（简化版）
                return self._parse_ncu_report(ncu_report)
            else:
                print(f"⚠️  Profiling 失败，使用模拟数据")
                return None

        except subprocess.TimeoutExpired:
            print(f"⚠️  Profiling 超时")
            return None
        except Exception as e:
            print(f"⚠️  Profiling 异常: {e}")
            return None

    def _parse_ncu_report(self, report_path: Path) -> Dict:
        """解析 NCU 报告（简化版）"""
        # 实际需要使用 ncu 命令行工具解析
        # 这里返回模拟数据
        return self._get_simulated_metrics()

    def _get_simulated_metrics(self) -> Dict:
        """获取模拟的性能指标"""
        return {
            'dram__bytes.sum': 1e9,
            'duration': 1e6,
            'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum': 1e7,
            'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum': 1e7,
            'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': 1e7,
            'sm__warps_active.avg.pct_of_peak_sustained_active': 50.0,
            'smsp__cycles_active.avg.pct_of_peak_sustained_elapsed': 60.0,
        }

    def _select_strategy(self, diagnosis, round_num: int):
        """选择优化策略"""
        # 获取适用的策略
        operator_type = OperatorType[self.config.operator_type.upper().replace('-', '_')]
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

    def _generate_optimized_code(self, current_code: str, strategy, diagnosis) -> str:
        """生成优化代码"""
        # 如果使用 LLM
        if self.config.use_llm:
            return self._generate_with_llm(current_code, strategy, diagnosis)

        # 否则使用模板
        return self._generate_with_template(current_code, strategy)

    def _generate_with_template(self, current_code: str, strategy) -> str:
        """使用模板生成代码"""
        # 如果策略有对应的模板文件，直接使用
        template_path = Path(__file__).parent.parent / "skills" / "strategy-library-skill" / "templates" / f"{strategy.name}.cu"

        if template_path.exists():
            print(f"  使用模板文件: {template_path}")
            with open(template_path, 'r') as f:
                return f.read()

        # 否则使用策略中的代码模板
        template = strategy.code_template

        # 选择参数
        parameters = {}
        for param_name, rules in strategy.parameter_rules.items():
            if 'default' in rules:
                parameters[param_name] = rules['default']
            elif 'candidates' in rules and rules['candidates']:
                parameters[param_name] = rules['candidates'][0]

        # 替换模板中的占位符
        optimized_code = template
        for param_name, param_value in parameters.items():
            placeholder = f"{{{param_name}}}"
            optimized_code = optimized_code.replace(placeholder, str(param_value))

        # 添加注释
        final_code = f"""// Optimized with strategy: {strategy.name}
// Expected speedup: {strategy.expected_speedup[0]:.1f}x - {strategy.expected_speedup[1]:.1f}x
// Parameters: {parameters}

{optimized_code}
"""
        return final_code

    def _generate_with_llm(self, current_code: str, strategy, diagnosis) -> str:
        """使用 LLM 生成代码"""
        # TODO: 集成 LLM API
        print("⚠️  LLM 集成尚未实现，使用模板")
        return self._generate_with_template(current_code, strategy)

    def _record_failure(self, round_num: int, strategy, error_type: str, code: str):
        """记录失败分支"""
        failure_log = {
            "round": round_num,
            "strategy": strategy.name,
            "error_type": error_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code": code
        }

        log_path = self.logs_dir / f"round_{round_num}_failure.json"
        with open(log_path, 'w') as f:
            json.dump(failure_log, f, indent=2)

    def _print_diagnosis(self, diagnosis):
        """打印诊断结果"""
        print(f"\n诊断结果:")
        print(f"  瓶颈: {diagnosis.bottleneck} (置信度: {diagnosis.bottleneck_confidence:.1%})")
        print(f"  带宽利用率: {diagnosis.memory_bandwidth_util:.1f}%")
        print(f"  占用率: {diagnosis.achieved_occupancy:.1f}%")
        print(f"  算术强度: {diagnosis.arithmetic_intensity:.2f} FLOPs/byte")
        print(f"  Roofline 区域: {diagnosis.roofline_region}")

        if diagnosis.issues:
            print(f"\n  主要问题:")
            for issue in diagnosis.issues[:3]:
                print(f"    - [{issue['severity']}] {issue['description']}")

    def _generate_report(self) -> Dict:
        """生成优化报告"""
        if not self.history.rounds:
            return {"success": False, "error": "no_rounds"}

        baseline = self.history.rounds[0]
        best = self.history.best_round

        report = {
            "success": True,
            "kernel": self.config.kernel_path,
            "operator_type": self.config.operator_type,
            "gpu": self.config.gpu_name,
            "total_rounds": len(self.history.rounds) - 1,
            "baseline": {
                "time_ms": baseline.execution_time_ms,
                "bottleneck": baseline.bottleneck,
            },
            "best": {
                "round": best.round_id if best else 0,
                "strategy": best.strategy_name if best else "baseline",
                "time_ms": best.execution_time_ms if best else baseline.execution_time_ms,
                "speedup": best.speedup if best else 1.0,
            },
            "total_speedup": baseline.execution_time_ms / (best.execution_time_ms if best else baseline.execution_time_ms),
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

        return report

    def _print_report(self, report: Dict):
        """打印报告"""
        print("\n" + "=" * 80)
        print("优化报告")
        print("=" * 80)

        print(f"\nKernel: {report['kernel']}")
        print(f"算子类型: {report['operator_type']}")
        print(f"GPU: {report['gpu']}")
        print(f"优化轮次: {report['total_rounds']}")

        print(f"\nBaseline:")
        print(f"  时间: {report['baseline']['time_ms']:.3f} ms")
        print(f"  瓶颈: {report['baseline']['bottleneck']}")

        print(f"\n最佳结果:")
        print(f"  轮次: {report['best']['round']}")
        print(f"  策略: {report['best']['strategy']}")
        print(f"  时间: {report['best']['time_ms']:.3f} ms")
        print(f"  加速比: {report['best']['speedup']:.2f}x")

        print(f"\n总加速比: {report['total_speedup']:.2f}x")

        print(f"\n优化历史:")
        print(f"  {'轮次':<6} {'策略':<25} {'时间(ms)':<12} {'加速比':<10} {'瓶颈':<20}")
        print(f"  {'-'*6} {'-'*25} {'-'*12} {'-'*10} {'-'*20}")
        for r in report['rounds']:
            print(f"  {r['round']:<6} {r['strategy']:<25} {r['time_ms']:<12.3f} {r['speedup']:<10.2f}x {r['bottleneck']:<20}")


def main():
    """主函数"""
    # 配置
    config = OptimizationConfig(
        kernel_path="/mnt/d/Agent/KernelForge-Optimizer/examples/matmul_baseline.cu",
        operator_type="matmul",
        gpu_name="RTX 5070",
        max_rounds=3,
        use_llm=False,
    )

    # 创建引擎
    engine = OptimizationEngine(config)

    # 运行优化
    report = engine.run_optimization()

    print("\n✅ 优化完成！")
    print(f"工作目录: {engine.kernel_workspace}")


if __name__ == "__main__":
    main()
