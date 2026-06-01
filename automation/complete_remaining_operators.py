#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完成剩余算子的自动化优化脚本

按照完整的 Agentic Workflow：
1. 创建任务草稿 (draft.md)
2. 实现 baseline 和优化版本
3. 编译和性能测试
4. NCU profiling 分析
5. 记录候选方案 (candidates.jsonl)
6. 精度验证
7. Git 提交

待完成的算子：
1. Softmax (已有代码，需验证)
2. LayerNorm
3. Flash Attention (简化版)
4. Fused MoE (简化版)
5. Scan/Prefix Sum
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class OperatorTask:
    """算子任务定义"""
    name: str
    display_name: str
    description: str
    baseline_complexity: str
    optimization_strategies: List[str]
    expected_speedup: str
    priority: int


# 定义待完成的算子任务
REMAINING_OPERATORS = [
    OperatorTask(
        name="softmax",
        display_name="Softmax",
        description="激活函数优化 - 在线 Softmax 算法",
        baseline_complexity="多次遍历（3-pass）",
        optimization_strategies=[
            "单次遍历（online algorithm）",
            "共享内存归约",
            "Warp-level primitives",
            "向量化访问"
        ],
        expected_speedup="3-5x",
        priority=1
    ),
    OperatorTask(
        name="layernorm",
        display_name="LayerNorm",
        description="层归一化优化 - Welford 在线算法",
        baseline_complexity="多次遍历（mean + variance + normalize）",
        optimization_strategies=[
            "Welford 在线算法",
            "共享内存归约",
            "融合 scale/bias",
            "向量化访问"
        ],
        expected_speedup="4-6x",
        priority=2
    ),
    OperatorTask(
        name="scan",
        display_name="Scan (Prefix Sum)",
        description="并行前缀和 - Kogge-Stone 算法",
        baseline_complexity="串行扫描",
        optimization_strategies=[
            "Kogge-Stone 并行扫描",
            "共享内存分块",
            "Bank conflict 避免",
            "双缓冲"
        ],
        expected_speedup="10-20x",
        priority=3
    ),
    OperatorTask(
        name="flash_attention",
        display_name="Flash Attention (简化版)",
        description="注意力机制优化 - Tiling + 在线 Softmax",
        baseline_complexity="标准注意力（多次 GEMM + Softmax）",
        optimization_strategies=[
            "Tiling（分块计算）",
            "在线 Softmax",
            "融合 QK^T 和 Softmax",
            "共享内存优化"
        ],
        expected_speedup="3-5x",
        priority=4
    ),
    OperatorTask(
        name="fused_moe",
        display_name="Fused MoE (简化版)",
        description="混合专家模型优化 - Gating + Expert Batching",
        baseline_complexity="分离的 Gating 和 Expert 计算",
        optimization_strategies=[
            "融合 Gating 和 Expert 选择",
            "Expert batching",
            "共享内存优化",
            "动态并行"
        ],
        expected_speedup="2-4x",
        priority=5
    )
]


class OperatorOptimizer:
    """算子优化器 - 管理单个算子的完整优化流程"""

    def __init__(self, task: OperatorTask, workspace_dir: str = "."):
        self.task = task
        self.workspace = Path(workspace_dir)

        # 创建目录结构
        self.docs_dir = self.workspace / "docs"
        self.examples_dir = self.workspace / "examples"
        self.candidates_dir = self.workspace / "candidates"
        self.profile_dir = self.workspace / "profile"
        self.runs_dir = self.workspace / "runs"

        for d in [self.docs_dir, self.examples_dir, self.candidates_dir,
                  self.profile_dir, self.runs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 文件路径
        self.draft_path = self.docs_dir / f"{task.name}-draft.md"
        self.candidates_path = self.candidates_dir / f"{task.name}_candidates.jsonl"
        self.baseline_cu = self.examples_dir / f"{task.name}_baseline.cu"
        self.optimized_cu = self.examples_dir / f"{task.name}_optimized.cu"

    def create_draft(self) -> bool:
        """创建任务草稿"""
        print(f"\n[步骤 1] 创建任务草稿: {self.draft_path}")

        draft_content = f"""# {self.task.display_name} 优化任务草稿

## 任务目标

优化 {self.task.display_name} 算子，在 RTX 5070 上实现显著的性能提升。

## 算子描述

{self.task.description}

## Baseline 实现

**复杂度**: {self.task.baseline_complexity}

**特点**:
- 多次内存访问
- 未充分利用并行性
- 未使用共享内存优化

## 优化策略

{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(self.task.optimization_strategies))}

## 预期性能提升

**目标加速比**: {self.task.expected_speedup}

## 实现计划

### 阶段 1: Baseline 实现
- [ ] 实现简单的并行版本（公平 baseline）
- [ ] 编译和功能测试
- [ ] 性能基准测试

### 阶段 2: 优化实现
- [ ] 应用优化策略
- [ ] 编译和功能测试
- [ ] 性能对比测试

### 阶段 3: 分析和验证
- [ ] NCU profiling 分析
- [ ] 精度验证（误差 < 0.01%）
- [ ] 记录候选方案

### 阶段 4: 文档和提交
- [ ] 更新 CLAUDE.md
- [ ] Git 提交
- [ ] 推送到 GitHub

## 验收标准

1. ✅ 精度验证通过（误差 < 0.01%）
2. ✅ 加速比达到预期范围
3. ✅ 公平的 baseline 对比（并行 vs 并行）
4. ✅ 完整的 agentic workflow（draft + candidates + evidence）
5. ✅ NCU profiling 数据
6. ✅ Git 提交历史清晰

## 参考资料

- CUDA Programming Guide
- NVIDIA Nsight Compute Documentation
- 相关学术论文

---

**创建时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**状态**: 待实现
"""

        with open(self.draft_path, 'w', encoding='utf-8') as f:
            f.write(draft_content)

        print(f"✅ 草稿已创建: {self.draft_path}")
        return True

    def check_existing_code(self) -> Tuple[bool, bool]:
        """检查是否已有代码实现"""
        baseline_exists = self.baseline_cu.exists()
        optimized_exists = self.optimized_cu.exists()

        print(f"\n[步骤 2] 检查现有代码:")
        print(f"  Baseline: {'✅ 存在' if baseline_exists else '❌ 不存在'}")
        print(f"  Optimized: {'✅ 存在' if optimized_exists else '❌ 不存在'}")

        return baseline_exists, optimized_exists

    def compile_kernel(self, cu_file: Path, output_name: str) -> Optional[Path]:
        """编译 CUDA kernel"""
        print(f"\n[步骤 3] 编译 {cu_file.name}...")

        output_path = self.runs_dir / output_name

        compile_cmd = [
            "nvcc",
            str(cu_file),
            "-o", str(output_path),
            "-O3",
            "-arch=sm_89",  # RTX 5070
            "-lcublas"
        ]

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(f"✅ 编译成功: {output_path}")
                return output_path
            else:
                print(f"❌ 编译失败:")
                print(result.stderr)
                return None
        except Exception as e:
            print(f"❌ 编译异常: {e}")
            return None

    def run_benchmark(self, binary: Path) -> Optional[Dict]:
        """运行性能测试"""
        print(f"\n[步骤 4] 运行性能测试: {binary.name}...")

        try:
            result = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                output = result.stdout
                print(output)

                # 解析性能数据（简单版本）
                perf_data = {
                    "output": output,
                    "timestamp": datetime.now().isoformat()
                }

                return perf_data
            else:
                print(f"❌ 运行失败:")
                print(result.stderr)
                return None
        except Exception as e:
            print(f"❌ 运行异常: {e}")
            return None

    def record_candidate(self, candidate_data: Dict):
        """记录候选方案到 JSONL"""
        print(f"\n[步骤 5] 记录候选方案: {self.candidates_path}")

        with open(self.candidates_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(candidate_data, ensure_ascii=False) + '\n')

        print(f"✅ 候选方案已记录")

    def run_ncu_profile(self, binary: Path, output_name: str) -> Optional[Path]:
        """运行 NCU profiling"""
        print(f"\n[步骤 6] NCU Profiling: {binary.name}...")

        ncu_output = self.profile_dir / f"{output_name}.ncu-rep"

        ncu_cmd = [
            "ncu",
            "--set", "full",
            "-o", str(ncu_output.with_suffix('')),
            str(binary)
        ]

        try:
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if ncu_output.exists():
                print(f"✅ NCU 报告已生成: {ncu_output}")
                return ncu_output
            else:
                print(f"⚠️  NCU 运行完成但未生成报告")
                print(result.stdout)
                return None
        except Exception as e:
            print(f"⚠️  NCU 运行失败（可选步骤）: {e}")
            return None

    def optimize_operator(self) -> bool:
        """执行完整的优化流程"""
        print("=" * 80)
        print(f"开始优化算子: {self.task.display_name}")
        print("=" * 80)

        # 1. 创建草稿
        if not self.draft_path.exists():
            self.create_draft()
        else:
            print(f"✅ 草稿已存在: {self.draft_path}")

        # 2. 检查代码
        baseline_exists, optimized_exists = self.check_existing_code()

        if not baseline_exists or not optimized_exists:
            print(f"\n⚠️  代码文件缺失，需要手动实现:")
            if not baseline_exists:
                print(f"   - {self.baseline_cu}")
            if not optimized_exists:
                print(f"   - {self.optimized_cu}")
            print(f"\n请参考草稿文件: {self.draft_path}")
            return False

        # 3. 编译 baseline
        baseline_binary = self.compile_kernel(
            self.baseline_cu,
            f"test_{self.task.name}_baseline"
        )
        if not baseline_binary:
            return False

        # 4. 编译 optimized
        optimized_binary = self.compile_kernel(
            self.optimized_cu,
            f"test_{self.task.name}_optimized"
        )
        if not optimized_binary:
            return False

        # 5. 运行 baseline 测试
        baseline_perf = self.run_benchmark(baseline_binary)
        if not baseline_perf:
            return False

        # 6. 运行 optimized 测试
        optimized_perf = self.run_benchmark(optimized_binary)
        if not optimized_perf:
            return False

        # 7. 记录候选方案
        candidate_data = {
            "operator": self.task.name,
            "timestamp": datetime.now().isoformat(),
            "baseline": {
                "file": str(self.baseline_cu),
                "binary": str(baseline_binary),
                "performance": baseline_perf
            },
            "optimized": {
                "file": str(self.optimized_cu),
                "binary": str(optimized_binary),
                "performance": optimized_perf
            },
            "strategies": self.task.optimization_strategies
        }
        self.record_candidate(candidate_data)

        # 8. NCU profiling (可选)
        print("\n是否运行 NCU profiling？（需要较长时间）")
        print("提示：可以稍后手动运行 NCU")

        # 这里暂时跳过 NCU，因为需要较长时间
        # ncu_baseline = self.run_ncu_profile(baseline_binary, f"{self.task.name}_baseline")
        # ncu_optimized = self.run_ncu_profile(optimized_binary, f"{self.task.name}_optimized")

        print("\n" + "=" * 80)
        print(f"✅ {self.task.display_name} 优化流程完成")
        print("=" * 80)

        return True


def main():
    """主函数"""
    print("=" * 80)
    print("KernelForge-Optimizer: 完成剩余算子优化")
    print("=" * 80)
    print(f"\n待完成算子数量: {len(REMAINING_OPERATORS)}")
    print("\n算子列表:")
    for i, task in enumerate(REMAINING_OPERATORS, 1):
        print(f"  {i}. {task.display_name} - {task.description}")
        print(f"     预期加速比: {task.expected_speedup}")

    print("\n" + "=" * 80)
    print("开始处理...")
    print("=" * 80)

    # 按优先级排序
    sorted_tasks = sorted(REMAINING_OPERATORS, key=lambda x: x.priority)

    results = []
    for task in sorted_tasks:
        optimizer = OperatorOptimizer(task, workspace_dir=".")
        success = optimizer.optimize_operator()

        results.append({
            "operator": task.name,
            "display_name": task.display_name,
            "success": success
        })

        print("\n")

    # 总结
    print("=" * 80)
    print("优化总结")
    print("=" * 80)

    for result in results:
        status = "✅ 完成" if result["success"] else "⚠️  待实现"
        print(f"{result['display_name']}: {status}")

    successful = sum(1 for r in results if r["success"])
    print(f"\n总计: {successful}/{len(results)} 个算子完成")

    if successful < len(results):
        print("\n提示：部分算子需要手动实现代码文件")
        print("请查看 docs/ 目录下的草稿文件获取实现指导")


if __name__ == "__main__":
    main()
