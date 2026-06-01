#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的 Kernel 优化流程 - 简化版

直接使用编译好的二进制文件进行性能测试和对比
"""

import subprocess
import json
import time
from pathlib import Path

def run_kernel_and_measure(binary_path: Path) -> dict:
    """运行 kernel 并测量性能"""
    try:
        result = subprocess.run(
            [str(binary_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            output = result.stdout
            # 解析输出
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
                if '分块版本:' in line:
                    try:
                        parts = line.split(',')
                        time_ms = float(parts[0].split(':')[1].strip().split()[0])
                        gflops = float(parts[1].strip().split()[0])
                    except:
                        pass

            return {
                'success': True,
                'time_ms': time_ms,
                'gflops': gflops,
                'output': output
            }
        else:
            return {
                'success': False,
                'error': result.stderr
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    workspace = Path("kernels-workspace/matmul/matmul_baseline")

    print("=" * 80)
    print("🚀 KernelForge 优化结果分析")
    print("=" * 80)

    # 测试 baseline
    print("\n[1] 测试 Baseline...")
    baseline_bin = workspace / "baseline.bin"
    baseline_result = run_kernel_and_measure(baseline_bin)

    if baseline_result['success']:
        print(f"✅ Baseline 运行成功")
        print(f"   时间: {baseline_result['time_ms']:.3f} ms")
        print(f"   性能: {baseline_result['gflops']:.2f} GFLOPS")
    else:
        print(f"❌ Baseline 运行失败: {baseline_result.get('error', 'Unknown')}")
        return

    # 测试优化版本
    print("\n[2] 测试优化版本...")
    optimized_bin = workspace / "round_1.bin"
    optimized_result = run_kernel_and_measure(optimized_bin)

    if optimized_result['success']:
        print(f"✅ 优化版本运行成功")
        print(optimized_result['output'])

        # 计算加速比
        if baseline_result['time_ms'] and optimized_result['time_ms']:
            speedup = baseline_result['time_ms'] / optimized_result['time_ms']
            print(f"\n⚡ 性能对比:")
            print(f"   Baseline: {baseline_result['time_ms']:.3f} ms")
            print(f"   Optimized: {optimized_result['time_ms']:.3f} ms")
            print(f"   加速比: {speedup:.2f}x")
            print(f"   性能提升: {(speedup - 1) * 100:.1f}%")
    else:
        print(f"❌ 优化版本运行失败: {optimized_result.get('error', 'Unknown')}")

    # 生成报告
    report = {
        "kernel": "matmul_baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": baseline_result,
        "optimized": optimized_result,
        "speedup": baseline_result['time_ms'] / optimized_result['time_ms'] if baseline_result.get('time_ms') and optimized_result.get('time_ms') else None
    }

    report_path = workspace / "performance_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 报告已保存: {report_path}")

if __name__ == "__main__":
    main()
