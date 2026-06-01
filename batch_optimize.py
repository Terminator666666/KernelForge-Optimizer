#!/usr/bin/env python3
"""
Batch optimization script for multiple kernels.
Generates comparison reports suitable for interview demonstrations.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add parent directory to path to import CudaForge modules
sys.path.insert(0, str(Path(__file__).parent.parent / "CudaForge-main" / "CudaForge-main"))

from main_real_gpu import EnhancedOptimizerRealGPU


def find_kernels(kernel_dir: str) -> List[str]:
    """Find all Python kernel files in directory."""
    kernel_dir = Path(kernel_dir)
    if not kernel_dir.exists():
        print(f"Error: Directory {kernel_dir} does not exist")
        return []

    kernels = list(kernel_dir.glob("*.py"))
    # Filter out __init__.py and other non-kernel files
    kernels = [k for k in kernels if not k.name.startswith("_")]
    return [str(k) for k in kernels]


def run_optimization(
    kernel_path: str,
    output_dir: str,
    max_rounds: int = 5,
    server_type: str = "deepseek",
    model_name: str = "deepseek-chat"
) -> Dict[str, Any]:
    """Run optimization for a single kernel."""

    kernel_name = Path(kernel_path).stem
    print(f"\n{'='*80}")
    print(f"Optimizing: {kernel_name}")
    print(f"{'='*80}\n")

    # Create output directory for this kernel
    kernel_output_dir = Path(output_dir) / kernel_name
    kernel_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize optimizer
    optimizer = EnhancedOptimizerRealGPU(
        server_type=server_type,
        model_name=model_name
    )

    # Run optimization
    start_time = time.time()
    try:
        result = optimizer.optimize_kernel(
            kernel_path=kernel_path,
            max_rounds=max_rounds,
            output_dir=str(kernel_output_dir)
        )
        elapsed_time = time.time() - start_time

        # Add metadata
        result["kernel_name"] = kernel_name
        result["kernel_path"] = kernel_path
        result["elapsed_time"] = elapsed_time
        result["success"] = True

        # Save detailed report
        report_path = kernel_output_dir / "optimization_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Optimization completed in {elapsed_time:.1f}s")
        print(f"  Final speedup: {result.get('final_speedup', 'N/A')}x")
        print(f"  Report saved to: {report_path}")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Optimization failed: {e}")

        return {
            "kernel_name": kernel_name,
            "kernel_path": kernel_path,
            "elapsed_time": elapsed_time,
            "success": False,
            "error": str(e)
        }


def generate_summary_report(results: List[Dict[str, Any]], output_dir: str):
    """Generate a summary report comparing all optimizations."""

    output_dir = Path(output_dir)

    # Generate markdown report
    report_lines = [
        "# KernelForge-Optimizer Batch Results",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Kernels:** {len(results)}",
        f"**Successful:** {sum(1 for r in results if r['success'])}",
        "",
        "## Summary Table",
        "",
        "| Kernel | Status | Rounds | Final Speedup | Time (s) |",
        "|--------|--------|--------|---------------|----------|"
    ]

    for result in results:
        if result["success"]:
            status = "✓"
            rounds = result.get("total_rounds", "N/A")
            speedup = f"{result.get('final_speedup', 0):.2f}x"
        else:
            status = "✗"
            rounds = "-"
            speedup = "-"

        time_str = f"{result['elapsed_time']:.1f}"
        kernel_name = result["kernel_name"]

        report_lines.append(
            f"| {kernel_name} | {status} | {rounds} | {speedup} | {time_str} |"
        )

    report_lines.extend([
        "",
        "## Detailed Results",
        ""
    ])

    # Add detailed results for each kernel
    for result in results:
        kernel_name = result["kernel_name"]
        report_lines.append(f"### {kernel_name}")
        report_lines.append("")

        if result["success"]:
            report_lines.extend([
                f"**Status:** Success ✓",
                f"**Total Rounds:** {result.get('total_rounds', 'N/A')}",
                f"**Final Speedup:** {result.get('final_speedup', 0):.2f}x",
                f"**Optimization Time:** {result['elapsed_time']:.1f}s",
                "",
                "**Optimization History:**",
                ""
            ])

            # Add round-by-round breakdown
            history = result.get("optimization_history", [])
            for i, round_info in enumerate(history, 1):
                speedup = round_info.get("speedup", 0)
                strategy = round_info.get("strategy_used", "Unknown")
                bottleneck = round_info.get("primary_bottleneck", "Unknown")

                report_lines.extend([
                    f"**Round {i}:**",
                    f"- Strategy: {strategy}",
                    f"- Bottleneck: {bottleneck}",
                    f"- Speedup: {speedup:.2f}x",
                    ""
                ])

            # Add final diagnosis
            final_diagnosis = result.get("final_diagnosis", {})
            if final_diagnosis:
                report_lines.extend([
                    "**Final Performance Analysis:**",
                    f"- Primary Bottleneck: {final_diagnosis.get('primary_bottleneck', 'N/A')}",
                    f"- Memory Bandwidth Utilization: {final_diagnosis.get('memory_bandwidth_utilization', 0):.1f}%",
                    f"- Compute Utilization: {final_diagnosis.get('compute_utilization', 0):.1f}%",
                    f"- Occupancy: {final_diagnosis.get('occupancy', 0):.1f}%",
                    ""
                ])
        else:
            report_lines.extend([
                f"**Status:** Failed ✗",
                f"**Error:** {result.get('error', 'Unknown error')}",
                f"**Time:** {result['elapsed_time']:.1f}s",
                ""
            ])

        report_lines.append("---")
        report_lines.append("")

    # Add statistics
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_speedup = sum(r.get("final_speedup", 0) for r in successful_results) / len(successful_results)
        max_speedup = max(r.get("final_speedup", 0) for r in successful_results)
        total_time = sum(r["elapsed_time"] for r in results)

        report_lines.extend([
            "## Statistics",
            "",
            f"- **Average Speedup:** {avg_speedup:.2f}x",
            f"- **Maximum Speedup:** {max_speedup:.2f}x",
            f"- **Total Optimization Time:** {total_time:.1f}s",
            f"- **Success Rate:** {len(successful_results)}/{len(results)} ({100*len(successful_results)/len(results):.1f}%)",
            ""
        ])

    # Save report
    report_path = output_dir / "summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'='*80}")
    print(f"Summary report saved to: {report_path}")
    print(f"{'='*80}\n")

    # Also save JSON version
    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"JSON report saved to: {json_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch optimize multiple CUDA kernels"
    )
    parser.add_argument(
        "--kernel_dir",
        type=str,
        required=True,
        help="Directory containing kernel files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=5,
        help="Maximum optimization rounds per kernel (default: 5)"
    )
    parser.add_argument(
        "--server_type",
        type=str,
        default="deepseek",
        choices=["deepseek", "openai", "anthropic", "together", "gemini"],
        help="LLM server type (default: deepseek)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-chat",
        help="Model name (default: deepseek-chat)"
    )
    parser.add_argument(
        "--kernels",
        type=str,
        nargs="+",
        help="Specific kernel files to optimize (optional)"
    )

    args = parser.parse_args()

    # Find kernels
    if args.kernels:
        kernel_paths = args.kernels
    else:
        kernel_paths = find_kernels(args.kernel_dir)

    if not kernel_paths:
        print("Error: No kernel files found")
        return 1

    print(f"\nFound {len(kernel_paths)} kernel(s) to optimize:")
    for kp in kernel_paths:
        print(f"  - {Path(kp).name}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run optimizations
    results = []
    for i, kernel_path in enumerate(kernel_paths, 1):
        print(f"\n[{i}/{len(kernel_paths)}] Processing {Path(kernel_path).name}...")

        result = run_optimization(
            kernel_path=kernel_path,
            output_dir=str(output_dir),
            max_rounds=args.max_rounds,
            server_type=args.server_type,
            model_name=args.model_name
        )
        results.append(result)

    # Generate summary report
    generate_summary_report(results, str(output_dir))

    # Print final summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*80}")
    print(f"Batch optimization complete!")
    print(f"  Total: {len(results)} kernels")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
