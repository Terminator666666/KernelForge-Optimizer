#!/usr/bin/env python3
"""
Quick environment verification script.
Tests CUDA tools, GPU detection, and API connectivity.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.gpu_arch_detection import detect_gpu


def check_command(cmd: str, name: str) -> bool:
    """Check if a command is available."""
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"[OK] {name} found: {version}")
            return True
        else:
            print(f"[FAIL] {name} not found or error")
            return False
    except FileNotFoundError:
        print(f"[FAIL] {name} not found in PATH")
        return False
    except Exception as e:
        print(f"[FAIL] {name} check failed: {e}")
        return False


def check_gpu() -> bool:
    """Check GPU detection."""
    try:
        result = detect_gpu()
        if result:
            # detect_gpu() 返回元组 (gpu_name, specs_dict)
            gpu_name, gpu_specs = result
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  Architecture: {gpu_specs['architecture']} (SM {gpu_specs['compute_capability']})")
            print(f"  SMs: {gpu_specs['sm_count']}")
            print(f"  Memory Bandwidth: {gpu_specs['peak_bandwidth_gbps']} GB/s")
            return True
        else:
            print("✗ No GPU detected")
            return False
    except Exception as e:
        print(f"✗ GPU detection failed: {e}")
        return False


def check_api_key() -> bool:
    """Check if API key is configured."""
    # Try to load from .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("DEEPSEEK_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    if key and not key.startswith("your_"):
                        print(f"✓ API key configured in .env")
                        return True

    # Try environment variable
    if os.environ.get("DEEPSEEK_API_KEY"):
        print(f"✓ API key configured in environment")
        return True

    print("✗ API key not configured")
    print("  Please set DEEPSEEK_API_KEY in .env file or environment")
    return False


def check_api_connection() -> bool:
    """Test API connectivity."""
    try:
        import openai
        from dotenv import load_dotenv

        # Load .env file
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("✗ API key not found")
            return False

        # Test connection
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )

        if response.choices:
            print("✓ DeepSeek API connection successful")
            return True
        else:
            print("✗ DeepSeek API returned empty response")
            return False

    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Run: pip install openai python-dotenv")
        return False
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False


def check_cudaforge() -> bool:
    """Check if CudaForge is available."""
    cudaforge_path = Path(__file__).parent.parent / "CudaForge-main" / "CudaForge-main"
    if cudaforge_path.exists():
        print(f"✓ CudaForge found at: {cudaforge_path}")

        # Check for KernelBench
        kernelbench_path = cudaforge_path / "KernelBench" / "level1"
        if kernelbench_path.exists():
            kernel_count = len(list(kernelbench_path.glob("*.py")))
            print(f"  KernelBench: {kernel_count} test kernels available")
        return True
    else:
        print(f"✗ CudaForge not found at: {cudaforge_path}")
        return False


def main():
    print("=" * 80)
    print("KernelForge-Optimizer Environment Check")
    print("=" * 80)
    print()

    checks = []

    # Check CUDA tools
    print("=== CUDA Tools ===")
    checks.append(("nvcc", check_command("nvcc", "CUDA Compiler")))
    checks.append(("ncu", check_command("ncu", "Nsight Compute")))
    print()

    # Check GPU
    print("=== GPU Detection ===")
    checks.append(("gpu", check_gpu()))
    print()

    # Check API configuration
    print("=== API Configuration ===")
    checks.append(("api_key", check_api_key()))
    checks.append(("api_connection", check_api_connection()))
    print()

    # Check CudaForge
    print("=== CudaForge Integration ===")
    checks.append(("cudaforge", check_cudaforge()))
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {name}")

    print()
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print()
        print("🎉 All checks passed! You're ready to run optimizations.")
        print()
        print("Quick start:")
        print("  python test_real_gpu.py  # Test with a simple kernel")
        print("  python main_real_gpu.py --kernel_path <path> --max_rounds 3")
        return 0
    else:
        print()
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        if not checks[0][1]:  # nvcc
            print("  - Install CUDA Toolkit from https://developer.nvidia.com/cuda-toolkit")
        if not checks[1][1]:  # ncu
            print("  - Install Nsight Compute (included in CUDA Toolkit)")
        if not checks[3][1]:  # api_key
            print("  - Set DEEPSEEK_API_KEY in .env file")
        if not checks[5][1]:  # cudaforge
            print("  - Clone CudaForge to parent directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
