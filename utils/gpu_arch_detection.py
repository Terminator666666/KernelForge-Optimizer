"""
GPU Architecture Detection - Detect GPU model and retrieve specifications.

This module provides utilities to detect the current GPU and retrieve its
hardware specifications for optimization decisions.
"""

from typing import Dict, Optional, Tuple
import subprocess
import re


class GPUArchitecture:
    """GPU architecture specifications database."""

    # Comprehensive GPU specifications database
    GPU_SPECS = {
        # NVIDIA RTX 50 Series (Blackwell)
        'RTX 5090': {
            'architecture': 'Blackwell',
            'compute_capability': 10.0,  # Estimated
            'sm_count': 170,  # Estimated based on rumors
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,  # 100 KB
            'shared_memory_per_block': 49152,  # 48 KB
            'peak_bandwidth_gbps': 1500,  # Estimated
            'peak_tflops_fp32': 125,  # Estimated
            'peak_tflops_fp16': 250,  # Estimated
            'peak_tflops_tf32': 125,  # Estimated
            'tensor_cores': True,
            'tensor_core_generation': 5,
        },
        'RTX 5080': {
            'architecture': 'Blackwell',
            'compute_capability': 10.0,  # Estimated
            'sm_count': 84,  # Estimated
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 960,  # Estimated
            'peak_tflops_fp32': 65,  # Estimated
            'peak_tflops_fp16': 130,  # Estimated
            'peak_tflops_tf32': 65,  # Estimated
            'tensor_cores': True,
            'tensor_core_generation': 5,
        },
        'RTX 5070': {
            'architecture': 'Blackwell',
            'compute_capability': 10.0,  # Estimated
            'sm_count': 48,  # Estimated based on typical 70-series
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 672,  # Estimated (GDDR7)
            'peak_tflops_fp32': 38,  # Estimated
            'peak_tflops_fp16': 76,  # Estimated
            'peak_tflops_tf32': 38,  # Estimated
            'tensor_cores': True,
            'tensor_core_generation': 5,
        },
        'RTX 5070 Ti': {
            'architecture': 'Blackwell',
            'compute_capability': 10.0,  # Estimated
            'sm_count': 60,  # Estimated
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 768,  # Estimated
            'peak_tflops_fp32': 48,  # Estimated
            'peak_tflops_fp16': 96,  # Estimated
            'peak_tflops_tf32': 48,  # Estimated
            'tensor_cores': True,
            'tensor_core_generation': 5,
        },

        # NVIDIA RTX 40 Series (Ada Lovelace)
        'RTX 4090': {
            'architecture': 'Ada Lovelace',
            'compute_capability': 8.9,
            'sm_count': 128,
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,  # 100 KB
            'shared_memory_per_block': 49152,  # 48 KB
            'peak_bandwidth_gbps': 1008,
            'peak_tflops_fp32': 82.6,
            'peak_tflops_fp16': 165.2,
            'peak_tflops_tf32': 82.6,
            'tensor_cores': True,
            'tensor_core_generation': 4,
        },
        'RTX 4080': {
            'architecture': 'Ada Lovelace',
            'compute_capability': 8.9,
            'sm_count': 76,
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 716,
            'peak_tflops_fp32': 48.7,
            'peak_tflops_fp16': 97.4,
            'peak_tflops_tf32': 48.7,
            'tensor_cores': True,
            'tensor_core_generation': 4,
        },
        'RTX 4070': {
            'architecture': 'Ada Lovelace',
            'compute_capability': 8.9,
            'sm_count': 46,
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 504,
            'peak_tflops_fp32': 29.1,
            'peak_tflops_fp16': 58.2,
            'peak_tflops_tf32': 29.1,
            'tensor_cores': True,
            'tensor_core_generation': 4,
        },

        # NVIDIA RTX 30 Series (Ampere)
        'RTX 3090': {
            'architecture': 'Ampere',
            'compute_capability': 8.6,
            'sm_count': 82,
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 936,
            'peak_tflops_fp32': 35.6,
            'peak_tflops_fp16': 71.0,
            'peak_tflops_tf32': 35.6,
            'tensor_cores': True,
            'tensor_core_generation': 3,
        },
        'RTX 3080': {
            'architecture': 'Ampere',
            'compute_capability': 8.6,
            'sm_count': 68,
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 760,
            'peak_tflops_fp32': 29.8,
            'peak_tflops_fp16': 59.5,
            'peak_tflops_tf32': 29.8,
            'tensor_cores': True,
            'tensor_core_generation': 3,
        },
        'RTX 3070': {
            'architecture': 'Ampere',
            'compute_capability': 8.6,
            'sm_count': 46,
            'max_threads_per_sm': 1536,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 102400,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 448,
            'peak_tflops_fp32': 20.3,
            'peak_tflops_fp16': 40.6,
            'peak_tflops_tf32': 20.3,
            'tensor_cores': True,
            'tensor_core_generation': 3,
        },

        # NVIDIA Data Center GPUs
        'A100': {
            'architecture': 'Ampere',
            'compute_capability': 8.0,
            'sm_count': 108,
            'max_threads_per_sm': 2048,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 167936,  # 164 KB
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 1555,
            'peak_tflops_fp32': 19.5,
            'peak_tflops_fp16': 312,
            'peak_tflops_tf32': 156,
            'tensor_cores': True,
            'tensor_core_generation': 3,
        },
        'A100-80GB': {
            'architecture': 'Ampere',
            'compute_capability': 8.0,
            'sm_count': 108,
            'max_threads_per_sm': 2048,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 167936,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 2039,
            'peak_tflops_fp32': 19.5,
            'peak_tflops_fp16': 312,
            'peak_tflops_tf32': 156,
            'tensor_cores': True,
            'tensor_core_generation': 3,
        },
        'H100': {
            'architecture': 'Hopper',
            'compute_capability': 9.0,
            'sm_count': 132,
            'max_threads_per_sm': 2048,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 228352,  # 223 KB
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 3350,
            'peak_tflops_fp32': 67,
            'peak_tflops_fp16': 1979,
            'peak_tflops_tf32': 989,
            'tensor_cores': True,
            'tensor_core_generation': 4,
        },
        'V100': {
            'architecture': 'Volta',
            'compute_capability': 7.0,
            'sm_count': 80,
            'max_threads_per_sm': 2048,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 98304,  # 96 KB
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 900,
            'peak_tflops_fp32': 15.7,
            'peak_tflops_fp16': 125,
            'peak_tflops_tf32': 0,  # No TF32 support
            'tensor_cores': True,
            'tensor_core_generation': 1,
        },
        'T4': {
            'architecture': 'Turing',
            'compute_capability': 7.5,
            'sm_count': 40,
            'max_threads_per_sm': 1024,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 65536,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 320,
            'peak_tflops_fp32': 8.1,
            'peak_tflops_fp16': 65,
            'peak_tflops_tf32': 0,
            'tensor_cores': True,
            'tensor_core_generation': 2,
        },

        # NVIDIA RTX 20 Series (Turing)
        'RTX 2080 Ti': {
            'architecture': 'Turing',
            'compute_capability': 7.5,
            'sm_count': 68,
            'max_threads_per_sm': 1024,
            'max_threads_per_block': 1024,
            'max_blocks_per_sm': 16,
            'registers_per_sm': 65536,
            'shared_memory_per_sm': 65536,
            'shared_memory_per_block': 49152,
            'peak_bandwidth_gbps': 616,
            'peak_tflops_fp32': 13.4,
            'peak_tflops_fp16': 26.9,
            'peak_tflops_tf32': 0,
            'tensor_cores': True,
            'tensor_core_generation': 2,
        },
    }

    @classmethod
    def get_specs(cls, gpu_name: str) -> Optional[Dict[str, any]]:
        """
        Get specifications for a GPU model.

        Args:
            gpu_name: GPU model name (e.g., 'RTX 4090', 'A100')

        Returns:
            Dictionary of GPU specifications or None if not found
        """
        # Normalize GPU name
        gpu_name_normalized = gpu_name.strip()

        # Try exact match first
        if gpu_name_normalized in cls.GPU_SPECS:
            return cls.GPU_SPECS[gpu_name_normalized].copy()

        # Try partial match
        for key in cls.GPU_SPECS.keys():
            if gpu_name_normalized.lower() in key.lower() or key.lower() in gpu_name_normalized.lower():
                return cls.GPU_SPECS[key].copy()

        return None

    @classmethod
    def detect_current_gpu(cls) -> Optional[Tuple[str, Dict[str, any]]]:
        """
        Detect current GPU and return its specifications.

        Returns:
            Tuple of (gpu_name, specs_dict) or None if detection fails
        """
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                specs = cls.get_specs(gpu_name)
                if specs:
                    return (gpu_name, specs)

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    @classmethod
    def get_compute_capability(cls, gpu_name: str) -> Optional[float]:
        """Get compute capability for a GPU."""
        specs = cls.get_specs(gpu_name)
        return specs['compute_capability'] if specs else None

    @classmethod
    def supports_tensor_cores(cls, gpu_name: str) -> bool:
        """Check if GPU supports Tensor Cores."""
        specs = cls.get_specs(gpu_name)
        return specs.get('tensor_cores', False) if specs else False

    @classmethod
    def get_optimal_block_size(cls, gpu_name: str, registers_per_thread: int = 32) -> int:
        """
        Calculate optimal block size for a GPU.

        Args:
            gpu_name: GPU model name
            registers_per_thread: Estimated registers per thread

        Returns:
            Recommended block size
        """
        specs = cls.get_specs(gpu_name)
        if not specs:
            return 256  # Default fallback

        max_threads_per_sm = specs['max_threads_per_sm']
        registers_per_sm = specs['registers_per_sm']

        # Calculate max threads limited by registers
        max_threads_by_regs = registers_per_sm // registers_per_thread

        # Calculate optimal block size (power of 2, multiple of warp size)
        optimal = min(max_threads_per_sm, max_threads_by_regs)
        optimal = min(optimal, specs['max_threads_per_block'])

        # Round down to nearest power of 2
        import math
        optimal = 2 ** int(math.log2(optimal))

        # Ensure it's at least 128 (4 warps)
        optimal = max(optimal, 128)

        return optimal

    @classmethod
    def calculate_occupancy(cls,
                           gpu_name: str,
                           block_size: int,
                           registers_per_thread: int,
                           shared_memory_per_block: int) -> float:
        """
        Calculate theoretical occupancy for given kernel configuration.

        Args:
            gpu_name: GPU model name
            block_size: Threads per block
            registers_per_thread: Registers used per thread
            shared_memory_per_block: Shared memory used per block (bytes)

        Returns:
            Theoretical occupancy (0.0 to 1.0)
        """
        specs = cls.get_specs(gpu_name)
        if not specs:
            return 0.0

        max_threads_per_sm = specs['max_threads_per_sm']
        max_blocks_per_sm = specs['max_blocks_per_sm']
        registers_per_sm = specs['registers_per_sm']
        shared_memory_per_sm = specs['shared_memory_per_sm']

        # Calculate limits
        blocks_by_threads = max_threads_per_sm // block_size
        blocks_by_regs = registers_per_sm // (registers_per_thread * block_size) if registers_per_thread > 0 else 999
        blocks_by_smem = shared_memory_per_sm // shared_memory_per_block if shared_memory_per_block > 0 else 999

        # Actual blocks per SM
        blocks_per_sm = min(blocks_by_threads, blocks_by_regs, blocks_by_smem, max_blocks_per_sm)

        # Occupancy
        occupancy = (blocks_per_sm * block_size) / max_threads_per_sm

        return min(occupancy, 1.0)


def detect_gpu() -> Optional[Tuple[str, Dict[str, any]]]:
    """
    Convenience function to detect current GPU.

    Returns:
        Tuple of (gpu_name, specs_dict) or None if detection fails
    """
    return GPUArchitecture.detect_current_gpu()


def get_gpu_specs(gpu_name: str) -> Optional[Dict[str, any]]:
    """
    Convenience function to get GPU specifications.

    Args:
        gpu_name: GPU model name

    Returns:
        Dictionary of GPU specifications or None if not found
    """
    return GPUArchitecture.get_specs(gpu_name)


def calculate_optimal_config(gpu_name: str,
                             registers_per_thread: int = 32,
                             shared_memory_per_block: int = 0) -> Dict[str, any]:
    """
    Calculate optimal kernel configuration for a GPU.

    Args:
        gpu_name: GPU model name
        registers_per_thread: Estimated registers per thread
        shared_memory_per_block: Shared memory per block (bytes)

    Returns:
        Dictionary with recommended configuration
    """
    specs = GPUArchitecture.get_specs(gpu_name)
    if not specs:
        return {
            'block_size': 256,
            'grid_size': 'auto',
            'occupancy': 0.0,
            'error': 'GPU not found in database'
        }

    # Calculate optimal block size
    block_size = GPUArchitecture.get_optimal_block_size(gpu_name, registers_per_thread)

    # Calculate occupancy
    occupancy = GPUArchitecture.calculate_occupancy(
        gpu_name, block_size, registers_per_thread, shared_memory_per_block
    )

    return {
        'block_size': block_size,
        'grid_size': 'auto',  # Depends on problem size
        'occupancy': occupancy,
        'max_blocks_per_sm': specs['max_blocks_per_sm'],
        'shared_memory_available': specs['shared_memory_per_block'],
        'registers_available': specs['registers_per_sm'] // block_size,
    }
