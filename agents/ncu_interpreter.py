"""
NCU Interpreter - Translates raw NCU metrics into high-level performance diagnostics.

This module analyzes NVIDIA Nsight Compute profiling data and generates actionable
insights for LLM-based optimization. It computes derived metrics, identifies bottlenecks,
and performs Roofline analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class PerformanceDiagnosis:
    """Complete performance diagnosis result."""

    # Primary bottleneck
    bottleneck: str  # 'memory_bandwidth', 'memory_latency', 'compute_bound', 'occupancy'
    bottleneck_confidence: float  # 0.0 to 1.0

    # Memory subsystem analysis
    memory_bandwidth_util: float  # Percentage of peak bandwidth utilized
    memory_access_pattern: str  # 'coalesced', 'strided', 'random', 'mixed'
    l1_cache_hit_rate: Optional[float]
    l2_cache_hit_rate: Optional[float]
    global_load_efficiency: Optional[float]
    global_store_efficiency: Optional[float]

    # Compute subsystem analysis
    achieved_occupancy: float  # Percentage
    theoretical_occupancy: float  # Percentage
    occupancy_limiting_factor: str  # 'registers', 'shared_memory', 'block_size', 'none'
    sm_efficiency: float  # Percentage of SM cycles doing useful work
    warp_execution_efficiency: Optional[float]

    # Roofline analysis
    arithmetic_intensity: float  # FLOPs per byte
    roofline_region: str  # 'memory_bound', 'compute_bound', 'balanced'
    distance_to_roofline: float  # How far from theoretical peak (0.0 to 1.0)

    # Prioritized issues
    issues: List[Dict[str, any]]  # List of {severity, category, description, suggestion}

    # Raw summary for reference
    summary: str


class NCUInterpreter:
    """Interprets NCU profiling metrics and generates performance diagnostics."""

    def __init__(self, gpu_specs: Dict[str, float]):
        """
        Initialize interpreter with GPU specifications.

        Args:
            gpu_specs: Dictionary containing:
                - peak_bandwidth_gbps: Peak memory bandwidth in GB/s
                - peak_tflops_fp32: Peak FP32 compute in TFLOPS
                - peak_tflops_fp16: Peak FP16 compute in TFLOPS (optional)
                - sm_count: Number of streaming multiprocessors
                - max_threads_per_sm: Maximum threads per SM
        """
        self.gpu_specs = gpu_specs
        self.peak_bandwidth = gpu_specs['peak_bandwidth_gbps']
        self.peak_flops_fp32 = gpu_specs['peak_tflops_fp32'] * 1e12
        self.peak_flops_fp16 = gpu_specs.get('peak_tflops_fp16', 0) * 1e12
        self.sm_count = gpu_specs['sm_count']
        self.max_threads_per_sm = gpu_specs['max_threads_per_sm']

    def interpret(self, metrics: Dict[str, float], kernel_info: Optional[Dict] = None) -> PerformanceDiagnosis:
        """
        Interpret NCU metrics and generate performance diagnosis.

        Args:
            metrics: Dictionary of NCU metrics (metric_name -> value)
            kernel_info: Optional kernel information (grid_size, block_size, etc.)

        Returns:
            PerformanceDiagnosis object with complete analysis
        """
        # Compute derived metrics
        derived = self._compute_derived_metrics(metrics, kernel_info)

        # Analyze memory subsystem
        memory_analysis = self._analyze_memory(metrics, derived)

        # Analyze compute subsystem
        compute_analysis = self._analyze_compute(metrics, derived, kernel_info)

        # Perform Roofline analysis
        roofline = self._analyze_roofline(derived, memory_analysis, compute_analysis)

        # Identify primary bottleneck
        bottleneck, confidence = self._identify_bottleneck(
            memory_analysis, compute_analysis, roofline
        )

        # Generate prioritized issues and suggestions
        issues = self._generate_issues(
            bottleneck, memory_analysis, compute_analysis, roofline
        )

        # Create summary
        summary = self._create_summary(
            bottleneck, memory_analysis, compute_analysis, roofline
        )

        return PerformanceDiagnosis(
            bottleneck=bottleneck,
            bottleneck_confidence=confidence,
            memory_bandwidth_util=memory_analysis['bandwidth_util'],
            memory_access_pattern=memory_analysis['access_pattern'],
            l1_cache_hit_rate=memory_analysis.get('l1_hit_rate'),
            l2_cache_hit_rate=memory_analysis.get('l2_hit_rate'),
            global_load_efficiency=memory_analysis.get('load_efficiency'),
            global_store_efficiency=memory_analysis.get('store_efficiency'),
            achieved_occupancy=compute_analysis['achieved_occupancy'],
            theoretical_occupancy=compute_analysis['theoretical_occupancy'],
            occupancy_limiting_factor=compute_analysis['limiting_factor'],
            sm_efficiency=compute_analysis['sm_efficiency'],
            warp_execution_efficiency=compute_analysis.get('warp_efficiency'),
            arithmetic_intensity=roofline['arithmetic_intensity'],
            roofline_region=roofline['region'],
            distance_to_roofline=roofline['distance_to_peak'],
            issues=issues,
            summary=summary
        )

    def _compute_derived_metrics(self, metrics: Dict[str, float],
                                 kernel_info: Optional[Dict]) -> Dict[str, float]:
        """Compute derived metrics from raw NCU data."""
        derived = {}

        # Memory bandwidth utilization
        if 'dram__bytes.sum' in metrics and 'duration' in metrics:
            bytes_transferred = metrics['dram__bytes.sum']
            duration_ns = metrics['duration']
            achieved_bw_gbps = (bytes_transferred / duration_ns)  # GB/s
            derived['bandwidth_util'] = (achieved_bw_gbps / self.peak_bandwidth) * 100
            derived['achieved_bandwidth'] = achieved_bw_gbps
        else:
            derived['bandwidth_util'] = 0.0
            derived['achieved_bandwidth'] = 0.0

        # Arithmetic intensity (FLOPs per byte)
        flops = metrics.get('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum', 0) + \
                metrics.get('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum', 0) + \
                metrics.get('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum', 0) * 2

        bytes_transferred = metrics.get('dram__bytes.sum', 1)
        derived['arithmetic_intensity'] = flops / max(bytes_transferred, 1)
        derived['total_flops'] = flops

        # Compute throughput
        if 'duration' in metrics and flops > 0:
            duration_s = metrics['duration'] * 1e-9
            derived['achieved_tflops'] = (flops / duration_s) / 1e12
            derived['compute_util'] = (derived['achieved_tflops'] * 1e12 / self.peak_flops_fp32) * 100
        else:
            derived['achieved_tflops'] = 0.0
            derived['compute_util'] = 0.0

        return derived

    def _analyze_memory(self, metrics: Dict[str, float],
                       derived: Dict[str, float]) -> Dict[str, any]:
        """Analyze memory subsystem performance."""
        analysis = {
            'bandwidth_util': derived['bandwidth_util'],
            'achieved_bandwidth': derived['achieved_bandwidth']
        }

        # Cache hit rates
        if 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum' in metrics:
            l1_requests = metrics['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum']
            l1_hits = metrics.get('l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit.sum', 0)
            analysis['l1_hit_rate'] = (l1_hits / max(l1_requests, 1)) * 100

        if 'lts__t_sectors_op_read.sum' in metrics:
            l2_requests = metrics['lts__t_sectors_op_read.sum']
            l2_hits = metrics.get('lts__t_sectors_op_read_hit.sum', 0)
            analysis['l2_hit_rate'] = (l2_hits / max(l2_requests, 1)) * 100

        # Memory efficiency
        if 'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct' in metrics:
            analysis['load_efficiency'] = metrics['smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct']

        if 'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct' in metrics:
            analysis['store_efficiency'] = metrics['smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct']

        # Infer access pattern
        load_eff = analysis.get('load_efficiency', 100)
        store_eff = analysis.get('store_efficiency', 100)
        avg_eff = (load_eff + store_eff) / 2

        if avg_eff > 80:
            analysis['access_pattern'] = 'coalesced'
        elif avg_eff > 50:
            analysis['access_pattern'] = 'strided'
        elif avg_eff > 25:
            analysis['access_pattern'] = 'mixed'
        else:
            analysis['access_pattern'] = 'random'

        # Diagnose memory issues
        issues = []
        if analysis['bandwidth_util'] < 30:
            issues.append('Low bandwidth utilization - kernel may be compute-bound or have insufficient parallelism')
        elif analysis['bandwidth_util'] > 80:
            issues.append('High bandwidth utilization - memory bandwidth is likely the bottleneck')

        if load_eff < 50:
            issues.append('Poor load efficiency - memory accesses are not coalesced')
        if store_eff < 50:
            issues.append('Poor store efficiency - memory writes are not coalesced')

        l1_hit = analysis.get('l1_hit_rate', 100)
        if l1_hit < 50:
            issues.append('Low L1 cache hit rate - poor data locality')

        analysis['issues'] = issues

        return analysis

    def _analyze_compute(self, metrics: Dict[str, float],
                        derived: Dict[str, float],
                        kernel_info: Optional[Dict]) -> Dict[str, any]:
        """Analyze compute subsystem performance."""
        analysis = {}

        # Occupancy
        achieved_occ = metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0)
        analysis['achieved_occupancy'] = achieved_occ

        # Theoretical occupancy (if kernel info available)
        if kernel_info:
            block_size = kernel_info.get('block_size', 256)
            registers_per_thread = kernel_info.get('registers_per_thread', 32)
            shared_mem_per_block = kernel_info.get('shared_mem_per_block', 0)

            # Calculate theoretical occupancy based on resource limits
            max_blocks_by_threads = self.max_threads_per_sm // block_size
            max_blocks_by_regs = (65536 // (registers_per_thread * block_size)) if registers_per_thread > 0 else 999
            max_blocks_by_smem = (49152 // shared_mem_per_block) if shared_mem_per_block > 0 else 999

            max_blocks = min(max_blocks_by_threads, max_blocks_by_regs, max_blocks_by_smem, 16)
            theoretical_occ = (max_blocks * block_size / self.max_threads_per_sm) * 100
            analysis['theoretical_occupancy'] = min(theoretical_occ, 100)

            # Identify limiting factor
            if max_blocks == max_blocks_by_regs:
                analysis['limiting_factor'] = 'registers'
            elif max_blocks == max_blocks_by_smem:
                analysis['limiting_factor'] = 'shared_memory'
            elif max_blocks == max_blocks_by_threads:
                analysis['limiting_factor'] = 'block_size'
            else:
                analysis['limiting_factor'] = 'none'
        else:
            analysis['theoretical_occupancy'] = 100
            analysis['limiting_factor'] = 'unknown'

        # SM efficiency
        sm_active = metrics.get('smsp__cycles_active.avg.pct_of_peak_sustained_elapsed', 0)
        analysis['sm_efficiency'] = sm_active

        # Warp execution efficiency
        if 'smsp__thread_inst_executed_per_inst_executed.ratio' in metrics:
            analysis['warp_efficiency'] = metrics['smsp__thread_inst_executed_per_inst_executed.ratio'] * 100

        # Compute utilization
        analysis['compute_util'] = derived['compute_util']

        # Diagnose compute issues
        issues = []
        if achieved_occ < 30:
            issues.append(f'Low occupancy ({achieved_occ:.1f}%) - limited by {analysis["limiting_factor"]}')

        if sm_active < 50:
            issues.append('Low SM efficiency - SMs are idle for significant time')

        warp_eff = analysis.get('warp_efficiency', 100)
        if warp_eff < 80:
            issues.append('Low warp efficiency - thread divergence detected')

        if derived['compute_util'] > 80:
            issues.append('High compute utilization - kernel is compute-bound')

        analysis['issues'] = issues

        return analysis

    def _analyze_roofline(self, derived: Dict[str, float],
                         memory_analysis: Dict[str, any],
                         compute_analysis: Dict[str, any]) -> Dict[str, any]:
        """Perform Roofline model analysis."""
        ai = derived['arithmetic_intensity']
        achieved_bw = derived['achieved_bandwidth']
        achieved_flops = derived['achieved_tflops'] * 1e12

        # Calculate ridge point (where memory and compute roofs intersect)
        ridge_point = self.peak_flops_fp32 / (self.peak_bandwidth * 1e9)  # FLOPs per byte

        # Determine region
        if ai < ridge_point * 0.5:
            region = 'memory_bound'
            theoretical_peak = achieved_bw * 1e9 * ai  # Bandwidth-limited
        elif ai > ridge_point * 2.0:
            region = 'compute_bound'
            theoretical_peak = self.peak_flops_fp32
        else:
            region = 'balanced'
            theoretical_peak = min(achieved_bw * 1e9 * ai, self.peak_flops_fp32)

        # Distance to roofline (efficiency)
        distance = achieved_flops / max(theoretical_peak, 1)

        return {
            'arithmetic_intensity': ai,
            'region': region,
            'ridge_point': ridge_point,
            'distance_to_peak': distance,
            'theoretical_peak_flops': theoretical_peak,
            'achieved_flops': achieved_flops
        }

    def _identify_bottleneck(self, memory_analysis: Dict[str, any],
                            compute_analysis: Dict[str, any],
                            roofline: Dict[str, any]) -> Tuple[str, float]:
        """Identify primary bottleneck with confidence score."""

        bw_util = memory_analysis['bandwidth_util']
        compute_util = compute_analysis['compute_util']
        occupancy = compute_analysis['achieved_occupancy']
        region = roofline['region']

        # Score each potential bottleneck
        scores = {
            'memory_bandwidth': 0.0,
            'memory_latency': 0.0,
            'compute_bound': 0.0,
            'occupancy': 0.0
        }

        # Memory bandwidth bottleneck indicators
        if bw_util > 70:
            scores['memory_bandwidth'] += 0.4
        if region == 'memory_bound':
            scores['memory_bandwidth'] += 0.3
        if memory_analysis['access_pattern'] in ['coalesced', 'strided']:
            scores['memory_bandwidth'] += 0.2

        # Memory latency bottleneck indicators
        if bw_util < 40 and occupancy < 50:
            scores['memory_latency'] += 0.4
        if memory_analysis['access_pattern'] in ['random', 'mixed']:
            scores['memory_latency'] += 0.3
        l1_hit = memory_analysis.get('l1_hit_rate', 100)
        if l1_hit < 50:
            scores['memory_latency'] += 0.2

        # Compute bound indicators
        if compute_util > 70:
            scores['compute_bound'] += 0.4
        if region == 'compute_bound':
            scores['compute_bound'] += 0.3
        if roofline['arithmetic_intensity'] > roofline['ridge_point']:
            scores['compute_bound'] += 0.2

        # Occupancy bottleneck indicators
        if occupancy < 30:
            scores['occupancy'] += 0.5
        if compute_analysis['limiting_factor'] in ['registers', 'shared_memory']:
            scores['occupancy'] += 0.3

        # Find primary bottleneck
        bottleneck = max(scores, key=scores.get)
        confidence = scores[bottleneck]

        # Normalize confidence to 0-1 range
        confidence = min(confidence, 1.0)

        return bottleneck, confidence

    def _generate_issues(self, bottleneck: str,
                        memory_analysis: Dict[str, any],
                        compute_analysis: Dict[str, any],
                        roofline: Dict[str, any]) -> List[Dict[str, any]]:
        """Generate prioritized list of issues and suggestions."""
        issues = []

        # Add bottleneck as highest priority issue
        if bottleneck == 'memory_bandwidth':
            issues.append({
                'severity': 'high',
                'category': 'memory',
                'description': f'Memory bandwidth bottleneck detected ({memory_analysis["bandwidth_util"]:.1f}% utilization)',
                'suggestion': 'Consider: 1) Reduce memory traffic via shared memory, 2) Increase arithmetic intensity, 3) Use texture memory for read-only data'
            })
        elif bottleneck == 'memory_latency':
            issues.append({
                'severity': 'high',
                'category': 'memory',
                'description': f'Memory latency bottleneck with {memory_analysis["access_pattern"]} access pattern',
                'suggestion': 'Consider: 1) Improve memory coalescing, 2) Increase occupancy to hide latency, 3) Use shared memory for frequently accessed data'
            })
        elif bottleneck == 'compute_bound':
            issues.append({
                'severity': 'high',
                'category': 'compute',
                'description': f'Compute bottleneck detected ({compute_analysis["compute_util"]:.1f}% utilization)',
                'suggestion': 'Consider: 1) Use specialized hardware (Tensor Cores), 2) Optimize instruction mix, 3) Reduce register pressure'
            })
        elif bottleneck == 'occupancy':
            issues.append({
                'severity': 'high',
                'category': 'occupancy',
                'description': f'Low occupancy ({compute_analysis["achieved_occupancy"]:.1f}%) limited by {compute_analysis["limiting_factor"]}',
                'suggestion': 'Consider: 1) Reduce register usage, 2) Reduce shared memory usage, 3) Adjust block size'
            })

        # Add secondary issues from subsystem analyses
        for issue_text in memory_analysis.get('issues', []):
            issues.append({
                'severity': 'medium',
                'category': 'memory',
                'description': issue_text,
                'suggestion': ''
            })

        for issue_text in compute_analysis.get('issues', []):
            issues.append({
                'severity': 'medium',
                'category': 'compute',
                'description': issue_text,
                'suggestion': ''
            })

        # Add Roofline insights
        if roofline['distance_to_peak'] < 0.5:
            issues.append({
                'severity': 'medium',
                'category': 'efficiency',
                'description': f'Kernel is only {roofline["distance_to_peak"]*100:.1f}% efficient relative to Roofline model',
                'suggestion': f'Focus on {roofline["region"]} optimizations to approach theoretical peak'
            })

        return issues

    def _create_summary(self, bottleneck: str,
                       memory_analysis: Dict[str, any],
                       compute_analysis: Dict[str, any],
                       roofline: Dict[str, any]) -> str:
        """Create human-readable summary of diagnosis."""
        summary_parts = []

        # Bottleneck summary
        summary_parts.append(f"Primary bottleneck: {bottleneck.replace('_', ' ').title()}")

        # Memory summary
        summary_parts.append(
            f"Memory: {memory_analysis['bandwidth_util']:.1f}% bandwidth utilization, "
            f"{memory_analysis['access_pattern']} access pattern"
        )

        # Compute summary
        summary_parts.append(
            f"Compute: {compute_analysis['achieved_occupancy']:.1f}% occupancy, "
            f"{compute_analysis['sm_efficiency']:.1f}% SM efficiency"
        )

        # Roofline summary
        summary_parts.append(
            f"Roofline: {roofline['region'].replace('_', ' ')}, "
            f"AI={roofline['arithmetic_intensity']:.2f} FLOPs/byte, "
            f"{roofline['distance_to_peak']*100:.1f}% of theoretical peak"
        )

        return " | ".join(summary_parts)


def create_interpreter_for_gpu(gpu_name: str) -> NCUInterpreter:
    """
    Factory function to create NCU interpreter for specific GPU.

    Args:
        gpu_name: GPU model name (e.g., 'RTX 4090', 'A100', 'V100')

    Returns:
        Configured NCUInterpreter instance
    """
    # GPU specifications database
    gpu_specs_db = {
        'RTX 5070': {
            'peak_bandwidth_gbps': 672,  # 21 Gbps × 192-bit / 8
            'peak_tflops_fp32': 35.1,
            'peak_tflops_fp16': 281,  # With Tensor Cores
            'sm_count': 48,
            'max_threads_per_sm': 1536
        },
        'RTX 4090': {
            'peak_bandwidth_gbps': 1008,
            'peak_tflops_fp32': 82.6,
            'peak_tflops_fp16': 165.2,
            'sm_count': 128,
            'max_threads_per_sm': 1536
        },
        'RTX 3090': {
            'peak_bandwidth_gbps': 936,
            'peak_tflops_fp32': 35.6,
            'peak_tflops_fp16': 71.0,
            'sm_count': 82,
            'max_threads_per_sm': 1536
        },
        'A100': {
            'peak_bandwidth_gbps': 1555,
            'peak_tflops_fp32': 19.5,
            'peak_tflops_fp16': 312,
            'sm_count': 108,
            'max_threads_per_sm': 2048
        },
        'V100': {
            'peak_bandwidth_gbps': 900,
            'peak_tflops_fp32': 15.7,
            'peak_tflops_fp16': 125,
            'sm_count': 80,
            'max_threads_per_sm': 2048
        },
        'H100': {
            'peak_bandwidth_gbps': 3350,
            'peak_tflops_fp32': 67,
            'peak_tflops_fp16': 1979,
            'sm_count': 132,
            'max_threads_per_sm': 2048
        }
    }

    if gpu_name not in gpu_specs_db:
        raise ValueError(f"Unknown GPU: {gpu_name}. Supported GPUs: {list(gpu_specs_db.keys())}")

    return NCUInterpreter(gpu_specs_db[gpu_name])
