"""
Operator Type Detection - Automatically detect CUDA kernel operator type.

This module analyzes kernel code and metadata to determine the operator type
(matmul, conv2d, elementwise, reduction, etc.).
"""

import re
from typing import Optional, Dict, List
from agents.strategy_templates import OperatorType


class OperatorDetector:
    """Detects operator type from kernel code and metadata."""

    def __init__(self):
        # Keyword patterns for each operator type
        self.patterns = {
            OperatorType.MATMUL: [
                r'matmul|gemm|matrix.*mult|mm_kernel',
                r'__shared__.*\[.*\]\[.*\].*__shared__.*\[.*\]\[.*\]',  # Two shared memory arrays
                r'for.*\(.*k.*<.*K.*\)',  # K-dimension loop
                r'A\[.*\*.*K.*\].*B\[.*\*.*N.*\]',  # Matrix indexing pattern
            ],
            OperatorType.CONV2D: [
                r'conv|convolution',
                r'filter|kernel_size|stride|padding',
                r'input.*height.*width|output.*height.*width',
                r'channel|in_channels|out_channels',
            ],
            OperatorType.ELEMENTWISE: [
                r'elementwise|element_wise',
                r'relu|sigmoid|tanh|gelu|silu',
                r'add|sub|mul|div.*element',
                r'^\s*output\[.*\]\s*=\s*.*input\[.*\]',  # Simple element mapping
            ],
            OperatorType.REDUCTION: [
                r'reduce|reduction|sum|mean|max|min',
                r'atomicAdd|atomicMax|atomicMin',
                r'__shfl_down|__shfl_xor',  # Warp shuffle for reduction
                r'warp.*reduce|block.*reduce',
            ],
            OperatorType.TRANSPOSE: [
                r'transpose|permute',
                r'output\[.*j.*\]\[.*i.*\].*input\[.*i.*\]\[.*j.*\]',  # Index swap pattern
            ],
            OperatorType.SOFTMAX: [
                r'softmax',
                r'exp.*sum|expf.*sum',
                r'max.*exp.*sum',  # Softmax pattern: max -> exp -> sum -> divide
            ],
            OperatorType.LAYERNORM: [
                r'layer.*norm|layernorm',
                r'mean.*variance|variance.*mean',
                r'rsqrt|1\.0.*sqrt',  # Normalization pattern
            ],
        }

    def detect_from_code(self, kernel_code: str) -> OperatorType:
        """
        Detect operator type from kernel code.

        Args:
            kernel_code: CUDA kernel source code

        Returns:
            Detected OperatorType
        """
        # Normalize code (lowercase, remove extra whitespace)
        code_lower = kernel_code.lower()
        code_normalized = re.sub(r'\s+', ' ', code_lower)

        # Score each operator type
        scores = {op_type: 0 for op_type in OperatorType}

        for op_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_normalized, re.IGNORECASE):
                    scores[op_type] += 1

        # Find operator type with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return OperatorType.UNKNOWN

        # Return operator type with highest score
        for op_type, score in scores.items():
            if score == max_score:
                return op_type

        return OperatorType.UNKNOWN

    def detect_from_filename(self, filename: str) -> Optional[OperatorType]:
        """
        Detect operator type from filename.

        Args:
            filename: Kernel filename or path

        Returns:
            Detected OperatorType or None if cannot determine
        """
        filename_lower = filename.lower()

        if 'matmul' in filename_lower or 'gemm' in filename_lower or 'mm' in filename_lower:
            return OperatorType.MATMUL
        elif 'conv' in filename_lower:
            return OperatorType.CONV2D
        elif 'elementwise' in filename_lower or 'eltwise' in filename_lower:
            return OperatorType.ELEMENTWISE
        elif 'reduce' in filename_lower or 'reduction' in filename_lower:
            return OperatorType.REDUCTION
        elif 'transpose' in filename_lower:
            return OperatorType.TRANSPOSE
        elif 'softmax' in filename_lower:
            return OperatorType.SOFTMAX
        elif 'layernorm' in filename_lower or 'layer_norm' in filename_lower:
            return OperatorType.LAYERNORM

        return None

    def detect_from_signature(self, kernel_signature: str) -> Optional[OperatorType]:
        """
        Detect operator type from kernel function signature.

        Args:
            kernel_signature: Kernel function signature (e.g., "matmul(float* A, float* B, float* C, int M, int N, int K)")

        Returns:
            Detected OperatorType or None if cannot determine
        """
        sig_lower = kernel_signature.lower()

        # Check for matrix multiplication signature (A, B, C with M, N, K dimensions)
        if re.search(r'\bm\b.*\bn\b.*\bk\b', sig_lower) or \
           re.search(r'matrix.*matrix', sig_lower):
            return OperatorType.MATMUL

        # Check for convolution signature
        if re.search(r'height.*width.*channel', sig_lower) or \
           re.search(r'filter.*stride.*padding', sig_lower):
            return OperatorType.CONV2D

        # Check for reduction signature (input array, output scalar/small array)
        if re.search(r'reduce|sum|mean', sig_lower):
            return OperatorType.REDUCTION

        return None

    def detect(self,
               kernel_code: str,
               filename: Optional[str] = None,
               kernel_signature: Optional[str] = None,
               metadata: Optional[Dict] = None) -> OperatorType:
        """
        Detect operator type using all available information.

        Args:
            kernel_code: CUDA kernel source code
            filename: Optional kernel filename
            kernel_signature: Optional kernel function signature
            metadata: Optional metadata dictionary

        Returns:
            Detected OperatorType
        """
        # Try filename first (most reliable if named well)
        if filename:
            op_type = self.detect_from_filename(filename)
            if op_type:
                return op_type

        # Try signature
        if kernel_signature:
            op_type = self.detect_from_signature(kernel_signature)
            if op_type:
                return op_type

        # Check metadata
        if metadata and 'operator_type' in metadata:
            try:
                return OperatorType(metadata['operator_type'])
            except ValueError:
                pass

        # Finally, analyze code
        return self.detect_from_code(kernel_code)

    def get_operator_characteristics(self, op_type: OperatorType) -> Dict[str, any]:
        """
        Get typical characteristics of an operator type.

        Args:
            op_type: Operator type

        Returns:
            Dictionary with operator characteristics
        """
        characteristics = {
            OperatorType.MATMUL: {
                'compute_intensity': 'high',
                'memory_pattern': 'structured',
                'typical_bottleneck': 'memory_bandwidth',
                'optimization_priority': ['tiling', 'tensor_cores', 'vectorization'],
                'arithmetic_intensity_range': (10, 100),  # FLOPs per byte
            },
            OperatorType.CONV2D: {
                'compute_intensity': 'high',
                'memory_pattern': 'structured',
                'typical_bottleneck': 'memory_bandwidth',
                'optimization_priority': ['im2col', 'winograd', 'tensor_cores'],
                'arithmetic_intensity_range': (5, 50),
            },
            OperatorType.ELEMENTWISE: {
                'compute_intensity': 'low',
                'memory_pattern': 'simple',
                'typical_bottleneck': 'memory_bandwidth',
                'optimization_priority': ['vectorization', 'fusion', 'coalescing'],
                'arithmetic_intensity_range': (0.1, 2),
            },
            OperatorType.REDUCTION: {
                'compute_intensity': 'low',
                'memory_pattern': 'irregular',
                'typical_bottleneck': 'memory_latency',
                'optimization_priority': ['warp_primitives', 'tree_reduction', 'occupancy'],
                'arithmetic_intensity_range': (0.5, 5),
            },
            OperatorType.TRANSPOSE: {
                'compute_intensity': 'very_low',
                'memory_pattern': 'strided',
                'typical_bottleneck': 'memory_bandwidth',
                'optimization_priority': ['shared_memory', 'bank_conflict_avoidance', 'coalescing'],
                'arithmetic_intensity_range': (0.01, 0.1),
            },
            OperatorType.SOFTMAX: {
                'compute_intensity': 'medium',
                'memory_pattern': 'structured',
                'typical_bottleneck': 'memory_latency',
                'optimization_priority': ['warp_primitives', 'shared_memory', 'fusion'],
                'arithmetic_intensity_range': (2, 10),
            },
            OperatorType.LAYERNORM: {
                'compute_intensity': 'medium',
                'memory_pattern': 'structured',
                'typical_bottleneck': 'memory_latency',
                'optimization_priority': ['warp_primitives', 'fusion', 'vectorization'],
                'arithmetic_intensity_range': (3, 15),
            },
            OperatorType.UNKNOWN: {
                'compute_intensity': 'unknown',
                'memory_pattern': 'unknown',
                'typical_bottleneck': 'unknown',
                'optimization_priority': [],
                'arithmetic_intensity_range': (0, 1000),
            },
        }

        return characteristics.get(op_type, characteristics[OperatorType.UNKNOWN])


def detect_operator_type(kernel_code: str,
                        filename: Optional[str] = None,
                        kernel_signature: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> OperatorType:
    """
    Convenience function to detect operator type.

    Args:
        kernel_code: CUDA kernel source code
        filename: Optional kernel filename
        kernel_signature: Optional kernel function signature
        metadata: Optional metadata dictionary

    Returns:
        Detected OperatorType
    """
    detector = OperatorDetector()
    return detector.detect(kernel_code, filename, kernel_signature, metadata)


def get_operator_info(op_type: OperatorType) -> Dict[str, any]:
    """
    Convenience function to get operator characteristics.

    Args:
        op_type: Operator type

    Returns:
        Dictionary with operator characteristics
    """
    detector = OperatorDetector()
    return detector.get_operator_characteristics(op_type)
