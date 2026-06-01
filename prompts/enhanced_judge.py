"""
Enhanced Judge Prompt - Uses structured performance diagnosis instead of raw NCU metrics.

This module builds prompts for the Judge LLM that include high-level performance
insights and prioritized recommendations from the NCU Interpreter.
"""

from typing import Dict, List, Optional
from agents.ncu_interpreter import PerformanceDiagnosis


def build_enhanced_judge_prompt(
    diagnosis: PerformanceDiagnosis,
    operator_type: str,
    current_code: str,
    optimization_history: Optional[List[Dict]] = None
) -> str:
    """
    Build enhanced judge prompt with structured performance diagnosis.

    Args:
        diagnosis: Performance diagnosis from NCU Interpreter
        operator_type: Type of operator (matmul, conv, reduction, etc.)
        current_code: Current kernel code
        optimization_history: Previous optimization attempts

    Returns:
        Formatted prompt string for Judge LLM
    """

    # Build performance summary
    perf_summary = _build_performance_summary(diagnosis)

    # Build bottleneck analysis
    bottleneck_analysis = _build_bottleneck_analysis(diagnosis)

    # Build optimization recommendations
    recommendations = _build_recommendations(diagnosis, operator_type)

    # Build history context if available
    history_context = ""
    if optimization_history:
        history_context = _build_history_context(optimization_history)

    prompt = f"""You are an expert CUDA performance analyst. Analyze the following kernel and provide optimization guidance.

## Operator Information
- **Type**: {operator_type}
- **Current Performance**: {diagnosis.performance_metrics.get('achieved_bandwidth_gbps', 'N/A')} GB/s bandwidth, {diagnosis.performance_metrics.get('achieved_occupancy', 'N/A')} occupancy

## Performance Diagnosis

{perf_summary}

## Bottleneck Analysis

{bottleneck_analysis}

## Current Kernel Code

```cuda
{current_code}
```

{history_context}

## Your Task

Based on the performance diagnosis above, provide:

1. **Root Cause Analysis**: What is the primary performance bottleneck? Why is it occurring in this specific code?

2. **Optimization Priority**: Rank the top 3 optimization opportunities by expected impact:
   - High impact (>2x speedup potential)
   - Medium impact (1.5-2x speedup potential)
   - Low impact (<1.5x speedup potential)

3. **Specific Recommendations**: For each high-priority issue, provide:
   - What to change (be specific about code patterns)
   - Why it will help (connect to the bottleneck)
   - Expected performance improvement

4. **Implementation Guidance**: What should the code generator focus on?
   - Key optimization techniques to apply
   - Parameters to tune (block size, tile size, etc.)
   - Potential pitfalls to avoid

## Output Format

Provide your analysis in the following structure:

**ROOT CAUSE**: [1-2 sentences identifying the primary bottleneck]

**OPTIMIZATION PRIORITIES**:
1. [High priority item] - Expected impact: [X]x
2. [Medium priority item] - Expected impact: [X]x
3. [Low priority item] - Expected impact: [X]x

**RECOMMENDATIONS**:
- **[Optimization 1]**: [What to change] → [Why it helps] → [Expected improvement]
- **[Optimization 2]**: [What to change] → [Why it helps] → [Expected improvement]
- **[Optimization 3]**: [What to change] → [Why it helps] → [Expected improvement]

**IMPLEMENTATION FOCUS**:
- Techniques: [List key techniques]
- Parameters: [Suggest parameter ranges]
- Pitfalls: [Warn about common mistakes]
"""

    return prompt


def _build_performance_summary(diagnosis: PerformanceDiagnosis) -> str:
    """Build high-level performance summary."""

    metrics = diagnosis.performance_metrics

    summary_lines = [
        "### Overall Performance",
        f"- **Bottleneck Type**: {diagnosis.bottleneck_type}",
        f"- **Bandwidth Utilization**: {metrics.get('bandwidth_utilization_pct', 0):.1f}%",
        f"- **Compute Utilization**: {metrics.get('compute_utilization_pct', 0):.1f}%",
        f"- **Occupancy**: {metrics.get('achieved_occupancy', 0):.2f} ({metrics.get('occupancy_pct', 0):.1f}% of theoretical)",
    ]

    # Add Roofline position
    if 'roofline_position' in metrics:
        summary_lines.append(f"- **Roofline Position**: {metrics['roofline_position']}")
        summary_lines.append(f"- **Arithmetic Intensity**: {metrics.get('arithmetic_intensity', 0):.2f} FLOPs/byte")

    # Add efficiency metrics
    if 'efficiency_pct' in metrics:
        summary_lines.append(f"- **Overall Efficiency**: {metrics['efficiency_pct']:.1f}%")

    return "\n".join(summary_lines)


def _build_bottleneck_analysis(diagnosis: PerformanceDiagnosis) -> str:
    """Build detailed bottleneck analysis."""

    bottleneck = diagnosis.bottleneck_type
    analysis_lines = [f"### Primary Bottleneck: {bottleneck}", ""]

    if bottleneck == "memory_bound":
        mem_diag = diagnosis.memory_diagnosis
        analysis_lines.extend([
            "**Memory System Issues**:",
            f"- Bandwidth utilization: {mem_diag.get('bandwidth_utilization_pct', 0):.1f}%",
            f"- Memory access pattern: {mem_diag.get('access_pattern', 'unknown')}",
            f"- L1 cache efficiency: {mem_diag.get('l1_efficiency_pct', 0):.1f}%",
            f"- L2 cache efficiency: {mem_diag.get('l2_efficiency_pct', 0):.1f}%",
            "",
            "**Key Issues**:"
        ])

        for issue in mem_diag.get('issues', []):
            analysis_lines.append(f"- {issue}")

    elif bottleneck == "compute_bound":
        comp_diag = diagnosis.compute_diagnosis
        analysis_lines.extend([
            "**Compute System Issues**:",
            f"- SM throughput: {comp_diag.get('sm_throughput_pct', 0):.1f}%",
            f"- Occupancy: {comp_diag.get('occupancy', 0):.2f}",
            f"- Warp efficiency: {comp_diag.get('warp_efficiency_pct', 0):.1f}%",
            "",
            "**Key Issues**:"
        ])

        for issue in comp_diag.get('issues', []):
            analysis_lines.append(f"- {issue}")

    elif bottleneck == "latency_bound":
        analysis_lines.extend([
            "**Latency Issues**:",
            "- High instruction latency or dependency chains",
            "- Insufficient parallelism to hide latency",
            "",
            "**Key Issues**:"
        ])

        for issue in diagnosis.recommendations[:3]:  # Top 3 issues
            analysis_lines.append(f"- {issue['issue']}")

    return "\n".join(analysis_lines)


def _build_recommendations(diagnosis: PerformanceDiagnosis, operator_type: str) -> str:
    """Build prioritized optimization recommendations."""

    rec_lines = ["### Prioritized Recommendations", ""]

    # Group recommendations by priority
    high_priority = [r for r in diagnosis.recommendations if r['priority'] == 'high']
    medium_priority = [r for r in diagnosis.recommendations if r['priority'] == 'medium']
    low_priority = [r for r in diagnosis.recommendations if r['priority'] == 'low']

    if high_priority:
        rec_lines.append("**High Priority** (Expected >2x impact):")
        for i, rec in enumerate(high_priority[:3], 1):
            rec_lines.append(f"{i}. **{rec['issue']}**")
            rec_lines.append(f"   - Recommendation: {rec['recommendation']}")
            rec_lines.append(f"   - Expected impact: {rec['expected_impact']}")
            rec_lines.append("")

    if medium_priority:
        rec_lines.append("**Medium Priority** (Expected 1.5-2x impact):")
        for i, rec in enumerate(medium_priority[:2], 1):
            rec_lines.append(f"{i}. **{rec['issue']}**")
            rec_lines.append(f"   - Recommendation: {rec['recommendation']}")
            rec_lines.append("")

    if low_priority:
        rec_lines.append("**Low Priority** (Expected <1.5x impact):")
        for i, rec in enumerate(low_priority[:2], 1):
            rec_lines.append(f"{i}. {rec['issue']}: {rec['recommendation']}")

    return "\n".join(rec_lines)


def _build_history_context(optimization_history: List[Dict]) -> str:
    """Build context from previous optimization attempts."""

    if not optimization_history:
        return ""

    history_lines = [
        "## Optimization History",
        "",
        "Previous attempts (most recent first):",
        ""
    ]

    for i, attempt in enumerate(optimization_history[-3:], 1):  # Last 3 attempts
        strategy = attempt.get('strategy', 'unknown')
        speedup = attempt.get('speedup', 0)
        bottleneck_before = attempt.get('bottleneck_before', 'unknown')
        bottleneck_after = attempt.get('bottleneck_after', 'unknown')

        history_lines.append(f"**Attempt {i}**: {strategy}")
        history_lines.append(f"- Speedup: {speedup:.2f}x")
        history_lines.append(f"- Bottleneck shift: {bottleneck_before} → {bottleneck_after}")

        if 'notes' in attempt:
            history_lines.append(f"- Notes: {attempt['notes']}")

        history_lines.append("")

    # Add trend analysis
    if len(optimization_history) >= 2:
        recent_speedups = [h.get('speedup', 1.0) for h in optimization_history[-3:]]
        avg_speedup = sum(recent_speedups) / len(recent_speedups)

        if avg_speedup < 1.1:
            history_lines.append("⚠️ **Warning**: Recent optimizations show diminishing returns. Consider a different approach.")
        elif all(s < 1.0 for s in recent_speedups[-2:]):
            history_lines.append("⚠️ **Warning**: Last two attempts degraded performance. Revert to a stable baseline.")

    return "\n".join(history_lines)


def build_simple_judge_prompt(
    operator_type: str,
    current_code: str,
    ncu_metrics: Dict[str, float]
) -> str:
    """
    Build a simple judge prompt with raw NCU metrics (fallback mode).

    Args:
        operator_type: Type of operator
        current_code: Current kernel code
        ncu_metrics: Raw NCU metrics dictionary

    Returns:
        Formatted prompt string
    """

    prompt = f"""You are an expert CUDA performance analyst. Analyze the following kernel and provide optimization guidance.

## Operator Information
- **Type**: {operator_type}

## Performance Metrics (from NVIDIA Nsight Compute)

```
{_format_ncu_metrics(ncu_metrics)}
```

## Current Kernel Code

```cuda
{current_code}
```

## Your Task

Analyze the performance metrics and code, then provide:

1. **Bottleneck Identification**: What is limiting performance?
2. **Optimization Recommendations**: What should be changed and why?
3. **Expected Impact**: Estimate the potential speedup for each recommendation.

Provide specific, actionable guidance for the code generator.
"""

    return prompt


def _format_ncu_metrics(metrics: Dict[str, float]) -> str:
    """Format NCU metrics for display."""

    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"{key}: {value:.2f}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)
