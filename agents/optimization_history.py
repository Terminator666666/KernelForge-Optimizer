"""
Optimization History Manager - Tracks optimization iterations and recommends strategies.

This module records optimization history, analyzes trends, detects bottleneck shifts,
and recommends next optimization strategies based on past performance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


@dataclass
class OptimizationRound:
    """Record of a single optimization iteration."""

    round_id: int
    timestamp: str

    # Input state
    kernel_code: str
    operator_type: str
    gpu_name: str

    # Performance diagnosis
    bottleneck: str
    bottleneck_confidence: float
    bandwidth_util: float
    occupancy: float
    arithmetic_intensity: float

    # Strategy applied
    strategy_name: str
    strategy_params: Dict[str, any]

    # Results
    execution_time_ms: float
    speedup: float  # Relative to previous round
    compilation_success: bool
    runtime_success: bool
    error_message: Optional[str] = None

    # Metadata
    notes: str = ""


@dataclass
class StrategyEffectiveness:
    """Statistics about a strategy's effectiveness."""

    strategy_name: str
    times_applied: int
    success_rate: float  # Percentage of successful applications
    avg_speedup: float
    best_speedup: float
    worst_speedup: float
    applicable_bottlenecks: List[str]


class OptimizationHistory:
    """Manages optimization history and provides intelligent recommendations."""

    def __init__(self):
        self.rounds: List[OptimizationRound] = []
        self.best_round: Optional[OptimizationRound] = None
        self.best_time_ms: float = float('inf')

    def add_round(self, round_data: OptimizationRound):
        """Add a new optimization round to history."""
        self.rounds.append(round_data)

        # Update best round
        if (round_data.runtime_success and
            round_data.execution_time_ms < self.best_time_ms):
            self.best_time_ms = round_data.execution_time_ms
            self.best_round = round_data

    def get_recent_trend(self, n: int = 3) -> str:
        """
        Analyze recent optimization trend.

        Args:
            n: Number of recent rounds to analyze

        Returns:
            Trend description: 'improving', 'stagnant', 'degrading', 'unstable'
        """
        if len(self.rounds) < 2:
            return 'insufficient_data'

        recent = self.rounds[-n:]
        successful = [r for r in recent if r.runtime_success]

        if len(successful) < 2:
            return 'unstable'

        # Calculate speedup trend
        speedups = [r.speedup for r in successful]
        avg_speedup = sum(speedups) / len(speedups)

        # Check variance
        variance = sum((s - avg_speedup) ** 2 for s in speedups) / len(speedups)
        std_dev = variance ** 0.5

        if std_dev > 0.3:  # High variance
            return 'unstable'
        elif avg_speedup > 1.1:  # Average speedup > 10%
            return 'improving'
        elif avg_speedup > 0.95:  # Roughly flat
            return 'stagnant'
        else:
            return 'degrading'

    def detect_bottleneck_shift(self, window: int = 3) -> Optional[Tuple[str, str]]:
        """
        Detect if the primary bottleneck has shifted.

        Args:
            window: Number of recent rounds to check

        Returns:
            Tuple of (old_bottleneck, new_bottleneck) if shift detected, None otherwise
        """
        if len(self.rounds) < window + 1:
            return None

        recent = self.rounds[-window:]
        previous = self.rounds[-(window + 1)]

        # Get most common recent bottleneck
        recent_bottlenecks = [r.bottleneck for r in recent]
        most_common = max(set(recent_bottlenecks), key=recent_bottlenecks.count)

        if most_common != previous.bottleneck:
            return (previous.bottleneck, most_common)

        return None

    def analyze_strategy_effectiveness(self) -> List[StrategyEffectiveness]:
        """
        Analyze effectiveness of each strategy that has been tried.

        Returns:
            List of StrategyEffectiveness objects, sorted by avg_speedup
        """
        strategy_stats = {}

        for round_data in self.rounds:
            strategy = round_data.strategy_name

            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'times_applied': 0,
                    'successes': 0,
                    'speedups': [],
                    'bottlenecks': set()
                }

            stats = strategy_stats[strategy]
            stats['times_applied'] += 1
            stats['bottlenecks'].add(round_data.bottleneck)

            if round_data.runtime_success:
                stats['successes'] += 1
                stats['speedups'].append(round_data.speedup)

        # Convert to StrategyEffectiveness objects
        effectiveness_list = []
        for strategy, stats in strategy_stats.items():
            if stats['speedups']:
                avg_speedup = sum(stats['speedups']) / len(stats['speedups'])
                best_speedup = max(stats['speedups'])
                worst_speedup = min(stats['speedups'])
            else:
                avg_speedup = 0.0
                best_speedup = 0.0
                worst_speedup = 0.0

            success_rate = (stats['successes'] / stats['times_applied']) * 100

            effectiveness_list.append(StrategyEffectiveness(
                strategy_name=strategy,
                times_applied=stats['times_applied'],
                success_rate=success_rate,
                avg_speedup=avg_speedup,
                best_speedup=best_speedup,
                worst_speedup=worst_speedup,
                applicable_bottlenecks=list(stats['bottlenecks'])
            ))

        # Sort by average speedup
        effectiveness_list.sort(key=lambda x: x.avg_speedup, reverse=True)

        return effectiveness_list

    def recommend_next_strategy(self,
                               current_bottleneck: str,
                               available_strategies: List[str],
                               avoid_recent: int = 2) -> Tuple[str, str]:
        """
        Recommend next optimization strategy based on history.

        Args:
            current_bottleneck: Current performance bottleneck
            available_strategies: List of applicable strategy names
            avoid_recent: Avoid strategies used in last N rounds

        Returns:
            Tuple of (strategy_name, reasoning)
        """
        if not self.rounds:
            # No history - return first available strategy
            return (available_strategies[0], "No history available, trying first applicable strategy")

        # Get recently tried strategies
        recent_strategies = set()
        if len(self.rounds) >= avoid_recent:
            recent_strategies = {r.strategy_name for r in self.rounds[-avoid_recent:]}

        # Analyze strategy effectiveness
        effectiveness = self.analyze_strategy_effectiveness()
        effectiveness_map = {e.strategy_name: e for e in effectiveness}

        # Check for bottleneck shift
        shift = self.detect_bottleneck_shift()
        if shift:
            old_bottleneck, new_bottleneck = shift
            reasoning = f"Bottleneck shifted from {old_bottleneck} to {new_bottleneck}. "
        else:
            reasoning = f"Continuing to address {current_bottleneck} bottleneck. "

        # Get trend
        trend = self.get_recent_trend()

        # Strategy selection logic
        if trend == 'improving':
            # Continue with similar strategies, but avoid exact repeats
            last_strategy = self.rounds[-1].strategy_name
            if last_strategy in effectiveness_map:
                # Look for strategies with similar effectiveness
                similar = [s for s in available_strategies
                          if s != last_strategy and s not in recent_strategies]
                if similar:
                    # Prefer strategies that worked well before
                    similar_with_history = [s for s in similar if s in effectiveness_map]
                    if similar_with_history:
                        best = max(similar_with_history,
                                 key=lambda s: effectiveness_map[s].avg_speedup)
                        reasoning += f"Trend is improving. Trying {best} which has shown {effectiveness_map[best].avg_speedup:.2f}x average speedup."
                        return (best, reasoning)

            # Fallback: try any untried strategy
            untried = [s for s in available_strategies if s not in effectiveness_map]
            if untried:
                reasoning += f"Trying untried strategy {untried[0]}."
                return (untried[0], reasoning)

        elif trend == 'stagnant':
            # Try a different approach - prefer untried strategies
            untried = [s for s in available_strategies
                      if s not in effectiveness_map and s not in recent_strategies]
            if untried:
                reasoning += f"Progress stagnant. Trying new strategy {untried[0]}."
                return (untried[0], reasoning)

            # If all tried, pick best performing that wasn't recent
            tried_not_recent = [s for s in available_strategies
                               if s in effectiveness_map and s not in recent_strategies]
            if tried_not_recent:
                best = max(tried_not_recent,
                         key=lambda s: effectiveness_map[s].avg_speedup)
                reasoning += f"Revisiting {best} which previously achieved {effectiveness_map[best].avg_speedup:.2f}x speedup."
                return (best, reasoning)

        elif trend == 'degrading':
            # Revert to best known strategy
            if self.best_round:
                best_strategy = self.best_round.strategy_name
                if best_strategy in available_strategies:
                    reasoning += f"Performance degrading. Reverting to best known strategy {best_strategy}."
                    return (best_strategy, reasoning)

            # Otherwise try most successful strategy
            if effectiveness:
                best = effectiveness[0].strategy_name
                if best in available_strategies:
                    reasoning += f"Trying most successful strategy {best} (avg {effectiveness[0].avg_speedup:.2f}x speedup)."
                    return (best, reasoning)

        elif trend == 'unstable':
            # Try simpler, more conservative strategies
            # Prefer strategies with high success rate
            stable_strategies = [s for s in available_strategies
                               if s in effectiveness_map and
                               effectiveness_map[s].success_rate > 80]
            if stable_strategies:
                best = max(stable_strategies,
                         key=lambda s: effectiveness_map[s].success_rate)
                reasoning += f"Results unstable. Trying reliable strategy {best} ({effectiveness_map[best].success_rate:.0f}% success rate)."
                return (best, reasoning)

        # Default fallback: pick first available strategy not recently used
        for strategy in available_strategies:
            if strategy not in recent_strategies:
                reasoning += f"Using default selection: {strategy}."
                return (strategy, reasoning)

        # Last resort: use any available strategy
        reasoning += f"All strategies recently tried. Retrying {available_strategies[0]}."
        return (available_strategies[0], reasoning)

    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of optimization history.

        Returns:
            Dictionary with summary statistics
        """
        if not self.rounds:
            return {
                'total_rounds': 0,
                'successful_rounds': 0,
                'best_time_ms': None,
                'total_speedup': 1.0,
                'strategies_tried': []
            }

        successful = [r for r in self.rounds if r.runtime_success]

        # Calculate total speedup (first round to best round)
        if successful and self.best_round:
            first_time = successful[0].execution_time_ms
            total_speedup = first_time / self.best_round.execution_time_ms
        else:
            total_speedup = 1.0

        strategies_tried = list(set(r.strategy_name for r in self.rounds))

        return {
            'total_rounds': len(self.rounds),
            'successful_rounds': len(successful),
            'success_rate': (len(successful) / len(self.rounds)) * 100,
            'best_time_ms': self.best_time_ms if self.best_round else None,
            'total_speedup': total_speedup,
            'strategies_tried': strategies_tried,
            'current_trend': self.get_recent_trend(),
            'bottleneck_shifts': self._count_bottleneck_shifts()
        }

    def _count_bottleneck_shifts(self) -> int:
        """Count number of times bottleneck has shifted."""
        if len(self.rounds) < 2:
            return 0

        shifts = 0
        prev_bottleneck = self.rounds[0].bottleneck

        for round_data in self.rounds[1:]:
            if round_data.bottleneck != prev_bottleneck:
                shifts += 1
                prev_bottleneck = round_data.bottleneck

        return shifts

    def save_to_file(self, filepath: str):
        """Save optimization history to JSON file."""
        data = {
            'rounds': [
                {
                    'round_id': r.round_id,
                    'timestamp': r.timestamp,
                    'operator_type': r.operator_type,
                    'gpu_name': r.gpu_name,
                    'bottleneck': r.bottleneck,
                    'bottleneck_confidence': r.bottleneck_confidence,
                    'bandwidth_util': r.bandwidth_util,
                    'occupancy': r.occupancy,
                    'arithmetic_intensity': r.arithmetic_intensity,
                    'strategy_name': r.strategy_name,
                    'strategy_params': r.strategy_params,
                    'execution_time_ms': r.execution_time_ms,
                    'speedup': r.speedup,
                    'compilation_success': r.compilation_success,
                    'runtime_success': r.runtime_success,
                    'error_message': r.error_message,
                    'notes': r.notes
                }
                for r in self.rounds
            ],
            'summary': self.get_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'OptimizationHistory':
        """Load optimization history from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        history = cls()

        for round_data in data['rounds']:
            round_obj = OptimizationRound(
                round_id=round_data['round_id'],
                timestamp=round_data['timestamp'],
                kernel_code="",  # Not saved to reduce file size
                operator_type=round_data['operator_type'],
                gpu_name=round_data['gpu_name'],
                bottleneck=round_data['bottleneck'],
                bottleneck_confidence=round_data['bottleneck_confidence'],
                bandwidth_util=round_data['bandwidth_util'],
                occupancy=round_data['occupancy'],
                arithmetic_intensity=round_data['arithmetic_intensity'],
                strategy_name=round_data['strategy_name'],
                strategy_params=round_data['strategy_params'],
                execution_time_ms=round_data['execution_time_ms'],
                speedup=round_data['speedup'],
                compilation_success=round_data['compilation_success'],
                runtime_success=round_data['runtime_success'],
                error_message=round_data.get('error_message'),
                notes=round_data.get('notes', '')
            )
            history.add_round(round_obj)

        return history

    def generate_report(self) -> str:
        """
        Generate human-readable optimization report.

        Returns:
            Formatted report string
        """
        if not self.rounds:
            return "No optimization rounds recorded."

        summary = self.get_summary()
        effectiveness = self.analyze_strategy_effectiveness()

        report = []
        report.append("=" * 80)
        report.append("OPTIMIZATION HISTORY REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary section
        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Total rounds: {summary['total_rounds']}")
        report.append(f"Successful rounds: {summary['successful_rounds']} ({summary['success_rate']:.1f}%)")
        report.append(f"Best execution time: {summary['best_time_ms']:.3f} ms")
        report.append(f"Total speedup: {summary['total_speedup']:.2f}x")
        report.append(f"Current trend: {summary['current_trend']}")
        report.append(f"Bottleneck shifts: {summary['bottleneck_shifts']}")
        report.append("")

        # Strategy effectiveness section
        report.append("STRATEGY EFFECTIVENESS")
        report.append("-" * 80)
        for eff in effectiveness:
            report.append(f"\n{eff.strategy_name}:")
            report.append(f"  Applied: {eff.times_applied} times")
            report.append(f"  Success rate: {eff.success_rate:.1f}%")
            report.append(f"  Average speedup: {eff.avg_speedup:.2f}x")
            report.append(f"  Best speedup: {eff.best_speedup:.2f}x")
            report.append(f"  Applicable bottlenecks: {', '.join(eff.applicable_bottlenecks)}")
        report.append("")

        # Round-by-round details
        report.append("OPTIMIZATION ROUNDS")
        report.append("-" * 80)
        for r in self.rounds:
            status = "✓" if r.runtime_success else "✗"
            report.append(f"\nRound {r.round_id} [{status}]:")
            report.append(f"  Strategy: {r.strategy_name}")
            report.append(f"  Bottleneck: {r.bottleneck} ({r.bottleneck_confidence:.2f} confidence)")
            report.append(f"  Time: {r.execution_time_ms:.3f} ms (speedup: {r.speedup:.2f}x)")
            report.append(f"  Bandwidth util: {r.bandwidth_util:.1f}%")
            report.append(f"  Occupancy: {r.occupancy:.1f}%")
            report.append(f"  Arithmetic intensity: {r.arithmetic_intensity:.2f}")
            if r.error_message:
                report.append(f"  Error: {r.error_message}")
            if r.notes:
                report.append(f"  Notes: {r.notes}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def create_optimization_history() -> OptimizationHistory:
    """Factory function to create optimization history manager."""
    return OptimizationHistory()
