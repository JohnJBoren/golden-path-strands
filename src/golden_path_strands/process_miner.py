"""Process mining for discovering patterns in agent traces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import statistics
import structlog

logger = structlog.get_logger()


@dataclass
class ProcessPattern:
    """Represents a discovered process pattern."""
    pattern_id: str
    sequence: List[str]  # Sequence of actions/tools
    frequency: int
    avg_score: float
    avg_time_ms: float
    avg_tokens: int
    success_rate: float
    examples: List[str]  # Trace IDs


@dataclass
class Bottleneck:
    """Represents a performance bottleneck."""
    location: str
    impact_severity: float  # 0-1 scale
    occurrences: int
    avg_delay_ms: float
    suggested_optimization: str


@dataclass
class PreferencePair:
    """Preference pair for DPO training."""
    prompt: str
    chosen_response: str
    chosen_trace_id: str
    chosen_score: float
    rejected_response: str
    rejected_trace_id: str
    rejected_score: float
    metadata: Dict[str, Any]


class ProcessMiner:
    """Discovers patterns and generates training data from agent traces.

    This class implements process mining techniques to:
    1. Discover common successful patterns (golden paths)
    2. Identify bottlenecks and inefficiencies
    3. Generate preference pairs for DPO training
    4. Analyze decision sequences for optimization
    """

    def __init__(self, min_pattern_frequency: int = 3):
        """Initialize the process miner.

        Args:
            min_pattern_frequency: Minimum frequency for a pattern to be significant
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.traces: List[Dict[str, Any]] = []
        self.patterns: List[ProcessPattern] = []
        self.bottlenecks: List[Bottleneck] = []

    def load_traces_from_jsonl(self, file_path: str | Path) -> None:
        """Load traces from a JSONL file.

        Args:
            file_path: Path to JSONL file containing traces
        """
        file_path = Path(file_path)

        with open(file_path, 'r') as f:
            line_num = 0
            for line in f:
                line_num += 1
                try:
                    data = json.loads(line)
                    # Skip header/metadata lines
                    if not data.get("_file_type"):
                        self.traces.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipped invalid JSON line",
                        file=str(file_path),
                        line_number=line_num,
                        error=str(e)
                    )
                    continue

        print(f"✓ Loaded {len(self.traces)} traces from {file_path.name}")

    def load_traces_from_directory(self, directory: str | Path) -> None:
        """Load all traces from JSONL files in a directory.

        Args:
            directory: Directory containing JSONL files
        """
        directory = Path(directory)

        for file_path in directory.glob("*.jsonl"):
            self.load_traces_from_jsonl(file_path)

        # Also check compressed files
        for file_path in directory.glob("*.jsonl.gz"):
            # Would need to decompress first
            print(f"⚠ Skipping compressed file: {file_path.name}")

    def discover_patterns(self) -> List[ProcessPattern]:
        """Discover common patterns in successful traces.

        Returns:
            List of discovered process patterns
        """
        # Extract sequences from successful traces
        successful_sequences = []

        for trace in self.traces:
            if trace.get("outcome") == "successful":
                sequence = self._extract_action_sequence(trace)
                if sequence:
                    successful_sequences.append({
                        "sequence": sequence,
                        "trace_id": trace.get("trace_id"),
                        "score": trace.get("evaluation_score", 0),
                        "time_ms": trace.get("total_time_ms", 0),
                        "tokens": self._calculate_total_tokens(trace)
                    })

        # Find common patterns
        pattern_groups = defaultdict(list)

        for seq_data in successful_sequences:
            # Create pattern key from sequence
            pattern_key = "->".join(seq_data["sequence"])
            pattern_groups[pattern_key].append(seq_data)

        # Create ProcessPattern objects for significant patterns
        self.patterns = []

        for pattern_key, instances in pattern_groups.items():
            if len(instances) >= self.min_pattern_frequency:
                pattern = ProcessPattern(
                    pattern_id=f"pattern_{len(self.patterns):03d}",
                    sequence=pattern_key.split("->"),
                    frequency=len(instances),
                    avg_score=statistics.mean(i["score"] for i in instances),
                    avg_time_ms=statistics.mean(i["time_ms"] for i in instances),
                    avg_tokens=int(statistics.mean(i["tokens"] for i in instances)),
                    success_rate=1.0,  # All successful by definition
                    examples=[i["trace_id"] for i in instances[:5]]  # Keep up to 5 examples
                )
                self.patterns.append(pattern)

        # Sort patterns by frequency and score
        self.patterns.sort(key=lambda p: (p.frequency, p.avg_score), reverse=True)

        print(f"✓ Discovered {len(self.patterns)} significant patterns")
        return self.patterns

    def identify_bottlenecks(self) -> List[Bottleneck]:
        """Identify performance bottlenecks in agent executions.

        Returns:
            List of identified bottlenecks
        """
        self.bottlenecks = []

        # Analyze decision points for delays
        decision_delays = defaultdict(list)
        tool_delays = defaultdict(list)

        for trace in self.traces:
            decisions = trace.get("decision_sequence", [])

            for i, decision in enumerate(decisions):
                location = f"decision_{i}"
                time_ms = decision.get("execution_time_ms", 0)

                if time_ms > 0:
                    decision_delays[location].append(time_ms)

                # Analyze tool delays
                for tool_call in decision.get("tool_calls", []):
                    tool_name = tool_call.get("tool", "unknown")
                    tool_delays[tool_name].append(time_ms)

        # Identify bottlenecks from delays
        for location, delays in decision_delays.items():
            if len(delays) >= 5:  # Need enough samples
                avg_delay = statistics.mean(delays)
                std_delay = statistics.stdev(delays) if len(delays) > 1 else 0

                # Bottleneck if average delay is high
                if avg_delay > 3000:  # 3 seconds
                    bottleneck = Bottleneck(
                        location=location,
                        impact_severity=min(1.0, avg_delay / 10000),  # Normalize to 0-1
                        occurrences=len(delays),
                        avg_delay_ms=avg_delay,
                        suggested_optimization=self._suggest_optimization(location, avg_delay)
                    )
                    self.bottlenecks.append(bottleneck)

        # Check tool-specific bottlenecks
        for tool_name, delays in tool_delays.items():
            if len(delays) >= 5:
                avg_delay = statistics.mean(delays)

                if avg_delay > 2000:  # 2 seconds for tools
                    bottleneck = Bottleneck(
                        location=f"tool:{tool_name}",
                        impact_severity=min(1.0, avg_delay / 5000),
                        occurrences=len(delays),
                        avg_delay_ms=avg_delay,
                        suggested_optimization=f"Consider caching {tool_name} results or using faster alternative"
                    )
                    self.bottlenecks.append(bottleneck)

        # Sort by severity
        self.bottlenecks.sort(key=lambda b: b.impact_severity, reverse=True)

        print(f"✓ Identified {len(self.bottlenecks)} bottlenecks")
        return self.bottlenecks

    def generate_preference_pairs(
        self,
        min_score_difference: float = 0.15
    ) -> List[PreferencePair]:
        """Generate preference pairs for DPO training.

        Args:
            min_score_difference: Minimum score difference for valid pair

        Returns:
            List of preference pairs
        """
        pairs = []

        # Group traces by similar prompts
        prompt_groups = defaultdict(list)

        for trace in self.traces:
            prompt = trace.get("user_message", "")
            if prompt:
                # Normalize prompt for grouping
                prompt_key = prompt[:100].lower().strip()
                prompt_groups[prompt_key].append(trace)

        # Generate pairs from each group
        for prompt_key, group_traces in prompt_groups.items():
            if len(group_traces) < 2:
                continue

            # Sort by evaluation score
            scored_traces = [
                t for t in group_traces
                if t.get("evaluation_score") is not None
            ]
            scored_traces.sort(key=lambda t: t.get("evaluation_score", 0), reverse=True)

            # Create pairs between high and low scoring traces
            for i in range(len(scored_traces) // 2):
                high_trace = scored_traces[i]
                low_trace = scored_traces[-(i + 1)]

                score_diff = high_trace.get("evaluation_score", 0) - low_trace.get("evaluation_score", 0)

                if score_diff >= min_score_difference:
                    pair = PreferencePair(
                        prompt=high_trace.get("user_message", ""),
                        chosen_response=high_trace.get("final_response", ""),
                        chosen_trace_id=high_trace.get("trace_id", ""),
                        chosen_score=high_trace.get("evaluation_score", 0),
                        rejected_response=low_trace.get("final_response", ""),
                        rejected_trace_id=low_trace.get("trace_id", ""),
                        rejected_score=low_trace.get("evaluation_score", 0),
                        metadata={
                            "score_difference": score_diff,
                            "chosen_tokens": self._calculate_total_tokens(high_trace),
                            "rejected_tokens": self._calculate_total_tokens(low_trace),
                            "chosen_time_ms": high_trace.get("total_time_ms", 0),
                            "rejected_time_ms": low_trace.get("total_time_ms", 0)
                        }
                    )
                    pairs.append(pair)

        print(f"✓ Generated {len(pairs)} preference pairs for DPO training")
        return pairs

    def analyze_tool_usage(self) -> Dict[str, Any]:
        """Analyze tool usage patterns across traces.

        Returns:
            Tool usage analysis
        """
        tool_stats = defaultdict(lambda: {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "avg_time_ms": [],
            "co_occurrences": Counter()
        })

        for trace in self.traces:
            trace_tools = set()

            for decision in trace.get("decision_sequence", []):
                for tool_call in decision.get("tool_calls", []):
                    tool_name = tool_call.get("tool", "unknown")
                    tool_stats[tool_name]["calls"] += 1

                    # Track success/failure
                    if trace.get("outcome") == "successful":
                        tool_stats[tool_name]["successes"] += 1
                    else:
                        tool_stats[tool_name]["failures"] += 1

                    # Track timing
                    if "execution_time_ms" in decision:
                        tool_stats[tool_name]["avg_time_ms"].append(
                            decision["execution_time_ms"]
                        )

                    trace_tools.add(tool_name)

            # Track co-occurrences
            for tool1 in trace_tools:
                for tool2 in trace_tools:
                    if tool1 != tool2:
                        tool_stats[tool1]["co_occurrences"][tool2] += 1

        # Calculate averages
        analysis = {}
        for tool_name, stats in tool_stats.items():
            avg_time = (
                statistics.mean(stats["avg_time_ms"])
                if stats["avg_time_ms"] else 0
            )

            analysis[tool_name] = {
                "total_calls": stats["calls"],
                "success_rate": (
                    stats["successes"] / stats["calls"]
                    if stats["calls"] > 0 else 0
                ),
                "avg_execution_time_ms": avg_time,
                "top_co_occurrences": dict(stats["co_occurrences"].most_common(3))
            }

        return analysis

    def export_golden_paths(self, output_file: str | Path) -> None:
        """Export discovered golden paths for training.

        Args:
            output_file: Output file path
        """
        output_file = Path(output_file)

        golden_data = {
            "patterns": [asdict(p) for p in self.patterns],
            "bottlenecks": [asdict(b) for b in self.bottlenecks],
            "tool_usage": self.analyze_tool_usage(),
            "statistics": self.get_statistics()
        }

        with open(output_file, 'w') as f:
            json.dump(golden_data, f, indent=2)

        print(f"✓ Exported golden paths to {output_file}")

    def export_dpo_dataset(
        self,
        output_file: str | Path,
        pairs: Optional[List[PreferencePair]] = None
    ) -> None:
        """Export preference pairs in DPO format.

        Args:
            output_file: Output file path
            pairs: Preference pairs to export (generates if not provided)
        """
        output_file = Path(output_file)

        if pairs is None:
            pairs = self.generate_preference_pairs()

        # Convert to DPO format
        dpo_data = []
        for pair in pairs:
            dpo_entry = {
                "prompt": pair.prompt,
                "chosen": pair.chosen_response,
                "rejected": pair.rejected_response,
                "metadata": {
                    **pair.metadata,
                    "chosen_trace_id": pair.chosen_trace_id,
                    "rejected_trace_id": pair.rejected_trace_id,
                    "chosen_score": pair.chosen_score,
                    "rejected_score": pair.rejected_score
                }
            }
            dpo_data.append(dpo_entry)

        # Write as JSONL
        with open(output_file, 'w') as f:
            for entry in dpo_data:
                f.write(json.dumps(entry) + '\n')

        print(f"✓ Exported {len(dpo_data)} DPO training examples to {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about processed traces.

        Returns:
            Statistics dictionary
        """
        if not self.traces:
            return {"total_traces": 0}

        successful = [t for t in self.traces if t.get("outcome") == "successful"]
        failed = [t for t in self.traces if t.get("outcome") == "failed"]

        all_times = [t.get("total_time_ms", 0) for t in self.traces if t.get("total_time_ms")]
        all_scores = [t.get("evaluation_score", 0) for t in self.traces if t.get("evaluation_score")]

        return {
            "total_traces": len(self.traces),
            "successful_traces": len(successful),
            "failed_traces": len(failed),
            "success_rate": len(successful) / len(self.traces) if self.traces else 0,
            "discovered_patterns": len(self.patterns),
            "identified_bottlenecks": len(self.bottlenecks),
            "avg_execution_time_ms": statistics.mean(all_times) if all_times else 0,
            "avg_evaluation_score": statistics.mean(all_scores) if all_scores else 0,
            "unique_prompts": len(set(t.get("user_message", "") for t in self.traces))
        }

    def _extract_action_sequence(self, trace: Dict[str, Any]) -> List[str]:
        """Extract action sequence from a trace.

        Args:
            trace: Agent trace

        Returns:
            List of actions/tools in sequence
        """
        sequence = []

        for decision in trace.get("decision_sequence", []):
            # Add tool calls to sequence
            for tool_call in decision.get("tool_calls", []):
                tool_name = tool_call.get("tool", "unknown")
                sequence.append(tool_name)

            # Could also add decision types, model calls, etc.

        return sequence

    def _calculate_total_tokens(self, trace: Dict[str, Any]) -> int:
        """Calculate total tokens used in a trace.

        Args:
            trace: Agent trace

        Returns:
            Total token count
        """
        tokens = trace.get("total_tokens", {})
        return tokens.get("prompt", 0) + tokens.get("completion", 0)

    def _suggest_optimization(self, location: str, avg_delay: float) -> str:
        """Suggest optimization for a bottleneck.

        Args:
            location: Bottleneck location
            avg_delay: Average delay in ms

        Returns:
            Optimization suggestion
        """
        if "decision" in location:
            if avg_delay > 5000:
                return "Consider breaking complex decisions into smaller steps"
            else:
                return "Consider caching similar decision results"
        elif "tool" in location:
            return "Optimize tool implementation or add result caching"
        else:
            return "Investigate and optimize this operation"