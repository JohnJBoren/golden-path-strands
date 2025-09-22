"""Strands Agents SDK telemetry integration for Golden Path data mining."""
from __future__ import annotations

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.trace import StatusCode
from opentelemetry.semconv.trace import SpanAttributes

import structlog

logger = structlog.get_logger()


@dataclass
class DecisionPoint:
    """Represents a decision point in an agent's execution."""
    timestamp: str
    cycle_number: int
    prompt: str
    model_response: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    tokens_used: Dict[str, int]
    execution_time_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class AgentTrace:
    """Complete trace of an agent's execution."""
    trace_id: str
    session_id: str
    agent_name: str
    user_message: str
    final_response: str
    decision_sequence: List[DecisionPoint]
    total_tokens: Dict[str, int]
    total_time_ms: float
    outcome: str  # "successful", "failed", "partial"
    evaluation_score: Optional[float] = None
    golden_path: bool = False
    metadata: Optional[Dict[str, Any]] = None


class StrandsTraceProcessor(SpanProcessor):
    """OpenTelemetry span processor for capturing Strands agent traces.

    This processor intercepts spans from Strands Agents SDK and converts
    them into structured data for Golden Path analysis.
    """

    def __init__(self, output_dir: str = "datasets/strands_traces"):
        """Initialize the trace processor.

        Args:
            output_dir: Directory to store captured traces in JSONL format
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_traces: Dict[str, AgentTrace] = {}
        self.decision_points: Dict[str, List[DecisionPoint]] = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Output file for this session
        self.output_file = self.output_dir / f"traces_{self.session_id}.jsonl"

        # Thread lock for file writing to prevent race conditions
        self._file_lock = threading.Lock()

    def on_start(self, span: ReadableSpan, parent_context=None) -> None:
        """Called when a span starts.

        Captures the beginning of agent operations, tool calls, and LLM invocations.
        """
        span_name = span.name
        trace_id = format(span.get_span_context().trace_id, '032x')

        # Identify span type and extract relevant data
        if "agent" in span_name.lower():
            # Agent-level span - initialize trace
            self._initialize_agent_trace(span, trace_id)
        elif "cycle" in span_name.lower():
            # Cycle span - prepare for decision point
            self._prepare_decision_point(span, trace_id)
        elif "tool" in span_name.lower():
            # Tool invocation
            self._capture_tool_start(span, trace_id)

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends.

        Captures completed operations and assembles trace data.
        """
        span_name = span.name
        trace_id = format(span.get_span_context().trace_id, '032x')

        # Process based on span type
        if "agent" in span_name.lower():
            # Agent completed - finalize and write trace
            self._finalize_agent_trace(span, trace_id)
        elif "cycle" in span_name.lower():
            # Cycle completed - capture decision point
            self._capture_decision_point(span, trace_id)
        elif "llm" in span_name.lower() or "model" in span_name.lower():
            # LLM call completed - extract prompt/response
            self._capture_llm_interaction(span, trace_id)
        elif "tool" in span_name.lower():
            # Tool completed - capture results
            self._capture_tool_result(span, trace_id)

    def _initialize_agent_trace(self, span: ReadableSpan, trace_id: str) -> None:
        """Initialize a new agent trace."""
        attributes = dict(span.attributes or {})

        agent_trace = AgentTrace(
            trace_id=trace_id,
            session_id=self.session_id,
            agent_name=attributes.get("agent.name", "unknown"),
            user_message=attributes.get("user.message", ""),
            final_response="",  # Will be filled on completion
            decision_sequence=[],
            total_tokens={"prompt": 0, "completion": 0},
            total_time_ms=0,
            outcome="in_progress",
            metadata={"start_time": span.start_time}
        )

        self.current_traces[trace_id] = agent_trace
        self.decision_points[trace_id] = []

    def _prepare_decision_point(self, span: ReadableSpan, trace_id: str) -> None:
        """Prepare to capture a decision point."""
        if trace_id not in self.decision_points:
            self.decision_points[trace_id] = []

    def _capture_decision_point(self, span: ReadableSpan, trace_id: str) -> None:
        """Capture a completed decision point from a cycle span."""
        attributes = dict(span.attributes or {})

        # Calculate execution time
        start_time = span.start_time
        end_time = span.end_time
        execution_time_ms = (end_time - start_time) / 1_000_000  # Convert ns to ms

        decision_point = DecisionPoint(
            timestamp=datetime.fromtimestamp(start_time / 1_000_000_000).isoformat(),
            cycle_number=attributes.get("cycle.number", 0),
            prompt=attributes.get("llm.prompt", ""),
            model_response=attributes.get("llm.response", ""),
            tool_calls=self._extract_tool_calls(attributes),
            tool_results=self._extract_tool_results(attributes),
            tokens_used={
                "prompt": attributes.get("llm.prompt_tokens", 0),
                "completion": attributes.get("llm.completion_tokens", 0)
            },
            execution_time_ms=execution_time_ms,
            success=span.status.status_code == StatusCode.OK,
            error=attributes.get("error.message") if span.status.status_code != StatusCode.OK else None
        )

        if trace_id in self.decision_points:
            self.decision_points[trace_id].append(decision_point)

    def _capture_llm_interaction(self, span: ReadableSpan, trace_id: str) -> None:
        """Capture LLM prompt and response from model invocation spans."""
        attributes = dict(span.attributes or {})

        # Extract prompt and response
        prompt = attributes.get("llm.prompt", attributes.get("gen_ai.prompt", ""))
        response = attributes.get("llm.response", attributes.get("gen_ai.response", ""))

        # Extract token usage
        prompt_tokens = attributes.get("llm.usage.prompt_tokens", 0)
        completion_tokens = attributes.get("llm.usage.completion_tokens", 0)

        # Update current trace if exists
        if trace_id in self.current_traces:
            trace = self.current_traces[trace_id]
            trace.total_tokens["prompt"] += prompt_tokens
            trace.total_tokens["completion"] += completion_tokens

    def _capture_tool_start(self, span: ReadableSpan, trace_id: str) -> None:
        """Capture the start of a tool invocation."""
        # Tool tracking can be enhanced here if needed
        pass

    def _capture_tool_result(self, span: ReadableSpan, trace_id: str) -> None:
        """Capture tool execution results."""
        attributes = dict(span.attributes or {})

        tool_result = {
            "tool_name": attributes.get("tool.name", "unknown"),
            "tool_input": attributes.get("tool.input", {}),
            "tool_output": attributes.get("tool.output", {}),
            "success": span.status.status_code == StatusCode.OK,
            "duration_ms": (span.end_time - span.start_time) / 1_000_000
        }

        # Store tool results for current decision point
        # This would be matched with the current cycle

    def _finalize_agent_trace(self, span: ReadableSpan, trace_id: str) -> None:
        """Finalize and write a completed agent trace."""
        if trace_id not in self.current_traces:
            return

        trace = self.current_traces[trace_id]
        attributes = dict(span.attributes or {})

        # Update final values
        trace.final_response = attributes.get("agent.response", "")
        trace.decision_sequence = self.decision_points.get(trace_id, [])
        trace.total_time_ms = (span.end_time - span.start_time) / 1_000_000

        # Determine outcome
        if span.status.status_code == StatusCode.OK:
            trace.outcome = "successful"
        elif span.status.status_code == StatusCode.ERROR:
            trace.outcome = "failed"
        else:
            trace.outcome = "partial"

        # Write to JSONL file
        self._write_trace_to_jsonl(trace)

        # Clean up
        del self.current_traces[trace_id]
        if trace_id in self.decision_points:
            del self.decision_points[trace_id]

    def _write_trace_to_jsonl(self, trace: AgentTrace) -> None:
        """Write a trace to the JSONL output file with error handling."""
        try:
            # Convert trace to dict, handling nested dataclasses
            trace_dict = self._trace_to_dict(trace)

            # Ensure output directory exists
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file with thread safety to prevent race conditions
            with self._file_lock:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(trace_dict, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure data is written immediately

        except Exception as e:
            logger.error(f"Failed to write trace to JSONL: {e}", trace_id=trace.trace_id)

    def _trace_to_dict(self, trace: AgentTrace) -> Dict[str, Any]:
        """Convert an AgentTrace to a dictionary."""
        result = {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "agent_name": trace.agent_name,
            "user_message": trace.user_message,
            "final_response": trace.final_response,
            "decision_sequence": [asdict(dp) for dp in trace.decision_sequence],
            "total_tokens": trace.total_tokens,
            "total_time_ms": trace.total_time_ms,
            "outcome": trace.outcome,
            "evaluation_score": trace.evaluation_score,
            "golden_path": trace.golden_path,
            "metadata": trace.metadata or {}
        }
        return result

    def _extract_tool_calls(self, attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from span attributes."""
        tool_calls = []

        # Look for tool-related attributes
        for key, value in attributes.items():
            if "tool.call" in key or "tool_call" in key:
                tool_calls.append({
                    "tool": key.split(".")[-1],
                    "arguments": value
                })

        return tool_calls

    def _extract_tool_results(self, attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool results from span attributes."""
        tool_results = []

        # Look for tool result attributes
        for key, value in attributes.items():
            if "tool.result" in key or "tool_result" in key:
                tool_results.append({
                    "tool": key.split(".")[-1],
                    "result": value
                })

        return tool_results

    def shutdown(self) -> None:
        """Shutdown the processor and flush any pending data."""
        # Flush any remaining traces
        for trace_id, trace in self.current_traces.items():
            trace.outcome = "incomplete"
            self._write_trace_to_jsonl(trace)

        self.current_traces.clear()
        self.decision_points.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending traces."""
        # For now, just ensure file is written
        return True


class StrandsTelemetryIntegration:
    """Integration class for setting up Strands telemetry with Golden Path."""

    def __init__(self, output_dir: str = "datasets/strands_traces"):
        """Initialize the telemetry integration.

        Args:
            output_dir: Directory for storing trace data
        """
        self.output_dir = output_dir
        self.processor = StrandsTraceProcessor(output_dir)

    def setup_with_strands(self) -> None:
        """Set up the telemetry processor with Strands SDK.

        This would be called when initializing a Strands agent.
        """
        try:
            from strands.telemetry import StrandsTelemetry

            # Initialize Strands telemetry
            telemetry = StrandsTelemetry()

            # Add our custom processor
            telemetry.add_span_processor(self.processor)

            # Optionally setup other exporters
            telemetry.setup_console_exporter()  # For debugging

            print(f"✓ Strands telemetry configured with Golden Path processor")
            print(f"  Traces will be written to: {self.output_dir}")

        except ImportError:
            print("⚠ Strands SDK not installed. Install with: pip install strands-agents")

    def get_processor(self) -> StrandsTraceProcessor:
        """Get the trace processor instance."""
        return self.processor