"""Real-time JSONL streaming writer for agent traces and golden paths."""
from __future__ import annotations

import json
import gzip
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, TextIO
from datetime import datetime
from threading import Lock
import asyncio
from dataclasses import dataclass, asdict


@dataclass
class StreamConfig:
    """Configuration for JSONL stream writer."""
    output_dir: str = "datasets/golden_paths"
    max_file_size_mb: int = 100
    rotation_interval_hours: int = 24
    compression: bool = True
    buffer_size: int = 100
    flush_interval_seconds: int = 30
    include_metadata: bool = True


class JSONLStreamWriter:
    """High-performance JSONL writer with rotation and compression.

    This class provides real-time streaming of agent traces to JSONL files
    with automatic rotation, compression, and buffering for optimal performance.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize the JSONL stream writer.

        Args:
            config: Configuration for the stream writer
        """
        self.config = config or StreamConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File management
        self.current_file: Optional[TextIO] = None
        self.current_file_path: Optional[Path] = None
        self.current_file_size: int = 0
        self.rotation_time: datetime = datetime.now()
        self.rotation_lock = Lock()  # Add lock for thread-safe rotation

        # Buffering
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_lock = Lock()
        self.last_flush_time = time.time()

        # Track background task
        self.flush_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "total_records": 0,
            "total_bytes": 0,
            "files_created": 0,
            "golden_paths": 0,
            "failed_paths": 0,
            "partial_paths": 0
        }

        # Initialize first file
        self._rotate_file()

        # Start async flush task
        self._start_flush_task()

    def write(self, data: Dict[str, Any]) -> None:
        """Write a single record to the stream.

        Args:
            data: Dictionary to write as JSON
        """
        with self.buffer_lock:
            # Add metadata if configured
            if self.config.include_metadata:
                data = self._add_metadata(data)

            self.buffer.append(data)

            # Update statistics
            self._update_stats(data)

            # Check if we need to flush
            if len(self.buffer) >= self.config.buffer_size:
                self._flush_buffer()

    def write_batch(self, records: List[Dict[str, Any]]) -> None:
        """Write multiple records to the stream.

        Args:
            records: List of dictionaries to write
        """
        for record in records:
            self.write(record)

    def write_agent_trace(self, trace: Dict[str, Any]) -> None:
        """Write an agent trace with special handling.

        Args:
            trace: Agent trace dictionary
        """
        # Enrich trace with analysis
        trace = self._analyze_trace(trace)

        # Mark golden paths
        if self._is_golden_path(trace):
            trace["golden_path"] = True
            self.stats["golden_paths"] += 1

        self.write(trace)

    def _add_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to a record.

        Args:
            data: Original data

        Returns:
            Data with metadata added
        """
        metadata = {
            "_timestamp": datetime.now().isoformat(),
            "_session_id": getattr(self, "session_id", "default"),
            "_writer_version": "1.0.0"
        }

        # Don't modify original data
        return {**metadata, **data}

    def _analyze_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a trace and add derived metrics.

        Args:
            trace: Agent trace

        Returns:
            Trace with analysis added
        """
        analysis = {}

        # Calculate decision efficiency
        if "decision_sequence" in trace:
            decisions = trace["decision_sequence"]
            analysis["decision_count"] = len(decisions)

            # Calculate average tokens per decision
            total_tokens = sum(
                d.get("tokens_used", {}).get("prompt", 0) +
                d.get("tokens_used", {}).get("completion", 0)
                for d in decisions
            )
            analysis["avg_tokens_per_decision"] = (
                total_tokens / len(decisions) if decisions else 0
            )

            # Calculate tool usage patterns
            tool_calls = []
            for d in decisions:
                tool_calls.extend(d.get("tool_calls", []))

            analysis["total_tool_calls"] = len(tool_calls)
            analysis["unique_tools"] = len(set(
                t.get("tool", "") for t in tool_calls
            ))

        # Add token efficiency metrics
        if "total_tokens" in trace:
            tokens = trace["total_tokens"]
            total = tokens.get("prompt", 0) + tokens.get("completion", 0)
            analysis["token_efficiency"] = (
                tokens.get("completion", 0) / total if total > 0 else 0
            )

        # Add time efficiency
        if "total_time_ms" in trace:
            analysis["speed_category"] = self._categorize_speed(
                trace["total_time_ms"]
            )

        trace["_analysis"] = analysis
        return trace

    def _is_golden_path(self, trace: Dict[str, Any]) -> bool:
        """Determine if a trace represents a golden path.

        Args:
            trace: Agent trace

        Returns:
            True if this is a golden path
        """
        # Check outcome
        if trace.get("outcome") != "successful":
            return False

        # Check evaluation score if available
        if "evaluation_score" in trace:
            if trace["evaluation_score"] < 0.8:
                return False

        # Check efficiency metrics
        analysis = trace.get("_analysis", {})

        # Fast execution is good
        if analysis.get("speed_category") == "fast":
            return True

        # Low token usage with success is good
        if analysis.get("token_efficiency", 0) > 0.7:
            return True

        # Few decisions with success is good
        if analysis.get("decision_count", 999) <= 3:
            return True

        return False

    def _categorize_speed(self, time_ms: float) -> str:
        """Categorize execution speed.

        Args:
            time_ms: Execution time in milliseconds

        Returns:
            Speed category
        """
        if time_ms < 1000:
            return "fast"
        elif time_ms < 5000:
            return "normal"
        elif time_ms < 10000:
            return "slow"
        else:
            return "very_slow"

    def _update_stats(self, data: Dict[str, Any]) -> None:
        """Update internal statistics.

        Args:
            data: Data being written
        """
        self.stats["total_records"] += 1

        # Track outcomes
        outcome = data.get("outcome", "")
        if outcome == "failed":
            self.stats["failed_paths"] += 1
        elif outcome == "partial":
            self.stats["partial_paths"] += 1

    def _flush_buffer(self) -> None:
        """Flush the buffer to disk."""
        if not self.buffer:
            return

        # Check if we need to rotate file
        self._check_rotation()

        # Write buffered data
        for record in self.buffer:
            line = json.dumps(record) + "\n"
            self.current_file.write(line)
            self.current_file_size += len(line.encode('utf-8'))
            self.stats["total_bytes"] += len(line.encode('utf-8'))

        self.current_file.flush()
        self.buffer.clear()
        self.last_flush_time = time.time()

    def _check_rotation(self) -> None:
        """Check if file rotation is needed."""
        with self.rotation_lock:
            # Size-based rotation
            size_limit_bytes = self.config.max_file_size_mb * 1024 * 1024
            if self.current_file_size >= size_limit_bytes:
                self._rotate_file()
                return

            # Time-based rotation
            hours_elapsed = (datetime.now() - self.rotation_time).total_seconds() / 3600
            if hours_elapsed >= self.config.rotation_interval_hours:
                self._rotate_file()

    def _rotate_file(self) -> None:
        """Rotate to a new file (must be called with rotation_lock held)."""
        # Close current file if open
        if self.current_file:
            self.current_file.close()

            # Compress if configured
            if self.config.compression and self.current_file_path:
                self._compress_file(self.current_file_path)

        # Create new file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"golden_paths_{timestamp}.jsonl"
        self.current_file_path = self.output_dir / filename

        self.current_file = open(self.current_file_path, 'w', encoding='utf-8')
        self.current_file_size = 0
        self.rotation_time = datetime.now()
        self.stats["files_created"] += 1

        # Write header with metadata
        header = {
            "_file_type": "golden_path_traces",
            "_created": datetime.now().isoformat(),
            "_version": "1.0.0",
            "_config": asdict(self.config) if isinstance(self.config, StreamConfig) else self.config
        }
        self.current_file.write(json.dumps(header) + "\n")

    def _compress_file(self, file_path: Path) -> None:
        """Compress a file using gzip.

        Args:
            file_path: Path to file to compress
        """
        compressed_path = file_path.with_suffix('.jsonl.gz')

        try:
            with open(file_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
                # Copy in chunks to handle large files efficiently
                shutil.copyfileobj(f_in, f_out)

            # Only remove original file after successful compression
            file_path.unlink()
        except Exception as e:
            # Clean up partial compressed file on error
            if compressed_path.exists():
                compressed_path.unlink()
            raise RuntimeError(f"Failed to compress file: {e}") from e

        print(f"✓ Compressed {file_path.name} -> {compressed_path.name}")

    def _start_flush_task(self) -> None:
        """Start async task for periodic flushing."""
        async def flush_periodically():
            while True:
                await asyncio.sleep(self.config.flush_interval_seconds)
                with self.buffer_lock:
                    if self.buffer:
                        self._flush_buffer()

        # Start task in background
        try:
            loop = asyncio.get_running_loop()
            self.flush_task = loop.create_task(flush_periodically())
        except RuntimeError:
            # No async loop running, flush will happen on buffer size
            logger.debug("no_async_loop",
                        message="No async loop running, periodic flush disabled")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "buffer_size": len(self.buffer),
            "current_file": str(self.current_file_path),
            "current_file_size_mb": self.current_file_size / (1024 * 1024),
            "success_rate": (
                self.stats["golden_paths"] / self.stats["total_records"]
                if self.stats["total_records"] > 0 else 0
            )
        }

    def close(self) -> None:
        """Close the writer and flush remaining data."""
        # Cancel background flush task if it exists
        if self.flush_task and not self.flush_task.done():
            self.flush_task.cancel()

        with self.buffer_lock:
            self._flush_buffer()

        if self.current_file:
            self.current_file.close()
            self.current_file = None

            # Final compression if needed
            if self.config.compression and self.current_file_path:
                self._compress_file(self.current_file_path)

        print(f"✓ JSONLStreamWriter closed")
        print(f"  Total records: {self.stats['total_records']}")
        print(f"  Golden paths: {self.stats['golden_paths']}")
        print(f"  Files created: {self.stats['files_created']}")
        print(f"  Total size: {self.stats['total_bytes'] / (1024*1024):.2f} MB")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()