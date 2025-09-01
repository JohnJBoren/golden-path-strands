"""Logging utilities for exploration runs."""
from __future__ import annotations

import json
from typing import Dict, List, Any


class ExplorationLogger:
    """Collects structured logs for decision points and successful paths."""

    def __init__(self) -> None:
        self._logs: List[str] = []
        self._successful: List[Dict[str, Any]] = []

    def log_decision_point(self, agent_result: Any, context: Dict) -> None:
        entry = {"event": "decision_point", "result": agent_result, "context": context}
        self._logs.append(json.dumps(entry))

    def mark_path_successful(self, score: float, metadata: Dict) -> None:
        entry = {"score": score, "metadata": metadata}
        self._logs.append(json.dumps({"event": "successful_path", **entry}))
        self._successful.append(entry)

    def get_successful_paths(self) -> List[Dict[str, Any]]:
        return list(self._successful)
