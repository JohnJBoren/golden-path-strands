"""Utilities for constructing training datasets from successful paths."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any


class DatasetCreator:
    """Write successful exploration paths to a JSONL dataset."""

    def __init__(self, output_path: str | Path | None = None) -> None:
        self.output_path = Path(output_path or "dataset.jsonl")

    def write_paths(self, paths: List[Dict[str, Any]]) -> str:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf8") as fh:
            for path in paths:
                fh.write(json.dumps(path) + "\n")
        return str(self.output_path)
