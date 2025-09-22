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
        """Write paths to JSONL file with proper error handling."""
        temp_path = None
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first to ensure atomicity
            temp_path = self.output_path.with_suffix('.tmp')
            with temp_path.open("w", encoding="utf8") as fh:
                for path in paths:
                    fh.write(json.dumps(path) + "\n")

            # Move temp file to final location
            temp_path.replace(self.output_path)
            return str(self.output_path)

        except Exception as e:
            # Clean up temp file if it exists
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to write dataset: {e}") from e
