"""Configuration management utilities."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class Settings:
    model_provider: str = "stub"
    telemetry_endpoint: str | None = None
    data_dir: str = "data"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Settings object to dictionary for compatibility."""
        result = {
            "model_provider": self.model_provider,
            "telemetry_endpoint": self.telemetry_endpoint,
            "data_dir": self.data_dir,
        }
        # Add extra fields directly to the result
        result.update(self.extra)
        return result


def load_config(path: str = "config.yaml") -> Settings:
    """Load configuration from YAML file and environment variables.

    YAML parsing is optional; if :mod:`pyyaml` is unavailable the file is assumed to
    contain JSON and parsed accordingly. Missing files result in default settings.
    """
    config: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as fh:
            text = fh.read()
            if yaml is not None:
                try:
                    config = yaml.safe_load(text) or {}
                except yaml.YAMLError as e:
                    # If YAML parsing fails, try JSON as fallback
                    try:
                        config = json.loads(text or "{}")
                    except json.JSONDecodeError:
                        raise ValueError(f"Failed to parse config file as YAML or JSON: {e}")
            else:
                try:
                    config = json.loads(text or "{}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse config file as JSON (PyYAML not installed): {e}")
    prefix = "GPS_"
    env_overrides = {k[len(prefix):].lower(): v for k, v in os.environ.items() if k.startswith(prefix)}
    config.update(env_overrides)
    extra = {k: v for k, v in config.items() if k not in Settings.__dataclass_fields__}
    settings_kwargs = {k: config.get(k) for k in Settings.__dataclass_fields__ if k in config}
    settings = Settings(**{**settings_kwargs, "extra": extra})
    return settings
