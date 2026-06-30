"""Load project configuration from YAML files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def get_project_root() -> Path:
    """Return repository root (directory containing configs/ and src/)."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "configs" / "default.yaml").exists():
            return parent
    raise FileNotFoundError("Could not locate project root (configs/default.yaml).")


@lru_cache(maxsize=4)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and parse a YAML config file. Paths are relative to project root."""
    root = get_project_root()
    path = Path(config_path) if config_path else root / DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return data


def resolve_path(relative: str | Path, root: Path | None = None) -> Path:
    """Resolve a config path relative to the project root."""
    base = root or get_project_root()
    p = Path(relative)
    return p if p.is_absolute() else base / p
