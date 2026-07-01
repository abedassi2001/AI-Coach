"""Load exercise-specific configuration (landmark groups, angles, form rules)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from backend.utils.config import get_project_root


def list_exercises() -> list[str]:
    """Return exercise IDs with config files in configs/exercises/."""
    root = get_project_root() / "configs" / "exercises"
    return sorted(
        p.stem
        for p in root.glob("*.yaml")
        if p.stem not in ("_template",) and not p.name.startswith("_")
    )


@lru_cache(maxsize=16)
def load_exercise_config(exercise_id: str) -> dict[str, Any]:
    """Load YAML config for one exercise (e.g. squat, deadlift)."""
    root = get_project_root()
    path = root / "configs" / "exercises" / f"{exercise_id}.yaml"
    if not path.exists():
        available = ", ".join(list_exercises()) or "(none)"
        raise FileNotFoundError(
            f"Exercise config not found: {exercise_id!r}. Available: {available}"
        )
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Exercise config must be a mapping: {path}")
    data.setdefault("id", exercise_id)
    return data
