"""Exercise-agnostic per-repetition feature vectors for ML models."""

from __future__ import annotations

import math
import statistics
from typing import Any

from src.features.feature_pipeline import FrameFeatures
from src.features.rep_segmentation import Repetition


def _numeric_frame_keys(frames: list[FrameFeatures]) -> list[str]:
    """Discover numeric feature names from the first frame."""
    if not frames:
        return []
    keys: set[str] = set()
    for name in frames[0].angles:
        keys.add(f"angle_{name}")
    for fr in frames[0].derived:
        keys.add(fr)
    if frames[0].torso_length == frames[0].torso_length:
        keys.add("torso_length")
    return sorted(keys)


def _series(frames: list[FrameFeatures], key: str) -> list[float]:
    out: list[float] = []
    for fr in frames:
        if key.startswith("angle_"):
            val = fr.angles.get(key.removeprefix("angle_"), float("nan"))
        elif key == "torso_length":
            val = fr.torso_length
        else:
            val = fr.derived.get(key, float("nan"))
        if val == val:
            out.append(float(val))
    return out


def _at_bottom_value(frames: list[FrameFeatures], bottom_frame: int, key: str) -> float:
    for fr in frames:
        if fr.frame_index == bottom_frame:
            if key.startswith("angle_"):
                return float(fr.angles.get(key.removeprefix("angle_"), float("nan")))
            if key == "torso_length":
                return float(fr.torso_length)
            return float(fr.derived.get(key, float("nan")))
    return float("nan")


def extract_rep_features(
    rep: Repetition,
    rep_frames: list[FrameFeatures],
    exercise: str,
) -> dict[str, Any]:
    """
    Build a fixed-schema feature dict for one repetition.

    Schema is auto-derived from frame columns so new exercises only need
    Phase 4 angle definitions — no ML code changes.
    """
    row: dict[str, Any] = {
        "source_id": "",  # filled by dataset builder
        "exercise": exercise,
        "rep_id": rep.rep_id,
        "rep_duration_sec": rep.duration_sec,
        "rep_bottom_knee_angle": rep.bottom_knee_angle,
        "rep_start_frame": rep.start_frame,
        "rep_end_frame": rep.end_frame,
    }

    keys = _numeric_frame_keys(rep_frames)
    for key in keys:
        series = _series(rep_frames, key)
        prefix = f"rep_{key}"
        if series:
            row[f"{prefix}_min"] = min(series)
            row[f"{prefix}_max"] = max(series)
            row[f"{prefix}_mean"] = statistics.mean(series)
            row[f"{prefix}_std"] = statistics.pstdev(series) if len(series) > 1 else 0.0
        else:
            row[f"{prefix}_min"] = float("nan")
            row[f"{prefix}_max"] = float("nan")
            row[f"{prefix}_mean"] = float("nan")
            row[f"{prefix}_std"] = float("nan")

        bottom_val = _at_bottom_value(rep_frames, rep.bottom_frame, key)
        row[f"{prefix}_at_bottom"] = bottom_val

    return row


def feature_columns(row: dict[str, Any]) -> list[str]:
    """Numeric ML feature column names (excludes ids and label)."""
    skip = {"source_id", "exercise", "rep_id", "label"}
    return sorted(
        k
        for k, v in row.items()
        if k not in skip and isinstance(v, (int, float)) and not math.isnan(float(v))
    )
