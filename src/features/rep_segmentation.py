"""Detect exercise repetitions from per-frame feature time series."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.features.feature_pipeline import FrameFeatures
from src.utils.config import load_config, resolve_path


class RepPhase(str, Enum):
    """Movement phases within one repetition."""

    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"
    FINISHED = "finished"


@dataclass
class PhaseSegment:
    phase: str
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float


@dataclass
class Repetition:
    """One detected rep (e.g. one squat)."""

    rep_id: int
    start_frame: int
    end_frame: int
    bottom_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    bottom_knee_angle: float
    phases: list[PhaseSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class RepSegmentationResult:
    source_id: str
    exercise: str
    repetitions: list[Repetition]
    signal_column: str
    frame_count: int
    output_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "exercise": self.exercise,
            "signal_column": self.signal_column,
            "frame_count": self.frame_count,
            "rep_count": len(self.repetitions),
            "repetitions": [r.to_dict() for r in self.repetitions],
        }

    def save_json(self, path: Path | None = None) -> Path:
        out = path or (self.output_dir / "reps.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out


def load_frame_features_from_csv(path: str | Path) -> tuple[str, str, list[FrameFeatures]]:
    """Load features.csv produced by Phase 4."""
    path = Path(path)
    frames: list[FrameFeatures] = []
    source_id = ""
    exercise = ""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row.get("source_id", source_id)
            exercise = row.get("exercise", exercise)
            angles = {}
            derived = {}
            for key, val in row.items():
                if key.startswith("angle_"):
                    angles[key.removeprefix("angle_")] = float(val)
                elif key in (
                    "knee_angle_avg",
                    "knee_angle_min",
                    "knee_asymmetry_deg",
                    "hip_angle_avg",
                ):
                    derived[key] = float(val)
            frames.append(
                FrameFeatures(
                    frame_index=int(row["frame_index"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    angles=angles,
                    derived=derived,
                    torso_length=float(row.get("torso_length", "nan")),
                )
            )
    return source_id, exercise, frames


def load_frame_features_from_json(path: str | Path) -> tuple[str, str, list[FrameFeatures]]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for fr in data["frames"]:
        frames.append(
            FrameFeatures(
                frame_index=fr["frame_index"],
                timestamp_sec=fr["timestamp_sec"],
                angles=fr.get("angles", {}),
                derived=fr.get("derived", {}),
                torso_length=fr.get("torso_length", float("nan")),
            )
        )
    return data["source_id"], data["exercise"], frames


def extract_signal(frames: list[FrameFeatures], column: str = "knee_angle_min") -> list[float]:
    """Build 1D signal for rep detection (lower = deeper squat for knee angles)."""
    values: list[float] = []
    for fr in frames:
        if column in fr.derived:
            values.append(fr.derived[column])
        elif column in fr.angles:
            values.append(fr.angles[column])
        else:
            values.append(float("nan"))
    return values


def _fill_nan_linear(values: list[float]) -> list[float]:
    """Simple gap-fill so peak detection does not break on NaN."""
    if not values:
        return values
    out = values[:]
    n = len(out)
    for i, v in enumerate(out):
        if v == v:
            continue
        left = next((out[j] for j in range(i - 1, -1, -1) if out[j] == out[j]), None)
        right = next((out[j] for j in range(i + 1, n) if out[j] == out[j]), None)
        if left is not None and right is not None:
            out[i] = (left + right) / 2.0
        elif left is not None:
            out[i] = left
        elif right is not None:
            out[i] = right
        else:
            out[i] = 0.0
    return out


def find_bottom_frames(
    signal: list[float],
    min_distance_frames: int = 5,
    min_prominence: float = 5.0,
) -> list[int]:
    """
    Find squat bottom frames as local minima of knee angle signal.

    Uses scipy if available; falls back to simple scan.
    """
    if len(signal) < 3:
        return []

    clean = _fill_nan_linear(signal)
    try:
        from scipy.signal import find_peaks

        # Valleys = peaks in inverted signal
        peaks, props = find_peaks(
            [-v for v in clean],
            distance=max(1, min_distance_frames),
            prominence=max(1.0, min_prominence),
        )
        return [int(p) for p in peaks]
    except ImportError:
        bottoms: list[int] = []
        for i in range(1, len(clean) - 1):
            if clean[i] <= clean[i - 1] and clean[i] <= clean[i + 1]:
                if not bottoms or i - bottoms[-1] >= min_distance_frames:
                    bottoms.append(i)
        return bottoms


def _find_peak_before(signal: list[float], idx: int) -> int:
    """Highest signal (standing) before index."""
    best_i = 0
    best_v = -math.inf
    for i in range(0, idx + 1):
        if signal[i] > best_v:
            best_v = signal[i]
            best_i = i
    return best_i


def _find_peak_after(signal: list[float], idx: int) -> int:
    """Highest signal (standing) after index."""
    best_i = len(signal) - 1
    best_v = -math.inf
    for i in range(idx, len(signal)):
        if signal[i] > best_v:
            best_v = signal[i]
            best_i = i
    return best_i


def classify_phases(
    start: int,
    bottom: int,
    end: int,
    frames: list[FrameFeatures],
) -> list[PhaseSegment]:
    """Split rep into standing / descending / bottom / ascending / finished."""
    if end <= start:
        return []

    def t(idx: int) -> float:
        return frames[idx].timestamp_sec

    # Bottom window: ±1 frame around minimum
    b0 = max(start, bottom - 1)
    b1 = min(end, bottom + 1)

    phases: list[PhaseSegment] = []
    if start < bottom:
        phases.append(
            PhaseSegment(RepPhase.DESCENDING.value, start, b0, t(start), t(b0))
        )
    phases.append(PhaseSegment(RepPhase.BOTTOM.value, b0, b1, t(b0), t(b1)))
    if b1 < end:
        phases.append(
            PhaseSegment(RepPhase.ASCENDING.value, b1, end, t(b1), t(end))
        )
    if phases:
        phases.insert(
            0,
            PhaseSegment(RepPhase.STANDING.value, start, start, t(start), t(start)),
        )
        phases.append(
            PhaseSegment(RepPhase.FINISHED.value, end, end, t(end), t(end)),
        )
    return phases


class RepSegmentationPipeline:
    """Segment repetitions from features time series."""

    def __init__(
        self,
        signal_column: str = "knee_angle_min",
        min_rep_duration_sec: float | None = None,
        min_distance_frames: int = 5,
        min_prominence: float = 5.0,
    ) -> None:
        cfg = load_config()
        rep_cfg = cfg.get("repetition", {})
        self.signal_column = signal_column
        self.min_rep_duration_sec = (
            min_rep_duration_sec
            if min_rep_duration_sec is not None
            else rep_cfg.get("min_rep_duration_sec", 1.0)
        )
        self.min_distance_frames = min_distance_frames
        self.min_prominence = min_prominence

    def segment(
        self,
        frames: list[FrameFeatures],
        source_id: str = "unknown",
        exercise: str = "squat",
    ) -> RepSegmentationResult:
        signal = extract_signal(frames, self.signal_column)
        bottoms = find_bottom_frames(
            signal,
            min_distance_frames=self.min_distance_frames,
            min_prominence=self.min_prominence,
        )

        repetitions: list[Repetition] = []
        for rep_id, bottom_idx in enumerate(bottoms, start=1):
            start_idx = _find_peak_before(signal, bottom_idx)
            end_idx = _find_peak_after(signal, bottom_idx)

            if start_idx >= end_idx:
                continue

            duration = frames[end_idx].timestamp_sec - frames[start_idx].timestamp_sec
            if duration < self.min_rep_duration_sec:
                continue

            phases = classify_phases(start_idx, bottom_idx, end_idx, frames)
            repetitions.append(
                Repetition(
                    rep_id=rep_id,
                    start_frame=frames[start_idx].frame_index,
                    end_frame=frames[end_idx].frame_index,
                    bottom_frame=frames[bottom_idx].frame_index,
                    start_time_sec=frames[start_idx].timestamp_sec,
                    end_time_sec=frames[end_idx].timestamp_sec,
                    duration_sec=round(duration, 4),
                    bottom_knee_angle=round(signal[bottom_idx], 2),
                    phases=phases,
                )
            )

        out_dir = resolve_path("data/processed/reps") / source_id
        out_dir.mkdir(parents=True, exist_ok=True)

        return RepSegmentationResult(
            source_id=source_id,
            exercise=exercise,
            repetitions=repetitions,
            signal_column=self.signal_column,
            frame_count=len(frames),
            output_dir=out_dir,
        )

    def segment_from_features_file(
        self,
        features_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> RepSegmentationResult:
        path = Path(features_path)
        if path.suffix.lower() == ".csv":
            source_id, exercise, frames = load_frame_features_from_csv(path)
        else:
            source_id, exercise, frames = load_frame_features_from_json(path)

        result = self.segment(frames, source_id=source_id, exercise=exercise)
        if output_dir is not None:
            result.output_dir = Path(output_dir)
            result.output_dir.mkdir(parents=True, exist_ok=True)
        result.save_json()
        return result
