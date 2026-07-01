"""Build per-frame feature tables from pose keypoint sequences."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.features.angles import compute_configured_angles, compute_derived_features, resolve_landmark
from backend.features.normalization import compute_body_scale, midpoint
from backend.pose.keypoint_schema import Keypoint, PoseFrame, PoseSequence
from backend.utils.config import load_config, resolve_path
from backend.utils.exercise_config import load_exercise_config


@dataclass
class FrameFeatures:
    """All engineered features for one video frame."""

    frame_index: int
    timestamp_sec: float
    angles: dict[str, float] = field(default_factory=dict)
    derived: dict[str, float] = field(default_factory=dict)
    torso_length: float = float("nan")

    def to_flat_dict(self, source_id: str, exercise: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source_id": source_id,
            "exercise": exercise,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "torso_length": self.torso_length,
        }
        for k, v in self.angles.items():
            row[f"angle_{k}"] = v
        for k, v in self.derived.items():
            row[k] = v
        return row


@dataclass
class FeatureExtractionResult:
    """Output of feature extraction on one recording."""

    source_id: str
    exercise: str
    frames: list[FrameFeatures]
    output_dir: Path
    csv_path: Path
    json_path: Path


def load_pose_sequence(json_path: str | Path) -> PoseSequence:
    """Load keypoints.json produced by Phase 3."""
    path = Path(json_path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for fr in data["frames"]:
        landmarks = {name: Keypoint(**vals) for name, vals in fr["landmarks"].items()}
        frames.append(
            PoseFrame(
                frame_index=fr["frame_index"],
                timestamp_sec=fr["timestamp_sec"],
                landmarks=landmarks,
                width=fr.get("width", 0),
                height=fr.get("height", 0),
            )
        )
    return PoseSequence(
        source_id=data["source_id"],
        exercise=data["exercise"],
        backend=data["backend"],
        landmark_names=data["landmark_names"],
        frames=frames,
        metadata=data.get("metadata", {}),
    )


def smooth_series(values: list[float], window: int) -> list[float]:
    """Simple moving average (odd window >= 1)."""
    if window <= 1 or len(values) <= 1:
        return values
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        chunk = [v for v in values[start:end] if v == v]  # skip NaN
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


class FeatureExtractionPipeline:
    """Compute joint angles and derived features from a PoseSequence."""

    def __init__(self, normalize_by: str = "torso_length", smoothing_window: int = 1) -> None:
        cfg = load_config()
        feat_cfg = cfg.get("features", {})
        self.normalize_by = normalize_by or feat_cfg.get("normalize_by", "torso_length")
        self.smoothing_window = smoothing_window or feat_cfg.get("smoothing_window", 1)

    def extract_from_sequence(
        self,
        sequence: PoseSequence,
        exercise_id: str | None = None,
    ) -> list[FrameFeatures]:
        exercise = exercise_id or sequence.exercise
        ex_cfg = load_exercise_config(exercise)
        angle_defs = ex_cfg.get("angles", [])

        baseline_hip_y = self._baseline_hip_y(sequence.frames)

        frame_features: list[FrameFeatures] = []
        for pose_frame in sequence.frames:
            scale = compute_body_scale(pose_frame, method=self.normalize_by)
            torso_len = scale.torso_length if scale else float("nan")

            angles = compute_configured_angles(pose_frame, angle_defs)
            derived = compute_derived_features(angles)
            derived.update(self._hip_drop_features(pose_frame, baseline_hip_y, torso_len))

            frame_features.append(
                FrameFeatures(
                    frame_index=pose_frame.frame_index,
                    timestamp_sec=pose_frame.timestamp_sec,
                    angles=angles,
                    derived=derived,
                    torso_length=torso_len,
                )
            )

        if self.smoothing_window > 1:
            self._apply_smoothing(frame_features)

        return frame_features

    @staticmethod
    def _baseline_hip_y(frames: list[PoseFrame], n: int = 8) -> float:
        """Standing hip height reference from opening frames (image y, downward +)."""
        ys: list[float] = []
        for pose_frame in frames[:n]:
            mid_hip = resolve_landmark("mid_hip", pose_frame.landmarks)
            if mid_hip is not None:
                ys.append(mid_hip.y)
        if not ys:
            return float("nan")
        return sum(ys) / len(ys)

    @staticmethod
    def _hip_drop_features(
        pose_frame: PoseFrame,
        baseline_hip_y: float,
        torso_len: float,
    ) -> dict[str, float]:
        """Normalized vertical hip travel — works for front and side cameras."""
        out: dict[str, float] = {}
        mid_hip = resolve_landmark("mid_hip", pose_frame.landmarks)
        if mid_hip is None or baseline_hip_y != baseline_hip_y or torso_len <= 0:
            return out
        out["hip_drop_norm"] = (mid_hip.y - baseline_hip_y) / torso_len
        mid_shoulder = resolve_landmark("mid_shoulder", pose_frame.landmarks)
        if mid_shoulder is not None:
            out["shoulder_hip_span"] = (mid_hip.y - mid_shoulder.y) / torso_len
        lm = pose_frame.landmarks
        if "left_hip" in lm and "right_hip" in lm:
            out["hip_width_norm"] = abs(lm["left_hip"].x - lm["right_hip"].x) / torso_len
        if "left_shoulder" in lm and "right_shoulder" in lm:
            out["shoulder_width_norm"] = abs(lm["left_shoulder"].x - lm["right_shoulder"].x) / torso_len
        if "hip_width_norm" in out and out["hip_width_norm"] > 1e-6:
            sw = out.get("shoulder_width_norm", out["hip_width_norm"])
            out["camera_frontality"] = sw / out["hip_width_norm"]
        return out

    def _apply_smoothing(self, frame_features: list[FrameFeatures]) -> None:
        if not frame_features:
            return
        angle_keys = list(frame_features[0].angles.keys())
        derived_keys: set[str] = set()
        for ff in frame_features:
            derived_keys.update(ff.derived.keys())

        for key in angle_keys:
            series = [ff.angles.get(key, float("nan")) for ff in frame_features]
            smoothed = smooth_series(series, self.smoothing_window)
            for ff, val in zip(frame_features, smoothed):
                ff.angles[key] = val

        for key in sorted(derived_keys):
            series = [ff.derived.get(key, float("nan")) for ff in frame_features]
            smoothed = smooth_series(series, self.smoothing_window)
            for ff, val in zip(frame_features, smoothed):
                ff.derived[key] = val

    def extract_from_keypoints_file(
        self,
        keypoints_path: str | Path,
        output_dir: str | Path | None = None,
        exercise_id: str | None = None,
    ) -> FeatureExtractionResult:
        sequence = load_pose_sequence(keypoints_path)
        frames = self.extract_from_sequence(sequence, exercise_id=exercise_id)

        out_dir = (
            Path(output_dir)
            if output_dir
            else resolve_path("data/processed/features") / sequence.source_id
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "features.csv"
        json_path = out_dir / "features.json"
        self._save_csv(frames, sequence.source_id, sequence.exercise, csv_path)
        self._save_json(frames, sequence, csv_path, json_path)

        return FeatureExtractionResult(
            source_id=sequence.source_id,
            exercise=sequence.exercise,
            frames=frames,
            output_dir=out_dir,
            csv_path=csv_path,
            json_path=json_path,
        )

    @staticmethod
    def _save_csv(
        frames: list[FrameFeatures],
        source_id: str,
        exercise: str,
        path: Path,
    ) -> None:
        if not frames:
            path.write_text("", encoding="utf-8")
            return
        rows = [f.to_flat_dict(source_id, exercise) for f in frames]
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _save_json(
        frames: list[FrameFeatures],
        sequence: PoseSequence,
        csv_path: Path,
        path: Path,
    ) -> None:
        payload = {
            "source_id": sequence.source_id,
            "exercise": sequence.exercise,
            "frame_count": len(frames),
            "metadata": sequence.metadata,
            "csv_path": str(csv_path),
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "timestamp_sec": f.timestamp_sec,
                    "torso_length": f.torso_length,
                    "angles": f.angles,
                    "derived": f.derived,
                }
                for f in frames
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
