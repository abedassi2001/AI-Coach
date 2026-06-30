"""Exercise-agnostic pose landmark definitions and serialization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class PoseBackend(str, Enum):
    """Supported pose estimation backends."""

    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"


# MediaPipe BlazePose 33-landmark body model (exercise-agnostic).
MEDIAPIPE_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

# Skeleton edges for visualization (pairs of landmark names).
POSE_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("left_shoulder", "right_shoulder"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
)


@dataclass(frozen=True)
class Keypoint:
    """Single body landmark in image-normalized coordinates."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "visibility": self.visibility}


@dataclass
class PoseFrame:
    """Pose landmarks detected in one video frame."""

    frame_index: int
    timestamp_sec: float
    landmarks: dict[str, Keypoint]
    width: int = 0
    height: int = 0

    def get(self, name: str) -> Keypoint | None:
        return self.landmarks.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "width": self.width,
            "height": self.height,
            "landmarks": {k: v.to_dict() for k, v in self.landmarks.items()},
        }


@dataclass
class PoseSequence:
    """
    Full pose time-series for one recording.

    Exercise type is metadata only at this stage — pose extraction is the same
    for squats, deadlifts, etc. Exercise-specific rules attach in later phases.
    """

    source_id: str
    exercise: str
    backend: str
    landmark_names: list[str]
    frames: list[PoseFrame] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "exercise": self.exercise,
            "backend": self.backend,
            "landmark_names": self.landmark_names,
            "frame_count": len(self.frames),
            "metadata": self.metadata,
            "frames": [f.to_dict() for f in self.frames],
        }

    def iter_landmark_rows(self) -> Iterator[dict[str, Any]]:
        """Flatten to one row per (frame, landmark) for CSV export."""
        for frame in self.frames:
            for name, kp in frame.landmarks.items():
                yield {
                    "source_id": self.source_id,
                    "exercise": self.exercise,
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "landmark": name,
                    "x": kp.x,
                    "y": kp.y,
                    "z": kp.z,
                    "visibility": kp.visibility,
                }

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    def save_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_id",
            "exercise",
            "frame_index",
            "timestamp_sec",
            "landmark",
            "x",
            "y",
            "z",
            "visibility",
        ]
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.iter_landmark_rows())
        return out

    def save(self, output_dir: str | Path, output_format: str = "json") -> dict[str, Path]:
        """Save keypoints in configured format(s)."""
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}
        fmt = output_format.lower()
        if fmt in ("json", "both"):
            saved["json"] = self.save_json(base / "keypoints.json")
        if fmt in ("csv", "both"):
            saved["csv"] = self.save_csv(base / "keypoints.csv")
        if fmt not in ("json", "csv", "both"):
            raise ValueError(f"Unsupported output_format: {output_format}")
        return saved
