"""Body-size normalization for camera-invariant features."""

from __future__ import annotations

import math
from dataclasses import dataclass

from backend.pose.keypoint_schema import Keypoint, PoseFrame


@dataclass(frozen=True)
class BodyScale:
    """Reference lengths derived from one pose frame."""

    torso_length: float
    hip_width: float
    shoulder_width: float

    @property
    def scale(self) -> float:
        """Primary normalization divisor (torso length)."""
        return max(self.torso_length, 1e-6)


def midpoint(a: Keypoint, b: Keypoint) -> Keypoint:
    return Keypoint(
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        z=(a.z + b.z) / 2.0,
        visibility=min(a.visibility, b.visibility),
    )


def euclidean_distance(a: Keypoint, b: Keypoint) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def compute_body_scale(frame: PoseFrame, method: str = "torso_length") -> BodyScale | None:
    """
    Estimate body scale from shoulder/hip landmarks.

    Torso length = distance from mid-shoulder to mid-hip (used to normalize
    positions so the same squat depth looks similar at different camera distances).
    """
    lm = frame.landmarks
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(name in lm for name in required):
        return None

    mid_shoulder = midpoint(lm["left_shoulder"], lm["right_shoulder"])
    mid_hip = midpoint(lm["left_hip"], lm["right_hip"])
    torso = euclidean_distance(mid_shoulder, mid_hip)
    hip_w = euclidean_distance(lm["left_hip"], lm["right_hip"])
    shoulder_w = euclidean_distance(lm["left_shoulder"], lm["right_shoulder"])

    if method == "hip_shoulder":
        scale = (hip_w + shoulder_w) / 2.0
        torso = scale

    return BodyScale(
        torso_length=max(torso, 1e-6),
        hip_width=hip_w,
        shoulder_width=shoulder_w,
    )


def normalize_keypoint(kp: Keypoint, origin: Keypoint, scale: float) -> tuple[float, float]:
    """Express a keypoint relative to origin, divided by body scale."""
    return (kp.x - origin.x) / scale, (kp.y - origin.y) / scale
