"""Joint angle computation from pose landmarks."""

from __future__ import annotations

import math
from typing import Any

from src.features.normalization import midpoint
from src.pose.keypoint_schema import Keypoint, PoseFrame

# Virtual landmark names resolved at runtime (not in MediaPipe output).
VIRTUAL_LANDMARKS = frozenset({"mid_hip", "mid_shoulder", "vertical_up"})


def angle_at_point_degrees(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    """
    Interior angle at point **b** formed by segments b→a and b→c.

    Example — knee angle:
        a = hip, b = knee, c = ankle
        Small angle ≈ deep squat; ~180° ≈ locked-out leg.
    """
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return float("nan")
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def resolve_landmark(name: str, landmarks: dict[str, Keypoint]) -> Keypoint | None:
    """Map config landmark name to Keypoint, including derived midpoints."""
    if name in landmarks:
        return landmarks[name]
    if name == "mid_hip":
        if "left_hip" not in landmarks or "right_hip" not in landmarks:
            return None
        return midpoint(landmarks["left_hip"], landmarks["right_hip"])
    if name == "mid_shoulder":
        if "left_shoulder" not in landmarks or "right_shoulder" not in landmarks:
            return None
        return midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
    if name == "vertical_up":
        # Synthetic point directly above mid_hip (image y decreases upward).
        mid_hip = resolve_landmark("mid_hip", landmarks)
        if mid_hip is None:
            return None
        return Keypoint(x=mid_hip.x, y=mid_hip.y - 0.1, z=mid_hip.z, visibility=mid_hip.visibility)
    return None


def compute_configured_angles(
    frame: PoseFrame,
    angle_definitions: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute all angles listed in an exercise config (e.g. squat.yaml)."""
    results: dict[str, float] = {}
    for spec in angle_definitions:
        name = spec["name"]
        points = spec["points"]
        if len(points) != 3:
            continue
        resolved = [resolve_landmark(p, frame.landmarks) for p in points]
        if any(p is None for p in resolved):
            results[name] = float("nan")
            continue
        results[name] = angle_at_point_degrees(resolved[0], resolved[1], resolved[2])
    return results


def _valid_knee_angle(value: float, min_deg: float = 35.0, max_deg: float = 175.0) -> bool:
    """Reject frontal-view / collinearity artifacts (e.g. 6 deg 'depth')."""
    return value == value and min_deg <= value <= max_deg


def compute_derived_features(angles: dict[str, float]) -> dict[str, float]:
    """Extra scalar features built from primary angles (squat-focused, reusable)."""
    left_k = angles.get("left_knee", float("nan"))
    right_k = angles.get("right_knee", float("nan"))
    valid_knees = [v for v in (left_k, right_k) if _valid_knee_angle(v)]

    left_h = angles.get("left_hip", float("nan"))
    right_h = angles.get("right_hip", float("nan"))
    hip_values = [v for v in (left_h, right_h) if not math.isnan(v)]
    hip_avg = sum(hip_values) / len(hip_values) if hip_values else float("nan")

    derived: dict[str, float] = {}
    if valid_knees:
        derived["knee_angle_avg"] = sum(valid_knees) / len(valid_knees)
        derived["knee_angle_min"] = min(valid_knees)
    elif hip_values:
        # Frontal / occluded knees: hip flexion tracks depth reasonably well.
        derived["knee_angle_avg"] = hip_avg
        derived["knee_angle_min"] = min(hip_values)
        derived["knee_from_hip_proxy"] = 1.0

    if not math.isnan(left_k) and not math.isnan(right_k):
        derived["knee_asymmetry_deg"] = abs(left_k - right_k)

    if hip_values:
        derived["hip_angle_avg"] = hip_avg

    # Depth signal for rep segmentation + ML — knees first; hips only if knees unreliable.
    if valid_knees:
        derived["squat_depth_angle"] = min(valid_knees)
    elif hip_values:
        derived["squat_depth_angle"] = min(hip_values)
        derived["knee_from_hip_proxy"] = 1.0

    return derived
