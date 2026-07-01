"""Draw pose skeleton overlays on video frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from backend.pose.keypoint_schema import POSE_CONNECTIONS, Keypoint, PoseFrame

if TYPE_CHECKING:
    pass

# BGR colors
_COLOR_JOINT = (0, 255, 128)
_COLOR_BONE = (255, 180, 0)
_COLOR_LOW_CONF = (0, 0, 255)
_MIN_VISIBILITY = 0.5


def _to_pixel(kp: Keypoint, width: int, height: int) -> tuple[int, int]:
    return int(kp.x * width), int(kp.y * height)


def draw_skeleton_on_frame(
    frame: np.ndarray,
    pose_frame: PoseFrame | None,
    min_visibility: float = _MIN_VISIBILITY,
    joint_radius: int = 4,
    bone_thickness: int = 2,
) -> np.ndarray:
    """Return a copy of frame with skeleton overlay."""
    out = frame.copy()
    if pose_frame is None:
        return out

    height, width = out.shape[:2]
    landmarks = pose_frame.landmarks

    for a_name, b_name in POSE_CONNECTIONS:
        a, b = landmarks.get(a_name), landmarks.get(b_name)
        if a is None or b is None:
            continue
        if a.visibility < min_visibility or b.visibility < min_visibility:
            continue
        pt_a = _to_pixel(a, width, height)
        pt_b = _to_pixel(b, width, height)
        cv2.line(out, pt_a, pt_b, _COLOR_BONE, bone_thickness, cv2.LINE_AA)

    for name, kp in landmarks.items():
        if kp.visibility < min_visibility:
            continue
        pt = _to_pixel(kp, width, height)
        color = _COLOR_JOINT if kp.visibility >= 0.7 else _COLOR_LOW_CONF
        cv2.circle(out, pt, joint_radius, color, -1, cv2.LINE_AA)

    cv2.putText(
        out,
        f"frame {pose_frame.frame_index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
