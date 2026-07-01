"""Factory for pose estimation backends."""

from __future__ import annotations

from typing import Any

from backend.pose.base import PoseEstimator
from backend.pose.keypoint_schema import PoseBackend


def create_pose_estimator(
    backend: str = PoseBackend.MEDIAPIPE.value,
    **kwargs: Any,
) -> PoseEstimator:
    """
    Create a pose estimator by backend name.

    kwargs are passed to the concrete estimator (e.g. min_detection_confidence).
    """
    name = backend.lower().strip()
    if name == PoseBackend.MEDIAPIPE.value:
        from backend.pose.mediapipe_estimator import MediaPipePoseEstimator

        return MediaPipePoseEstimator(**kwargs)
    if name == PoseBackend.YOLO.value:
        raise NotImplementedError(
            "YOLO Pose backend is not implemented yet. Use backend='mediapipe'."
        )
    raise ValueError(f"Unknown pose backend: {backend!r}. Choose: mediapipe, yolo")
