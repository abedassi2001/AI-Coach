"""MediaPipe BlazePose backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.pose.base import PoseEstimator
from src.pose.keypoint_schema import MEDIAPIPE_LANDMARK_NAMES, Keypoint, PoseFrame


class MediaPipePoseEstimator(PoseEstimator):
    """Wraps MediaPipe Pose for per-frame landmark detection."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = False,
    ) -> None:
        import mediapipe as mp

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmark_names = list(MEDIAPIPE_LANDMARK_NAMES)

    @property
    def backend_name(self) -> str:
        return "mediapipe"

    @property
    def landmark_names(self) -> list[str]:
        return self._landmark_names

    def estimate(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
    ) -> PoseFrame | None:
        import mediapipe as mp

        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        rgb = frame[:, :, ::-1]  # BGR -> RGB
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None

        landmarks: dict[str, Keypoint] = {}
        for name, lm in zip(self._landmark_names, results.pose_landmarks.landmark):
            landmarks[name] = Keypoint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(getattr(lm, "visibility", 1.0)),
            )

        return PoseFrame(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            landmarks=landmarks,
            width=width,
            height=height,
        )

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
