"""MediaPipe BlazePose backend (Tasks API, MediaPipe >= 0.10)."""

from __future__ import annotations

import numpy as np

from backend.pose.base import PoseEstimator
from backend.pose.keypoint_schema import MEDIAPIPE_LANDMARK_NAMES, Keypoint, PoseFrame
from backend.pose.model_assets import get_pose_model_path


class MediaPipePoseEstimator(PoseEstimator):
    """Wraps MediaPipe PoseLandmarker for per-frame landmark detection."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,  # kept for API compatibility; lite/full set via model file
        static_image_mode: bool = False,
        model_path: str | None = None,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        model_asset = str(model_path) if model_path else str(get_pose_model_path())
        running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if static_image_mode
            else mp.tasks.vision.RunningMode.VIDEO
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_asset),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._running_mode = running_mode
        self._landmark_names = list(MEDIAPIPE_LANDMARK_NAMES)
        self._video_timestamp_ms = 0

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
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        rgb = frame[:, :, ::-1].copy()
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if self._running_mode == self._mp.tasks.vision.RunningMode.VIDEO:
            timestamp_ms = int(timestamp_sec * 1000)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks: dict[str, Keypoint] = {}
        for name, lm in zip(self._landmark_names, result.pose_landmarks[0]):
            landmarks[name] = Keypoint(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility) if lm.visibility is not None else 1.0,
            )

        return PoseFrame(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            landmarks=landmarks,
            width=width,
            height=height,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
