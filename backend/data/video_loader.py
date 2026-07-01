"""OpenCV-based video loading and metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    """Summary statistics for a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float

    @property
    def stem(self) -> str:
        return self.path.stem


class VideoLoader:
    """Load exercise videos and expose capture + metadata."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> cv2.VideoCapture:
        """Open the video capture (idempotent)."""
        if self._capture is None or not self._capture.isOpened():
            self._capture = cv2.VideoCapture(str(self.video_path))
            if not self._capture.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
        return self._capture

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoLoader:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_metadata(self) -> VideoMetadata:
        """Read width, height, FPS, frame count, and duration."""
        cap = self.open()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0  # fallback when container metadata is missing
        if frame_count <= 0:
            frame_count = self._count_frames_manually(cap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        duration_sec = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            path=self.video_path.resolve(),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
        )

    @staticmethod
    def _count_frames_manually(cap: cv2.VideoCapture) -> int:
        """Count frames when CAP_PROP_FRAME_COUNT is unreliable."""
        count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        return count
