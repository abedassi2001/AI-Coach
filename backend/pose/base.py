"""Abstract pose estimator interface — swap backends without changing the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from backend.pose.keypoint_schema import PoseFrame


class PoseEstimator(ABC):
    """Exercise-agnostic pose backend (MediaPipe, YOLO, etc.)."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier stored in PoseSequence.backend."""

    @property
    @abstractmethod
    def landmark_names(self) -> list[str]:
        """Ordered landmark names produced by this backend."""

    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
    ) -> PoseFrame | None:
        """Run pose estimation on a single BGR image frame."""

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
