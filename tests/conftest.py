"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a short synthetic MP4 for video pipeline tests."""
    import cv2
    import numpy as np

    path = tmp_path / "test_squat.mp4"
    width, height = 640, 480
    fps = 60.0
    n_frames = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = min(255, i * 8)
        cv2.putText(
            frame,
            f"frame {i}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
    writer.release()
    return path
