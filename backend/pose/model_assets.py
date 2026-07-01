"""Download MediaPipe pose landmarker model weights."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from backend.utils.config import get_project_root

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = "models/pose/pose_landmarker_lite.task"


def get_pose_model_path() -> Path:
    """Return path to pose landmarker .task file, downloading if needed."""
    path = get_project_root() / DEFAULT_MODEL_PATH
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {path} ...")
    urllib.request.urlretrieve(POSE_MODEL_URL, path)
    return path
