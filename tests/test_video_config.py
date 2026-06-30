"""Tests for config utilities and pure video math (no OpenCV required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.video_math import compute_sample_stride
from src.utils.config import get_project_root, load_config, resolve_path


def test_load_config_has_video_section():
    cfg = load_config()
    assert "video" in cfg
    assert cfg["video"]["target_fps"] == 30


def test_get_project_root():
    root = get_project_root()
    assert (root / "configs" / "default.yaml").exists()
    assert (root / "src" / "data" / "video_loader.py").exists()


def test_resolve_path():
    root = get_project_root()
    p = resolve_path("data/interim")
    assert p == root / "data" / "interim"


def test_compute_sample_stride():
    assert compute_sample_stride(60.0, 30.0) == 2
    assert compute_sample_stride(24.0, 30.0) == 1
    assert compute_sample_stride(30.0, 30.0) == 1
