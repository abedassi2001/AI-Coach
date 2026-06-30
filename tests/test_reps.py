"""Tests for repetition segmentation (Phase 5)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.features.feature_pipeline import FrameFeatures
from src.features.rep_segmentation import (
    RepSegmentationPipeline,
    classify_phases,
    extract_signal,
    find_bottom_frames,
)


def _make_squat_signal(n_frames: int = 40, bottom_at: list[int] | None = None) -> list[FrameFeatures]:
    """Synthetic knee angle curve: high → low → high per rep."""
    bottoms = bottom_at or [10, 30]
    frames: list[FrameFeatures] = []
    for i in range(n_frames):
        # Start at 160°, dip to 85° at bottoms
        angle = 160.0
        for b in bottoms:
            angle = min(angle, 85.0 + abs(i - b) * 8.0)
        frames.append(
            FrameFeatures(
                frame_index=i,
                timestamp_sec=i / 10.0,
                angles={},
                derived={"knee_angle_min": angle},
            )
        )
    return frames


def test_extract_signal():
    frames = _make_squat_signal(5)
    sig = extract_signal(frames, "knee_angle_min")
    assert len(sig) == 5
    assert sig[0] > sig[2]


def test_find_bottom_frames_synthetic():
    frames = _make_squat_signal(40, bottom_at=[10, 30])
    signal = extract_signal(frames)
    bottoms = find_bottom_frames(signal, min_distance_frames=8, min_prominence=10.0)
    assert len(bottoms) >= 1
    assert 10 in bottoms or any(abs(b - 10) <= 2 for b in bottoms)


def test_classify_phases():
    frames = _make_squat_signal(20, bottom_at=[10])
    phases = classify_phases(2, 10, 18, frames)
    names = [p.phase for p in phases]
    assert "bottom" in names
    assert "descending" in names
    assert "ascending" in names


def test_rep_pipeline_counts_reps():
    frames = _make_squat_signal(50, bottom_at=[12, 37])
    pipeline = RepSegmentationPipeline(min_rep_duration_sec=0.5, min_distance_frames=10)
    result = pipeline.segment(frames, source_id="test", exercise="squat")
    assert len(result.repetitions) >= 1
    for rep in result.repetitions:
        assert rep.bottom_knee_angle < 120
        assert rep.duration_sec > 0


def test_segment_from_sample_features():
    features = Path("data/processed/features/sample_squat/features.csv")
    if not features.exists():
        pytest.skip("run compute_features first")

    pipeline = RepSegmentationPipeline(min_rep_duration_sec=0.1, min_prominence=3.0)
    result = pipeline.segment_from_features_file(features)
    assert result.output_dir.joinpath("reps.json").exists()
    assert result.frame_count == 30
