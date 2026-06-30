"""Tests for rep-level features and baseline classifier."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.feature_pipeline import FrameFeatures
from src.features.rep_features import extract_rep_features
from src.features.rep_segmentation import Repetition
from src.models.baseline_classifier import BaselineFormClassifier
from src.training.synthetic import augment_labeled_reps
from src.training.train_baseline import train_baseline_classifier


def _sample_rep_frames() -> list[FrameFeatures]:
    return [
        FrameFeatures(0, 0.0, {"torso_lean": 20.0}, {"knee_angle_min": 150.0}),
        FrameFeatures(5, 0.2, {"torso_lean": 25.0}, {"knee_angle_min": 85.0}),
        FrameFeatures(10, 0.4, {"torso_lean": 22.0}, {"knee_angle_min": 140.0}),
    ]


def test_extract_rep_features_schema():
    rep = Repetition(
        rep_id=1, start_frame=0, end_frame=10, bottom_frame=5,
        start_time_sec=0.0, end_time_sec=0.4, duration_sec=0.4, bottom_knee_angle=85.0,
    )
    row = extract_rep_features(rep, _sample_rep_frames(), "squat")
    assert row["exercise"] == "squat"
    assert "rep_knee_angle_min_min" in row
    assert "rep_knee_angle_min_at_bottom" in row
    assert row["rep_knee_angle_min_at_bottom"] == pytest.approx(85.0)


def test_baseline_classifier_fit_predict():
    rows = []
    for i in range(12):
        good = i % 2 == 0
        rows.append(
            {
                "exercise": "squat" if i < 8 else "deadlift",
                "rep_id": i,
                "source_id": f"vid_{i // 3}",
                "rep_knee_angle_min_at_bottom": 75.0 if good else 110.0,
                "rep_angle_torso_lean_max": 20.0 if good else 55.0,
                "rep_duration_sec": 2.0,
                "label": "good" if good else "bad",
            }
        )
    df = pd.DataFrame(rows)
    model = BaselineFormClassifier(model_type="logistic")
    model.fit(df, df["label"])
    preds = model.predict(df.iloc[:2])
    assert len(preds) == 2


def test_contrastive_augment_preserves_good_labels():
    rows = [
        {
            "source_id": "sample_squat",
            "exercise": "squat",
            "rep_id": 1,
            "label": "good",
            "rep_bottom_knee_angle": 72.0,
            "rep_knee_angle_min_at_bottom": 70.0,
            "rep_knee_asymmetry_deg_at_bottom": 12.0,
            "rep_angle_torso_lean_at_bottom": 20.0,
            "rep_knee_angle_min_std": 8.0,
        }
    ]
    df = augment_labeled_reps(pd.DataFrame(rows), target_size=12, mode="contrastive", variants_per_rep=3)
    good = df[df["label"] == "good"]
    bad = df[df["label"] == "bad"]
    assert len(good) >= 1
    assert len(bad) >= 3
    assert "synthetic_mistake" in bad.columns or bad["synthetic"].any()


def test_train_baseline_with_synthetic_augment(tmp_path, monkeypatch):
    """End-to-end train on tiny synthetic set."""
    rows = []
    for i in range(4):
        rows.append(
            {
                "source_id": f"s{i}",
                "exercise": "squat",
                "rep_id": i + 1,
                "rep_knee_angle_min_at_bottom": 80.0 if i % 2 == 0 else 105.0,
                "rep_angle_torso_lean_max": 18.0 if i % 2 == 0 else 50.0,
                "rep_duration_sec": 1.5,
                "label": "good" if i % 2 == 0 else "bad",
            }
        )
    df = augment_labeled_reps(pd.DataFrame(rows), target_size=20, mode="noise")

    def fake_build(self, source_ids=None):
        return df

    monkeypatch.setattr(
        "src.training.train_baseline.RepDatasetBuilder.build",
        fake_build,
    )
    monkeypatch.setattr(
        "src.training.train_baseline.RepDatasetBuilder.labeled_only",
        lambda self, d: d,
    )

    cfg = {
        "model": {"type": "logistic", "random_state": 0},
        "training": {"min_samples": 8, "test_size": 0.25},
        "paths": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "reports_dir": str(tmp_path / "reports"),
        },
    }
    result = train_baseline_classifier(config=cfg)
    assert result.model_path.exists()
    assert result.metrics["accuracy"] >= 0.0
