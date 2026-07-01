"""Tests for safe ML model loading."""

from __future__ import annotations

from pathlib import Path

from backend.inference.model_loader import try_load_predictor


def test_try_load_missing_returns_none():
    assert try_load_predictor(Path("nonexistent/model.joblib")) is None


def test_try_load_real_model_if_present():
    path = Path("models/checkpoints/baseline/form_classifier.joblib")
    if not path.exists():
        return
    predictor = try_load_predictor(path)
    assert predictor is not None
