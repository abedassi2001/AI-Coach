"""Verify repo layout and entry points after restructure."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "rel",
    [
        "backend/__init__.py",
        "backend/inference/video_pipeline.py",
        "frontend/streamlit_app.py",
        "app/streamlit_app.py",
        "scripts/run_app.py",
        "configs/default.yaml",
        "models/README.md",
    ],
)
def test_required_paths_exist(rel: str) -> None:
    path = ROOT / rel
    assert path.exists(), f"Missing required path: {rel}"


def test_run_app_points_at_frontend() -> None:
    text = (ROOT / "scripts" / "run_app.py").read_text(encoding="utf-8")
    assert 'PROJECT_ROOT / "frontend" / "streamlit_app.py"' in text


def test_compat_shim_targets_frontend() -> None:
    text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "frontend" in text
    assert (ROOT / "frontend" / "streamlit_app.py").exists()
