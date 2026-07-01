"""Reusable UI components for the gym coach app."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from frontend.video_modal import (
    clear_video_modal_request,
    queue_upload_video_callback,
    queue_annotated_video_callback,
    queue_demo_annotated_callback,
    request_annotated_video_modal,
    request_upload_video_modal,
    watch_annotated_button,
    watch_demo_annotated_button,
    watch_upload_button,
)

DIMENSION_LABELS = {
    "depth_score": "Depth",
    "knee_tracking_score": "Knee tracking",
    "torso_control_score": "Torso control",
    "symmetry_score": "Symmetry",
    "stability_score": "Stability",
    "heel_control_score": "Heel control",
}


def load_form_analysis(project_root: Path, source_id: str) -> dict | None:
    path = project_root / "data/processed/analysis" / source_id / "form_analysis.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def render_coaching_card(text_path: Path | None) -> None:
    st.markdown("#### Coaching report")
    if text_path and text_path.exists():
        st.text(text_path.read_text(encoding="utf-8"))
    else:
        st.info("Enable coaching in the sidebar and re-run analysis.")


__all__ = [
    "load_form_analysis",
    "render_coaching_card",
    "request_annotated_video_modal",
    "request_upload_video_modal",
    "clear_video_modal_request",
    "queue_upload_video_callback",
    "queue_annotated_video_callback",
    "queue_demo_annotated_callback",
    "watch_annotated_button",
    "watch_demo_annotated_button",
    "watch_upload_button",
]
