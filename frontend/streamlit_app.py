#!/usr/bin/env python3
"""Iron Form Coach — portfolio-ready squat analysis UI (Streamlit)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from frontend.analysis_view import build_full_view, build_summary
from frontend.components import load_form_analysis
from frontend.gym_theme import inject_gym_theme, render_hero
from frontend.ui_panels import render_full_analysis_panel, show_summary_modal, _reset_analysis_state
from frontend.ui_states import render_empty_state, render_error_state, render_loading_state
from frontend.video_modal import (
    clear_video_modal_request,
    render_video_flash_message,
    render_video_modal_if_requested,
    watch_annotated_button,
    watch_demo_annotated_button,
    watch_upload_button,
)
from backend.inference.video_pipeline import (
    list_demo_videos,
    load_existing_result,
    run_full_pipeline,
    stage_uploaded_video,
)
from backend.utils.web_video import resolve_playable_path

AnalysisStatus = Literal["idle", "uploading", "analyzing", "complete", "error"]

st.set_page_config(
    page_title="Iron Form Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_gym_theme()

_DEFAULTS = {
    "analysis_result": None,
    "analysis_status": "idle",
    "pending_summary_modal": False,
    "show_full_analysis": False,
    "analysis_error": None,
    "evaluation_video_path": None,
    "raw_video_path": None,
    "video_modal_request": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _show_video_inline(source: Path | bytes, *, filename: str = "video.mp4") -> None:
    try:
        path = resolve_playable_path(source, filename=filename)
        st.video(str(path.resolve()))
    except FileNotFoundError:
        st.warning("Video file not found.")
    except Exception as exc:
        st.error(f"Could not play video: {exc}")


def _complete_analysis(result) -> None:
    clear_video_modal_request()
    st.session_state.analysis_result = result
    st.session_state.analysis_status = "complete"
    st.session_state.pending_summary_modal = True
    st.session_state.show_full_analysis = False
    st.session_state.analysis_error = None
    if result.video_path:
        st.session_state.raw_video_path = str(result.video_path)
    if result.evaluation_video:
        st.session_state.evaluation_video_path = str(result.evaluation_video)


def _run_pipeline(
    video_path: Path,
    source_id: str,
    exercise: str,
    coaching_provider: str,
    include_coaching: bool,
    on_progress,
) -> None:
    st.session_state.analysis_status = "analyzing"
    result = run_full_pipeline(
        video_path,
        exercise=exercise,
        source_id=source_id,
        coaching_provider=coaching_provider,
        generate_coaching_report=include_coaching,
        on_progress=on_progress,
    )
    _complete_analysis(result)


def _load_cached_demo_callback() -> None:
    choice = st.session_state.get("demo_choice")
    if not choice:
        st.session_state.flash_video_error = "Select a sample workout first."
        return
    loaded = load_existing_result(choice)
    if loaded:
        _complete_analysis(loaded)
    else:
        st.session_state.flash_video_error = "No cached results — run analysis first."


def _queue_demo_analysis() -> None:
    st.session_state.demo_analysis_requested = True


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🏋️ GYM COACH")
    exercise = st.selectbox("Exercise", ["squat"], index=0)
    coaching_provider = st.selectbox("Coaching voice", ["template", "auto", "ollama", "openai"], index=0)
    include_coaching = st.checkbox("Generate coaching report", value=True)
    st.divider()
    st.markdown(
        """
        **Your flow**
        1. Upload a squat video
        2. Tap **Analyze my squat**
        3. Read the summary popup
        4. Open **full analysis** for details

        *Side-view filming works best.*
        """
    )

render_hero()

# Single top-level video modal (must run before nested summary dialog)
render_video_modal_if_requested()
render_video_flash_message()

status: AnalysisStatus = st.session_state.analysis_status

if st.session_state.pending_summary_modal and st.session_state.analysis_result and not st.session_state.get("video_modal_request"):
    analysis = load_form_analysis(PROJECT_ROOT, st.session_state.analysis_result.source_id)
    show_summary_modal(build_summary(analysis))

tab_upload, tab_demo = st.tabs(["📤 Upload your set", "🎬 Sample workouts"])

with tab_upload:
    if status == "idle" and st.session_state.analysis_result is None:
        render_empty_state()

    uploaded = st.file_uploader(
        "Drop your squat video here",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        st.session_state.upload_preview_bytes = uploaded.getvalue()
        st.session_state.upload_preview_name = uploaded.name

    has_upload = bool(st.session_state.get("upload_preview_bytes"))

    if has_upload:
        ucol1, ucol2 = st.columns([1, 1])
        with ucol1:
            watch_upload_button(key="upload_tab_watch_raw", label="📹 Watch uploaded video")
        with ucol2:
            st.caption("Preview inline below, or open popup for a larger player.")

        with st.expander("Inline preview", expanded=status == "idle"):
            _show_video_inline(
                st.session_state.upload_preview_bytes,
                filename=st.session_state.get("upload_preview_name", "video.mp4"),
            )

    if status == "error" and st.session_state.analysis_error:
        if render_error_state(st.session_state.analysis_error):
            _reset_analysis_state()
            st.rerun()

    run_upload = st.button(
        "Analyze my squat",
        type="primary",
        disabled=not has_upload and uploaded is None,
        use_container_width=True,
    )

    if run_upload:
        clear_video_modal_request()
        file_bytes = uploaded.getvalue() if uploaded else st.session_state.get("upload_preview_bytes")
        file_name = uploaded.name if uploaded else st.session_state.get("upload_preview_name", "video.mp4")
        if not file_bytes:
            st.warning("Please upload a video first.")
        else:
            progress_slot = st.empty()
            status_slot = st.empty()

            def on_progress(msg: str, frac: float) -> None:
                with progress_slot.container():
                    render_loading_state(msg, frac)

            try:
                st.session_state.analysis_status = "uploading"
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix or ".mp4") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = Path(tmp.name)
                video_path, source_id = stage_uploaded_video(tmp_path)
                tmp_path.unlink(missing_ok=True)
                st.session_state.analysis_status = "analyzing"
                _run_pipeline(video_path, source_id, exercise, coaching_provider, include_coaching, on_progress)
                progress_slot.empty()
                status_slot.success("Analysis complete — check the summary popup!")
                st.rerun()
            except Exception as exc:
                st.session_state.analysis_status = "error"
                st.session_state.analysis_error = str(exc)
                progress_slot.empty()
                st.rerun()

    # After analysis — results + video buttons always on the upload tab
    result = st.session_state.analysis_result
    if result is not None and status == "complete":
        analysis = load_form_analysis(PROJECT_ROOT, result.source_id)
        if analysis:
            from frontend.ui_panels import render_inline_analysis_summary

            render_inline_analysis_summary(build_summary(analysis))
        else:
            st.warning("Analysis finished but results file is missing. Try re-running.")

        st.markdown("#### Your videos")
        acol1, acol2 = st.columns(2)
        with acol1:
            watch_upload_button(key="upload_tab_watch_after", label="📹 Watch uploaded video")
        with acol2:
            watch_annotated_button(key="upload_tab_annotated_after", label="🎬 Watch annotated video")

with tab_demo:
    demos = list_demo_videos()
    if not demos:
        st.warning("No sample videos in `data/raw/videos/`.")
    else:
        choice = st.selectbox("Sample workout", [d["source_id"] for d in demos])
        st.session_state.demo_choice = choice
        selected = next(d for d in demos if d["source_id"] == choice)
        _show_video_inline(Path(selected["path"]))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.button(
                "Analyze sample",
                type="primary",
                use_container_width=True,
                key="demo_analyze_sample",
                on_click=_queue_demo_analysis,
            )
        with c2:
            st.button(
                "Load cached",
                use_container_width=True,
                key="demo_load_cached",
                on_click=_load_cached_demo_callback,
            )
        with c3:
            watch_demo_annotated_button(key="demo_watch_annotated", label="🎬 Annotated")
        with c4:
            if st.button("Reset", use_container_width=True):
                _reset_analysis_state()
                st.rerun()

        if st.session_state.pop("demo_analysis_requested", False):
            prog = st.empty()

            def on_prog(msg: str, frac: float) -> None:
                with prog.container():
                    render_loading_state(msg, frac)

            try:
                clear_video_modal_request()
                _run_pipeline(Path(selected["path"]), choice, exercise, coaching_provider, include_coaching, on_prog)
                st.rerun()
            except Exception as exc:
                st.session_state.analysis_status = "error"
                st.session_state.analysis_error = str(exc)
                st.rerun()

result = st.session_state.analysis_result
if result is not None and st.session_state.show_full_analysis:
    st.divider()
    analysis = load_form_analysis(PROJECT_ROOT, result.source_id)
    view = build_full_view(
        analysis,
        evaluation_video=str(result.evaluation_video) if result.evaluation_video else None,
    )
    render_full_analysis_panel(view, result)
