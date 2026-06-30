#!/usr/bin/env python3
"""Streamlit demo app — upload a squat video and review form analysis end-to-end."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.inference.video_pipeline import (
    list_demo_videos,
    load_existing_result,
    run_full_pipeline,
    stage_uploaded_video,
)
from src.utils.web_video import prepare_video_for_browser

st.set_page_config(
    page_title="AI Gym Form Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


def _show_video(source: Path | bytes, *, filename: str = "video.mp4") -> None:
    """Display a video in Streamlit using browser-compatible bytes."""
    try:
        data, mime = prepare_video_for_browser(source, filename=filename)
        st.video(data, format=mime)
    except FileNotFoundError:
        st.warning("Video file not found.")
    except Exception as exc:
        st.error(f"Could not play video: {exc}")


def _quality_badge(label: str | None) -> str:
    if not label:
        return "—"
    low = label.lower()
    if low == "good":
        return "✅ good"
    if low in ("bad", "poor"):
        return "❌ bad"
    return "⚠️ needs work"


def _render_results(result) -> None:
    report = result.report
    reps = report.get("repetitions", [])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reps detected", report.get("rep_count", len(reps)))
    with col2:
        avg_score = (
            sum(r.get("form_score") or 0 for r in reps) / len(reps) if reps else 0
        )
        st.metric("Avg form score", f"{avg_score:.0f}/100")
    with col3:
        good = sum(1 for r in reps if (r.get("model_prediction") or "").lower() == "good")
        st.metric("ML: good reps", f"{good}/{len(reps)}" if reps else "—")

    if reps:
        st.subheader("Per-rep breakdown")
        rows = []
        for r in reps:
            mistakes = r.get("mistakes") or []
            rows.append(
                {
                    "Rep": r.get("rep_id"),
                    "Frames": r.get("frames"),
                    "Knee @ bottom": f"{r.get('bottom_knee_angle', 0):.0f}°",
                    "Rules": _quality_badge(r.get("rule_quality")),
                    "Score": r.get("form_score"),
                    "Model": r.get("model_prediction", "n/a"),
                    "Confidence": (
                        f"{r.get('model_confidence', 0) * 100:.0f}%"
                        if r.get("model_confidence") is not None
                        else "—"
                    ),
                    "Flags": ", ".join(mistakes) if mistakes else "—",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    vid_col, chart_col = st.columns(2)
    with vid_col:
        st.subheader("Annotated video")
        if result.evaluation_video and result.evaluation_video.exists():
            _show_video(result.evaluation_video)
        else:
            st.info("Run analysis to generate the annotated evaluation video.")

    with chart_col:
        st.subheader("Knee angle chart")
        if result.chart_path and result.chart_path.exists():
            st.image(str(result.chart_path), use_container_width=True)
        else:
            st.info("Knee angle chart appears after a full analysis run.")

    st.subheader("Coaching report")
    if result.coaching_text_path and result.coaching_text_path.exists():
        if result.coaching_provider:
            st.caption(f"Provider: {result.coaching_provider} (free)")
        st.text(result.coaching_text_path.read_text(encoding="utf-8"))
    else:
        st.info("Enable coaching in the sidebar and re-run to generate feedback.")


st.title("AI Gym Form Coach")
st.caption(
    "Upload a squat video — pose estimation, rep segmentation, rule + ML analysis, "
    "and free coaching feedback. No API key required."
)

with st.sidebar:
    st.header("Settings")
    exercise = st.selectbox("Exercise", ["squat"], index=0)
    coaching_provider = st.selectbox(
        "Coaching",
        options=["template", "auto", "ollama", "openai"],
        index=0,
        help="template = free built-in coach (default). ollama = free local AI if installed.",
    )
    include_coaching = st.checkbox("Generate coaching report", value=True)
    st.divider()
    st.markdown(
        "**Tip:** First run takes longer (pose + ML). "
        "Use **sample_squat** below to test quickly if already processed."
    )

tab_upload, tab_demo = st.tabs(["Upload video", "Sample videos"])

result = st.session_state.analysis_result

with tab_upload:
    uploaded = st.file_uploader(
        "Squat video (mp4, mov, avi)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
    )
    if uploaded is not None:
        upload_bytes = uploaded.getvalue()
        st.session_state.upload_preview_bytes = upload_bytes
        st.session_state.upload_preview_name = uploaded.name
        st.subheader("Preview")
        _show_video(upload_bytes, filename=uploaded.name)
    elif st.session_state.get("upload_preview_bytes"):
        st.subheader("Preview")
        _show_video(
            st.session_state.upload_preview_bytes,
            filename=st.session_state.get("upload_preview_name", "video.mp4"),
        )

    run_upload = st.button("Analyze uploaded video", type="primary", disabled=uploaded is None)

    if run_upload and uploaded is not None:
        progress = st.progress(0.0, text="Starting…")
        status = st.empty()

        def on_progress(msg: str, frac: float) -> None:
            progress.progress(min(frac, 1.0), text=msg)
            status.caption(msg)

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix or ".mp4"
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            video_path, source_id = stage_uploaded_video(tmp_path)
            tmp_path.unlink(missing_ok=True)

            with st.spinner(f"Processing `{source_id}` — this may take a few minutes…"):
                result = run_full_pipeline(
                    video_path,
                    exercise=exercise,
                    source_id=source_id,
                    coaching_provider=coaching_provider,
                    generate_coaching_report=include_coaching,
                    on_progress=on_progress,
                )
            st.session_state.analysis_result = result
            progress.progress(1.0, text="Complete")
            status.success(f"Finished analysis for **{result.source_id}**")
        except Exception as exc:
            progress.empty()
            status.error(f"Analysis failed: {exc}")
            st.exception(exc)

with tab_demo:
    demos = list_demo_videos()
    if not demos:
        st.warning("No videos in `data/raw/videos/`. Upload one or add a sample mp4.")
    else:
        labels = [d["source_id"] for d in demos]
        choice = st.selectbox("Choose a video", labels, index=0)
        selected = next(d for d in demos if d["source_id"] == choice)
        st.subheader("Preview")
        _show_video(Path(selected["path"]))

        col_a, col_b = st.columns(2)
        with col_a:
            run_demo = st.button("Run full analysis", type="primary")
        with col_b:
            load_only = st.button("Load existing results")

        if load_only:
            loaded = load_existing_result(choice)
            if loaded is None:
                st.warning("No processed data yet — click **Run full analysis** first.")
            else:
                st.session_state.analysis_result = loaded
                result = loaded
                st.success(f"Loaded cached results for **{choice}**")

        if run_demo:
            progress = st.progress(0.0, text="Starting…")
            status = st.empty()

            def on_progress_demo(msg: str, frac: float) -> None:
                progress.progress(min(frac, 1.0), text=msg)
                status.caption(msg)

            try:
                with st.spinner(f"Processing `{choice}`…"):
                    result = run_full_pipeline(
                        Path(selected["path"]),
                        exercise=exercise,
                        source_id=choice,
                        coaching_provider=coaching_provider,
                        generate_coaching_report=include_coaching,
                        on_progress=on_progress_demo,
                    )
                st.session_state.analysis_result = result
                progress.progress(1.0, text="Complete")
                status.success(f"Finished analysis for **{result.source_id}**")
            except Exception as exc:
                progress.empty()
                status.error(f"Analysis failed: {exc}")
                st.exception(exc)

result = st.session_state.analysis_result
if result is not None:
    st.divider()
    _render_results(result)
