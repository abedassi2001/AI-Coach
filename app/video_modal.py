"""Video player panel — top-level (never inside another dialog)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.utils.web_video import ensure_browser_playable, resolve_playable_path


def _resolve_annotated_path_from_session() -> Path | None:
    ev = st.session_state.get("evaluation_video_path")
    if ev and Path(ev).exists():
        return ensure_browser_playable(Path(ev))
    result = st.session_state.get("analysis_result")
    if result and getattr(result, "evaluation_video", None) and result.evaluation_video.exists():
        return ensure_browser_playable(Path(result.evaluation_video))
    return None


def request_upload_video_modal() -> None:
    st.session_state.video_modal_request = {"type": "upload"}
    st.session_state.pending_summary_modal = False


def request_annotated_video_modal(video_path: Path | str, source_id: str) -> None:
    path = ensure_browser_playable(Path(video_path))
    st.session_state.video_modal_request = {
        "type": "annotated",
        "path": str(path.resolve()),
        "source_id": source_id,
    }
    st.session_state.pending_summary_modal = False


def clear_video_modal_request() -> None:
    st.session_state.pop("video_modal_request", None)


def queue_upload_video_callback() -> None:
    request_upload_video_modal()


def queue_annotated_video_callback() -> None:
    path = _resolve_annotated_path_from_session()
    result = st.session_state.get("analysis_result")
    source_id = result.source_id if result else "squat"
    if path is not None and path.exists():
        request_annotated_video_modal(path, source_id)
    else:
        st.session_state.flash_video_error = "Annotated video not found. Re-run analysis."


def queue_demo_annotated_callback() -> None:
    from src.inference.video_pipeline import load_existing_result

    choice = st.session_state.get("demo_choice")
    if not choice:
        st.session_state.flash_video_error = "Select a sample workout first."
        return
    loaded = load_existing_result(choice)
    if loaded and loaded.evaluation_video and loaded.evaluation_video.exists():
        request_annotated_video_modal(loaded.evaluation_video, choice)
    else:
        st.session_state.flash_video_error = "Run analysis on this sample first."


def _resolve_upload_video_path() -> Path | None:
    upload_bytes = st.session_state.get("upload_preview_bytes")
    upload_name = st.session_state.get("upload_preview_name", "upload.mp4")
    if upload_bytes:
        try:
            return resolve_playable_path(upload_bytes, filename=upload_name)
        except Exception:
            return None

    raw = st.session_state.get("raw_video_path")
    if raw and Path(raw).exists():
        return ensure_browser_playable(Path(raw))

    result = st.session_state.get("analysis_result")
    if result is not None and getattr(result, "video_path", None):
        path = Path(result.video_path)
        if path.exists():
            return ensure_browser_playable(path)
    return None


def _resolve_annotated_video_path(req: dict) -> Path | None:
    path = Path(req.get("path", ""))
    if not path.exists():
        path = _resolve_annotated_path_from_session() or path
    if path.exists():
        return ensure_browser_playable(path)
    return None


def _render_video_body(req: dict) -> bool:
    """Render video using a file path (reliable in Streamlit)."""
    try:
        if req["type"] == "upload":
            path = _resolve_upload_video_path()
            if path is None:
                st.error("Upload video not available. Upload a file or run analysis first.")
                return False
            st.markdown("### Your uploaded video")
            st.caption("Original recording before overlays")
        else:
            path = _resolve_annotated_video_path(req)
            if path is None:
                st.error("Annotated video not found. Run analysis first.")
                return False
            st.markdown("### Annotated form analysis")
            st.caption("Skeleton overlay · rep phases · rule scores · ML predictions")

        st.video(str(path.resolve()))
        st.download_button(
            label="Download video",
            data=path.read_bytes(),
            file_name=path.name,
            mime="video/mp4",
            use_container_width=True,
            key=f"dl_{req['type']}_{req.get('source_id', 'upload')}",
        )
        return True
    except Exception as exc:
        st.error(f"Could not load video: {exc}")
        return False


def render_video_modal_if_requested() -> None:
    """
    Show video in a top-level panel (not st.dialog — dialogs break HTML5 playback).
    """
    req = st.session_state.get("video_modal_request")
    if not req:
        return

    title = (
        "Your uploaded video"
        if req["type"] == "upload"
        else f"Annotated analysis — {req.get('source_id', 'squat')}"
    )

    st.markdown("---")
    head_l, head_r = st.columns([5, 1])
    with head_l:
        st.markdown(f"#### 🎬 {title}")
    with head_r:
        if st.button("Close", key="video_panel_close", use_container_width=True):
            clear_video_modal_request()
            st.rerun()

    with st.spinner("Loading video…"):
        _render_video_body(req)
    st.markdown("---")


def render_video_flash_message() -> None:
    msg = st.session_state.pop("flash_video_error", None)
    if msg:
        st.warning(msg)


def watch_upload_button(*, key: str = "watch_upload_video", label: str = "📹 Watch uploaded video") -> None:
    st.button(
        label,
        use_container_width=True,
        key=key,
        on_click=queue_upload_video_callback,
    )


def watch_annotated_button(
    *,
    key: str = "watch_annotated_video",
    label: str = "🎬 Watch annotated video",
) -> None:
    st.button(
        label,
        type="primary",
        use_container_width=True,
        key=key,
        on_click=queue_annotated_video_callback,
    )


def watch_demo_annotated_button(
    *,
    key: str = "demo_watch_annotated",
    label: str = "🎬 Annotated",
) -> None:
    st.button(
        label,
        use_container_width=True,
        key=key,
        on_click=queue_demo_annotated_callback,
    )
