"""Summary modal and full analysis panel."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.analysis_view import AnalysisSummary, FullAnalysisView
from app.video_modal import watch_annotated_button, watch_upload_button
from app.ui_score import render_dimension_grid, render_issue_explanation, render_rep_breakdown_card
from app.ui_states import render_score_badge, render_score_ring


def show_summary_modal(summary: AnalysisSummary) -> None:
    """Post-analysis modal — auto-opens when analysis completes."""
    if not hasattr(st, "dialog"):
        _render_summary_inline(summary)
        return

    @st.dialog("Squat Analysis Complete", width="large")
    def _modal() -> None:
        _render_summary_body(summary)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View full analysis", type="primary", use_container_width=True, key="modal_full"):
                st.session_state.show_full_analysis = True
                st.session_state.pending_summary_modal = False
                st.rerun()
        with col2:
            if st.button("Analyze another video", use_container_width=True, key="modal_reset"):
                _reset_analysis_state()
                st.rerun()

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            watch_upload_button(key="modal_watch_upload", label="📹 Watch uploaded video")
        with vcol2:
            watch_annotated_button(key="modal_watch_annotated", label="🎬 Watch annotated video")

    _modal()


def _render_summary_inline(summary: AnalysisSummary) -> None:
    st.success("Squat analysis complete!")
    _render_summary_body(summary)


def render_inline_analysis_summary(summary: AnalysisSummary) -> None:
    """Always-visible results on the upload tab (not only in the popup)."""
    st.success("Squat analysis complete")
    _render_summary_body(summary)
    if st.button("View full analysis", type="primary", key="inline_view_full", use_container_width=True):
        st.session_state.show_full_analysis = True
        st.session_state.pending_summary_modal = False
        st.rerun()


def _render_summary_body(summary: AnalysisSummary) -> None:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        render_score_ring(summary.overall_score)
    with col_r:
        render_score_badge(summary.performance_label)
        st.markdown(f"### {summary.overall_score:.0f}/100")
        st.markdown(f"**{summary.rep_count}** reps analyzed")

    st.markdown("---")
    st.markdown("#### Main issue")
    st.markdown(f"**{summary.main_issue_title}**")
    st.caption(summary.main_issue_explanation)
    st.info(f"**Quick fix:** {summary.quick_fix}")

    st.markdown("#### What's working")
    st.success(f"**{summary.positive_title}** — {summary.positive_explanation}")


def render_full_analysis_panel(view: FullAnalysisView, result) -> None:
    """Detailed results section — anchor for scroll from modal."""
    st.markdown('<div id="full-analysis"></div>', unsafe_allow_html=True)
    st.markdown("## Full analysis")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        render_score_ring(view.overall_score, size="medium")
        render_score_badge(view.performance_label)
    with col_b:
        st.markdown("#### Coach explanation")
        st.markdown(view.coach_narrative)

    # Video actions
    st.markdown("#### Videos")
    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        watch_upload_button(key="full_watch_upload", label="📹 Watch uploaded video")
    with vcol2:
        watch_annotated_button(key="full_watch_annotated", label="🎬 Watch annotated video")
    with vcol3:
        if st.button("📋 Show summary again", use_container_width=True, key="reopen_summary"):
            st.session_state.pending_summary_modal = True
            st.rerun()

    if view.main_issues:
        st.markdown("#### Main issues")
        for flag in view.main_issues[:4]:
            render_issue_explanation(flag)

    if view.average_scores:
        st.markdown("#### Form dimensions")
        best = (view.best_dimension or "").replace("_score", "").replace("_", " ").title()
        worst = (view.worst_dimension or "").replace("_score", "").replace("_", " ").title()
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(f"🟢 **Strongest:** {best or '—'}")
        with bc2:
            st.markdown(f"🔴 **Focus area:** {worst or '—'}")
        render_dimension_grid(view.average_scores)

    if view.repetitions:
        st.markdown("#### Rep-by-rep breakdown")
        cols = st.columns(min(3, len(view.repetitions)))
        for i, rep in enumerate(view.repetitions):
            with cols[i % len(cols)]:
                render_rep_breakdown_card(rep)

    if st.session_state.get("show_full_analysis"):
        components.html(
            """<script>
            const el = document.getElementById('full-analysis');
            if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
            </script>""",
            height=0,
        )


def _reset_analysis_state() -> None:
    from app.video_modal import clear_video_modal_request

    for key in (
        "analysis_result",
        "analysis_status",
        "pending_summary_modal",
        "show_full_analysis",
        "analysis_error",
        "upload_preview_bytes",
        "upload_preview_name",
        "evaluation_video_path",
        "raw_video_path",
    ):
        st.session_state.pop(key, None)
    clear_video_modal_request()
    st.session_state.analysis_status = "idle"
