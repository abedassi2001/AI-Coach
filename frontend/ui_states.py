"""Analysis UI states: idle, loading, error."""

from __future__ import annotations

import streamlit as st

from frontend.gym_theme import score_color


def render_empty_state() -> None:
    st.markdown(
        """
        <div style="background:#1e1e28;border:1px dashed #444455;border-radius:16px;
        padding:2.5rem;text-align:center;margin:1rem 0;">
        <p style="font-size:2.5rem;margin:0;">🏋️</p>
        <p style="color:#fff;font-size:1.2rem;font-weight:700;margin:0.75rem 0 0.5rem 0;">
        Upload your squat set to get started
        </p>
        <p style="color:#9999aa;font-size:0.95rem;max-width:480px;margin:0 auto;line-height:1.5;">
        We track your body pose, count reps, score your form from 0–100,
        and tell you exactly what to fix — no gym partner needed.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_state(message: str, progress: float | None = None) -> None:
    st.markdown(
        f"""
        <div style="background:#1e1e28;border:1px solid #ff6b4a44;border-radius:16px;
        padding:2rem;text-align:center;margin:1rem 0;">
        <p style="color:#ff6b4a;font-weight:700;font-size:1.1rem;margin:0 0 0.5rem 0;">
        Analyzing your squat form…
        </p>
        <p style="color:#aaa;margin:0;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if progress is not None:
        st.progress(min(max(progress, 0.0), 1.0))


def render_error_state(message: str, on_retry_key: str = "retry_analysis") -> bool:
    st.markdown(
        f"""
        <div style="background:#2a1a1a;border:1px solid #ff4d4d55;border-radius:16px;
        padding:1.5rem;margin:1rem 0;">
        <p style="color:#ff6b4a;font-weight:700;margin:0 0 0.5rem 0;">Something went wrong</p>
        <p style="color:#ccc;margin:0;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.button("Try again", type="primary", key=on_retry_key)


def render_score_ring(score: float, size: str = "large") -> None:
    color = score_color(score)
    font = "3.5rem" if size == "large" else "2rem"
    st.markdown(
        f"""
        <div style="text-align:center;">
        <div style="display:inline-flex;align-items:center;justify-content:center;
        width:120px;height:120px;border-radius:50%;
        border:4px solid {color};background:#1a1a24;">
        <span style="font-size:{font};font-weight:800;color:{color};">{score:.0f}</span>
        </div>
        <p style="color:#888;margin:0.5rem 0 0 0;font-size:0.8rem;">OUT OF 100</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_badge(label: str, color: str = "#ff6b4a") -> None:
    st.markdown(
        f'<span style="background:{color}22;color:{color};padding:0.35rem 0.75rem;'
        f'border-radius:20px;font-size:0.85rem;font-weight:600;border:1px solid {color}55;">'
        f"{label}</span>",
        unsafe_allow_html=True,
    )
