"""Score widgets: dimension cards, rep breakdown, issue explanations."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.analysis_view import DIMENSION_KEYS
from app.gym_theme import score_color
from app.issue_copy import dimension_copy_for, issue_copy_for


def render_issue_explanation(flag: str) -> None:
    copy = issue_copy_for(flag)
    st.markdown(
        f"""
        <div style="background:#1e1e28;border-left:3px solid #ff6b4a;padding:0.75rem 1rem;
        border-radius:0 8px 8px 0;margin-bottom:0.5rem;">
        <p style="color:#fff;font-weight:600;margin:0 0 0.25rem 0;">{copy['title']}</p>
        <p style="color:#aaa;font-size:0.9rem;margin:0 0 0.35rem 0;">{copy['explanation']}</p>
        <p style="color:#ff8c42;font-size:0.85rem;margin:0;">→ {copy['cue']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dimension_score_card(dim_key: str, score: float) -> None:
    copy = dimension_copy_for(dim_key, score)
    color = score_color(score)
    st.markdown(
        f"""
        <div style="background:#1e1e28;border:1px solid #333342;border-radius:12px;
        padding:1rem;margin-bottom:0.75rem;height:100%;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="color:#fff;font-weight:700;">{copy['title']}</span>
        <span style="color:{color};font-weight:800;font-size:1.25rem;">{score:.0f}</span>
        </div>
        <p style="color:{color};font-size:0.75rem;margin:0.25rem 0;text-transform:uppercase;
        letter-spacing:0.05em;">{copy['status']}</p>
        <div style="background:#2a2a38;border-radius:4px;height:6px;margin:0.5rem 0;">
        <div style="width:{min(score,100)}%;height:100%;background:{color};border-radius:4px;"></div>
        </div>
        <p style="color:#aaa;font-size:0.85rem;margin:0 0 0.35rem 0;">{copy['explanation']}</p>
        <p style="color:#888;font-size:0.8rem;margin:0;">💡 {copy['cue']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rep_breakdown_card(rep: dict[str, Any]) -> None:
    rid = rep.get("rep_id", "?")
    overall = rep.get("overall_score", rep.get("form_score", 0))
    color = score_color(float(overall or 0))
    coaching = rep.get("coaching") or {}
    top_issues = coaching.get("top_issues") or rep.get("feedback", [])[:2]
    positive = coaching.get("positive_point") or ""
    cue = coaching.get("correction_cue") or ""

    issues_html = ""
    for issue in top_issues[:2]:
        if isinstance(issue, str):
            issues_html += f'<li style="color:#ccc;margin-bottom:0.25rem;">{issue}</li>'

    st.markdown(
        f"""
        <div style="background:#1e1e28;border:1px solid #333342;border-radius:12px;padding:1rem;margin-bottom:0.75rem;">
        <div style="display:flex;justify-content:space-between;">
        <span style="color:#fff;font-weight:700;">Rep {rid}</span>
        <span style="color:{color};font-weight:800;">{overall:.0f}/100</span>
        </div>
        {f'<p style="color:#3dd68c;font-size:0.85rem;margin:0.5rem 0;">✓ {positive}</p>' if positive else ''}
        <ul style="margin:0.5rem 0;padding-left:1.2rem;font-size:0.85rem;">{issues_html}</ul>
        {f'<p style="color:#ff8c42;font-size:0.85rem;margin:0.5rem 0 0 0;">→ {cue}</p>' if cue else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dimension_grid(avg_scores: dict[str, float]) -> None:
    cols = st.columns(3)
    for i, key in enumerate(DIMENSION_KEYS):
        score = float(avg_scores.get(key, 50))
        with cols[i % 3]:
            render_dimension_score_card(key, score)
