"""Gym-themed CSS for the Streamlit demo app."""

from __future__ import annotations

import streamlit as st

GYM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(165deg, #0a0a0f 0%, #12121a 40%, #1a1520 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141418 0%, #0d0d12 100%);
    border-right: 1px solid #2a2a35;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Hero */
.gym-hero {
    background: linear-gradient(135deg, #1f1f28 0%, #2d1f1a 50%, #1a1a24 100%);
    border: 1px solid #3d3d4a;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.gym-hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 0.06em;
    color: #fff;
    margin: 0;
    line-height: 1;
}
.gym-hero .tagline {
    color: #a0a0b0;
    font-size: 1rem;
    margin-top: 0.5rem;
}
.gym-hero .accent-bar {
    height: 4px;
    width: 80px;
    background: linear-gradient(90deg, #ff4d4d, #ff8c42);
    border-radius: 2px;
    margin: 1rem 0;
}

/* Stat cards */
.stat-card {
    background: #1e1e28;
    border: 1px solid #333342;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-card .value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #ff6b4a;
    line-height: 1.1;
}
.stat-card .label {
    color: #888899;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* Dimension score bar */
.dim-row {
    margin-bottom: 0.65rem;
}
.dim-label {
    color: #c0c0cc;
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
    display: flex;
    justify-content: space-between;
}
.dim-track {
    background: #2a2a38;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.dim-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease;
}

/* CTA button area */
.cta-wrap {
    background: linear-gradient(90deg, rgba(255,77,77,0.12), rgba(255,140,66,0.08));
    border: 1px solid #ff6b4a44;
    border-radius: 14px;
    padding: 1.25rem;
    margin: 1rem 0 1.5rem 0;
    text-align: center;
}
.cta-wrap p {
    color: #b0b0c0;
    margin: 0 0 0.75rem 0;
    font-size: 0.95rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: #1e1e28;
    border-radius: 8px 8px 0 0;
    border: 1px solid #333342;
    color: #aaa;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #2a2220 !important;
    color: #ff6b4a !important;
    border-color: #ff6b4a55 !important;
}

/* Metrics override */
div[data-testid="stMetric"] {
    background: #1e1e28;
    border: 1px solid #333342;
    border-radius: 12px;
    padding: 0.75rem 1rem;
}

/* Hide Streamlit footer branding slightly cleaner */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Dialog (modal) styling when supported */
div[data-testid="stDialog"] > div {
    background: #141418 !important;
    border: 1px solid #ff6b4a44;
}
</style>
"""


def inject_gym_theme() -> None:
    st.markdown(GYM_CSS, unsafe_allow_html=True)


def score_color(value: float) -> str:
    if value >= 80:
        return "#3dd68c"
    if value >= 60:
        return "#ffb020"
    return "#ff4d4d"


def render_hero() -> None:
    st.markdown(
        """
        <div class="gym-hero">
            <h1>IRON FORM COACH</h1>
            <div class="accent-bar"></div>
            <p class="tagline">
                AI-powered squat analysis — pose tracking, rep counting,
                biomechanical scoring &amp; coaching. No gym buddy required.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(reps: int, overall: float, quality: str) -> None:
    q_display = quality.replace("_", " ").title()
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
            <div class="stat-card">
                <div class="value">{reps}</div>
                <div class="label">Reps detected</div>
            </div>
            <div class="stat-card">
                <div class="value">{overall:.0f}</div>
                <div class="label">Overall score</div>
            </div>
            <div class="stat-card">
                <div class="value" style="font-size:1.4rem; padding-top:0.4rem;">{q_display}</div>
                <div class="label">Form quality</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dimension_bars(scores: dict[str, float]) -> None:
    labels = {
        "depth_score": "Depth",
        "knee_tracking_score": "Knee tracking",
        "torso_control_score": "Torso control",
        "symmetry_score": "Symmetry",
        "stability_score": "Stability",
        "heel_control_score": "Heel control",
    }
    parts = ['<div style="background:#1e1e28;border:1px solid #333342;border-radius:12px;padding:1rem 1.25rem;">']
    parts.append('<p style="color:#fff;font-weight:700;margin:0 0 0.75rem 0;">Form breakdown</p>')
    for key, label in labels.items():
        val = scores.get(key, 0)
        color = score_color(val)
        parts.append(f'''
        <div class="dim-row">
            <div class="dim-label"><span>{label}</span><span style="color:{color};font-weight:600;">{val:.0f}</span></div>
            <div class="dim-track"><div class="dim-fill" style="width:{min(val,100)}%;background:{color};"></div></div>
        </div>''')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
