"""Build resilient view models from form_analysis.json + pipeline results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from frontend.issue_copy import (
    DIMENSION_COPY,
    dimension_copy_for,
    issue_copy_for,
    performance_label,
)

DIMENSION_KEYS = [
    "depth_score",
    "knee_tracking_score",
    "torso_control_score",
    "symmetry_score",
    "stability_score",
    "heel_control_score",
]

FLAG_FOR_DIMENSION: dict[str, str] = {
    "depth_score": "shallow_depth",
    "knee_tracking_score": "knee_valgus",
    "torso_control_score": "forward_lean",
    "symmetry_score": "asymmetry",
    "stability_score": "unstable_path",
    "heel_control_score": "heel_lift",
}


@dataclass
class AnalysisSummary:
    """Data for the post-analysis summary modal."""

    overall_score: float
    performance_label: str
    main_issue_title: str
    main_issue_explanation: str
    quick_fix: str
    positive_title: str
    positive_explanation: str
    coach_narrative: str
    rep_count: int
    source_id: str


@dataclass
class FullAnalysisView:
    """Data for the detailed analysis panel."""

    source_id: str
    overall_score: float
    performance_label: str
    average_scores: dict[str, float]
    best_dimension: str | None
    worst_dimension: str | None
    main_issues: list[str]
    coach_narrative: str
    repetitions: list[dict[str, Any]] = field(default_factory=list)
    evaluation_video: str | None = None


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _avg_scores_from_reps(reps: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in DIMENSION_KEYS:
        vals = [_safe_float(r.get("scores", {}).get(key)) for r in reps if r.get("scores")]
        vals = [v for v in vals if v > 0 or key in (r.get("scores", {}) for r in reps)]
        if vals:
            out[key] = round(sum(vals) / len(vals), 1)
    return out


def _best_worst(avg: dict[str, float]) -> tuple[str | None, str | None]:
    if not avg:
        return None, None
    return max(avg, key=avg.get), min(avg, key=avg.get)


def _main_issue_from_summary(
    summary: dict[str, Any],
    avg_scores: dict[str, float],
) -> tuple[str, str, str]:
    """Return (title, explanation, cue) for the primary issue."""
    flags = summary.get("main_issues") or []
    if flags:
        copy = issue_copy_for(str(flags[0]))
        return copy["title"], copy["explanation"], copy["cue"]

    worst = summary.get("worst_dimension") or (min(avg_scores, key=avg_scores.get) if avg_scores else None)
    if worst:
        flag = FLAG_FOR_DIMENSION.get(str(worst), "shallow_depth")
        copy = issue_copy_for(flag)
        return copy["title"], copy["explanation"], copy["cue"]

    return (
        "Keep refining your form",
        "No major red flags, but small improvements will add up.",
        "Film your next set and compare rep to rep.",
    )


def _positive_from_summary(
    summary: dict[str, Any],
    avg_scores: dict[str, float],
) -> tuple[str, str]:
    best = summary.get("best_dimension")
    if not best and avg_scores:
        best = max(avg_scores, key=avg_scores.get)
    if best and avg_scores.get(best, 0) >= 70:
        copy = dimension_copy_for(str(best), avg_scores[best])
        dim = DIMENSION_COPY.get(str(best), {})
        positive_text = dim.get("good", copy["explanation"]) if avg_scores[best] >= 75 else copy["explanation"]
        return copy["title"], positive_text
    for key in sorted(avg_scores, key=lambda k: -avg_scores[k]):
        if avg_scores[key] >= 80:
            copy = dimension_copy_for(key, avg_scores[key])
            dim = DIMENSION_COPY.get(key, {})
            return copy["title"], dim.get("good", copy["explanation"])
    return "Consistency", "You completed a full set — great start for analysis."


def build_coach_narrative(
    overall: float,
    main_title: str,
    positive_title: str,
    avg_scores: dict[str, float],
) -> str:
    perf = performance_label(overall).lower()
    worst = min(avg_scores, key=avg_scores.get) if avg_scores else None
    worst_name = DIMENSION_COPY.get(worst or "", {}).get("title", "form") if worst else "technique"
    return (
        f"Your squat is {perf} overall ({overall:.0f}/100). "
        f"Your main limitation is **{main_title.lower()}** — focus there first. "
        f"**{positive_title}** is a positive sign. "
        f"On the next set, prioritize {worst_name.lower() if worst else 'one technique fix'}."
    )


def build_summary(analysis: dict[str, Any] | None) -> AnalysisSummary:
    if not analysis:
        return AnalysisSummary(
            overall_score=0,
            performance_label="Needs Work",
            main_issue_title="Analysis incomplete",
            main_issue_explanation="We could not load your form scores.",
            quick_fix="Try running analysis again.",
            positive_title="Effort",
            positive_explanation="Uploading a video is the first step.",
            coach_narrative="Re-run analysis to get coaching feedback.",
            rep_count=0,
            source_id="unknown",
        )

    summary = analysis.get("video_summary") or {}
    reps = analysis.get("repetitions") or analysis.get("reps") or []
    overall = _safe_float(analysis.get("overall_score"))
    if not overall and reps:
        overall = sum(_safe_float(r.get("overall_score", r.get("form_score"))) for r in reps) / len(reps)

    avg = summary.get("average_scores") or _avg_scores_from_reps(reps)
    main_title, main_expl, quick_fix = _main_issue_from_summary(summary, avg)
    pos_title, pos_expl = _positive_from_summary(summary, avg)
    narrative = build_coach_narrative(overall, main_title, pos_title, avg)

    return AnalysisSummary(
        overall_score=round(overall, 1),
        performance_label=performance_label(overall),
        main_issue_title=main_title,
        main_issue_explanation=main_expl,
        quick_fix=quick_fix,
        positive_title=pos_title,
        positive_explanation=pos_expl,
        coach_narrative=narrative,
        rep_count=int(summary.get("num_reps") or len(reps)),
        source_id=str(analysis.get("source_id", "unknown")),
    )


def build_full_view(
    analysis: dict[str, Any] | None,
    *,
    evaluation_video: str | None = None,
) -> FullAnalysisView:
    if not analysis:
        return FullAnalysisView(
            source_id="unknown",
            overall_score=0,
            performance_label="Needs Work",
            average_scores={},
            best_dimension=None,
            worst_dimension=None,
            main_issues=[],
            coach_narrative="No analysis data available.",
        )

    summary = analysis.get("video_summary") or {}
    reps = analysis.get("repetitions") or analysis.get("reps") or []
    overall = _safe_float(analysis.get("overall_score"))
    avg = summary.get("average_scores") or _avg_scores_from_reps(reps)
    best, worst = summary.get("best_dimension"), summary.get("worst_dimension")
    if not best or not worst:
        b, w = _best_worst(avg)
        best = best or b
        worst = worst or w

    mod = build_summary(analysis)
    return FullAnalysisView(
        source_id=str(analysis.get("source_id", "unknown")),
        overall_score=mod.overall_score,
        performance_label=mod.performance_label,
        average_scores=avg,
        best_dimension=str(best) if best else None,
        worst_dimension=str(worst) if worst else None,
        main_issues=[str(f) for f in (summary.get("main_issues") or [])],
        coach_narrative=mod.coach_narrative,
        repetitions=reps,
        evaluation_video=evaluation_video,
    )
