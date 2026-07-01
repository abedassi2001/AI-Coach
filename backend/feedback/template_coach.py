"""Free built-in coach — no API key, no internet required."""

from __future__ import annotations

from typing import Any

IMPROVEMENT_CUES: dict[str, str] = {
    "insufficient_depth": (
        "Go deeper: sit hips back first, keep heels flat, and aim for thighs near parallel "
        "(knee angle at bottom ideally <= 90°)."
    ),
    "excessive_forward_lean": (
        "Stay more upright: brace your core hard, eyes forward, and let hips travel back "
        "before the knees bend much."
    ),
    "asymmetry": (
        "Balance left and right: press evenly through both mid-feet; film from the front "
        "once to check if one side drops."
    ),
    "unstable_path": (
        "Slow the rep: 2–3 seconds down, a one-second pause at the bottom, then drive up "
        "without bouncing."
    ),
    "heel_lift": (
        "Glue your heels: widen stance slightly, stretch calves/ankles, or use a thin heel "
        "wedge until mobility improves."
    ),
    "knee_valgus": (
        "Push knees out in line with toes: think 'spread the floor' with your feet and "
        "avoid letting knees cave inward."
    ),
}

PRACTICE_DRILLS: dict[str, str] = {
    "insufficient_depth": "Drill: Goblet squat to a box/bench — lightly touch and stand, 3×8.",
    "excessive_forward_lean": "Drill: Wall-facing squat — toes 6 inches from wall, squat without knees touching.",
    "asymmetry": "Drill: Single-leg sit-to-stand on each leg, 3×5, to find the weaker side.",
    "unstable_path": "Drill: Tempo squat 3-1-2 (3 sec down, 1 sec pause, 2 sec up), 3×5.",
    "heel_lift": "Drill: Heel-elevated goblet squat + daily calf stretch, 2×30 sec each leg.",
    "knee_valgus": "Drill: Mini-band around knees — push out on the way down, 3×10.",
}

MISTAKE_LABELS: dict[str, str] = {
    "insufficient_depth": "Not deep enough",
    "excessive_forward_lean": "Too much forward lean",
    "asymmetry": "Left/right imbalance",
    "unstable_path": "Unstable movement",
    "heel_lift": "Heels lifting",
    "knee_valgus": "Knees caving in",
}


def _severity_rank(severity: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(severity, 3)


def _rep_metric_notes(metrics: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    depth = metrics.get("bottom_knee_angle")
    if depth == depth and depth is not None:
        if depth > 100:
            notes.append(f"Depth was shallow at the bottom (knee ~{depth:.0f}° — target <= 90°).")
        elif depth > 90:
            notes.append(f"Depth was close but not quite there (knee ~{depth:.0f}°).")
        else:
            notes.append(f"Good depth on this rep (knee ~{depth:.0f}° at bottom).")
    lean = metrics.get("bottom_torso_lean")
    if lean == lean and lean is not None and lean > 40:
        notes.append(f"Torso leaned forward a lot ({lean:.0f}°) — chest up on the next rep.")
    return notes


def _opening_line(score: float, quality: str, exercise: str, good_reps: int, bad_reps: int) -> str:
    if score >= 80 and bad_reps == 0:
        return (
            f"Strong {exercise} set ({score:.0f}/100). Keep what you're doing and tighten the "
            "small details below."
        )
    if good_reps > 0 and bad_reps > 0:
        return (
            f"Mixed {exercise} set ({score:.0f}/100). You had {good_reps} solid rep(s) and "
            f"{bad_reps} that need work — copy the feel of the good ones onto the next set."
        )
    return (
        f"Your {exercise} set scored {score:.0f}/100 ({quality}). "
        "The fixes below are ordered by impact — tackle #1 first on your next set."
    )


def generate_template_coaching(context_dict: dict[str, Any], max_actions: int = 5) -> dict[str, Any]:
    """Rich coaching from structured analysis — 100% free, no network."""
    reps = context_dict.get("repetitions", [])
    exercise = context_dict.get("exercise", "squat")
    score = float(context_dict.get("overall_score", 0))
    quality = str(context_dict.get("overall_quality", "unknown"))

    good_reps = sum(1 for r in reps if r.get("model_prediction") == "good")
    bad_reps = sum(1 for r in reps if r.get("model_prediction") == "bad")

    # Collect mistakes across set, ranked by severity
    all_mistakes: list[dict[str, Any]] = []
    for rep in reps:
        for m in rep.get("mistakes", []):
            all_mistakes.append({**m, "rep_id": rep.get("rep_id")})
    all_mistakes.sort(key=lambda m: _severity_rank(str(m.get("severity", "low"))))

    seen_ids: set[str] = set()
    action_plan: list[str] = []
    drills: list[str] = []
    for m in all_mistakes:
        mid = str(m.get("id", ""))
        if mid in seen_ids or mid not in IMPROVEMENT_CUES:
            continue
        seen_ids.add(mid)
        label = MISTAKE_LABELS.get(mid, mid.replace("_", " "))
        action_plan.append(f"[{label}] {IMPROVEMENT_CUES[mid]}")
        if mid in PRACTICE_DRILLS and len(drills) < 3:
            drills.append(PRACTICE_DRILLS[mid])
        if len(action_plan) >= max_actions:
            break

    if not action_plan:
        action_plan.append(
            "Movement looks solid overall. Add one more rep next set while keeping the same "
            "tempo and braced core."
        )

    overall_summary = _opening_line(score, quality, exercise, good_reps, bad_reps)

    rep_feedback = []
    for rep in reps:
        rid = rep.get("rep_id")
        mistakes = sorted(
            rep.get("mistakes", []),
            key=lambda x: _severity_rank(str(x.get("severity", "low"))),
        )
        metrics = rep.get("metrics", {})
        metric_notes = _rep_metric_notes(metrics)

        cues = [IMPROVEMENT_CUES.get(str(m.get("id")), "") for m in mistakes if m.get("id")]
        cues = [c for c in cues if c][:3]

        pred = rep.get("model_prediction")
        rule_score = rep.get("rule_score")
        if pred == "good":
            tone = "Classifier: good form on this rep."
        elif pred == "bad":
            tone = "Classifier: needs work on this rep."
        else:
            tone = f"Rule score: {rule_score}/100."

        summary_parts = [f"Rep {rid}: {tone}"]
        summary_parts.extend(metric_notes[:2])
        if mistakes:
            names = ", ".join(MISTAKE_LABELS.get(str(m.get("id")), str(m.get("id"))) for m in mistakes[:3])
            summary_parts.append(f"Flags: {names}.")

        rep_feedback.append(
            {
                "rep_id": rid,
                "summary": " ".join(summary_parts),
                "focus_areas": [str(m.get("id")) for m in mistakes],
                "cues": cues,
            }
        )

    return {
        "source_id": context_dict.get("source_id"),
        "exercise": exercise,
        "provider": "template",
        "overall_summary": overall_summary,
        "action_plan": action_plan,
        "practice_drills": drills,
        "rep_feedback": rep_feedback,
    }
