"""Derive flags and deterministic coaching text from continuous rep scores."""

from __future__ import annotations

from typing import Any

DIMENSION_LABELS: dict[str, str] = {
    "depth_score": "depth",
    "knee_tracking_score": "knee tracking",
    "torso_control_score": "torso control",
    "symmetry_score": "symmetry",
    "stability_score": "stability",
    "heel_control_score": "heel control",
}

POSITIVE_LINES: dict[str, str] = {
    "depth_score": "Your depth looks solid on this rep.",
    "knee_tracking_score": "Your knees track well overall.",
    "torso_control_score": "Your torso stays well controlled through the rep.",
    "symmetry_score": "Left and right sides look balanced.",
    "stability_score": "The movement path looks smooth and controlled.",
    "heel_control_score": "Your heels stay grounded at the bottom.",
}

ISSUE_LINES: dict[str, str] = {
    "depth_score": "You are stopping slightly high — sit the hips a bit lower.",
    "knee_tracking_score": "Your knees cave inward slightly — push them out over the toes.",
    "torso_control_score": "Your torso leans forward near the bottom — keep the chest taller.",
    "symmetry_score": "One side drops more than the other — press evenly through both feet.",
    "stability_score": "The rep looks rushed or wobbly — slow the descent and pause briefly at the bottom.",
    "heel_control_score": "Your heels may be lifting — keep pressure through the midfoot.",
}

SEVERE_ISSUE_LINES: dict[str, str] = {
    "depth_score": "Depth is well above parallel — treat this as a priority fix.",
    "knee_tracking_score": "Knee valgus is pronounced — reduce load and focus on pushing knees out.",
    "torso_control_score": "Forward lean is excessive — brace harder and reduce load if needed.",
    "symmetry_score": "Strong left/right imbalance — film from the front and address the weaker side.",
    "stability_score": "Movement is very unstable — use a slower tempo and lighter load.",
    "heel_control_score": "Heels appear to lift significantly — check ankle mobility and stance width.",
}

CORRECTION_CUES: dict[str, str] = {
    "depth_score": "Sit hips back and down until thighs are closer to parallel while keeping heels flat.",
    "knee_tracking_score": "Think 'spread the floor' and align knees with toes on the way down.",
    "torso_control_score": "Brace your core and keep eyes forward so the chest stays tall.",
    "symmetry_score": "Press evenly through both mid-feet and avoid shifting to one side.",
    "stability_score": "Use a 3-second descent, a 1-second pause at the bottom, then drive up smoothly.",
    "heel_control_score": "Widen stance slightly and keep weight through midfoot/heel.",
}


def flags_from_scores(
    scores: dict[str, float],
    flag_thresholds: dict[str, list[list[Any]]],
) -> list[str]:
    """Derive ordered flags from dimension scores (lower score = more severe flags)."""
    flags: list[str] = []
    for dim_key, rules in flag_thresholds.items():
        score = scores.get(dim_key, 100.0)
        for entry in rules:
            flag_id, threshold = entry[0], float(entry[1])
            if score < threshold:
                flags.append(str(flag_id))
    return flags


def mistakes_from_flags(
    flags: list[str],
    scores: dict[str, float],
    metrics: dict[str, float],
    legacy_map: dict[str, str],
    frame_index: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build legacy mistake objects for backward compatibility.

    Deduplicates by mistake_id (keeps most severe flag per mistake type).
    """
    from backend.feedback.templates import format_mistake

    severity_order = {"severe": 0, "normal": 1}
    seen: dict[str, dict[str, Any]] = {}

    for flag in flags:
        mistake_id = legacy_map.get(flag, flag)
        is_severe = flag.startswith("severe_")
        severity = "high" if is_severe else ("medium" if "severe" in flag else "medium")
        if "unstable" in flag or flag == "unstable_path":
            severity = "low" if not is_severe else "medium"

        value, threshold = _value_threshold_for_flag(flag, metrics)
        message = format_mistake(mistake_id, value, threshold)

        candidate = {
            "mistake_id": mistake_id,
            "severity": severity,
            "message": message,
            "value": value,
            "threshold": threshold,
            "frame_index": frame_index,
            "flag": flag,
        }
        prev = seen.get(mistake_id)
        if prev is None or severity_order.get(
            "severe" if "severe" in str(prev.get("flag", "")) else "normal", 1
        ) > severity_order.get("severe" if is_severe else "normal", 1):
            seen[mistake_id] = candidate

    return list(seen.values())


def _value_threshold_for_flag(flag: str, metrics: dict[str, float]) -> tuple[float, float]:
    if "depth" in flag or flag == "shallow_depth" or flag == "severe_shallow_depth":
        return metrics.get("bottom_knee_angle", 0.0), 90.0
    if "lean" in flag or "torso" in flag:
        return metrics.get("bottom_torso_lean", 0.0), 45.0
    if "asymmetry" in flag or "symmetry" in flag:
        return metrics.get("bottom_knee_asymmetry", 0.0), 15.0
    if "unstable" in flag or "stability" in flag:
        return metrics.get("knee_angle_std", 0.0), 25.0
    if "valgus" in flag or "knee" in flag:
        return metrics.get("knee_valgus_score", 0.0), 0.15
    if "heel" in flag:
        return metrics.get("heel_lift_signal", 0.0), 0.02
    return 0.0, 0.0


def generate_rep_coaching(
    scores: dict[str, float],
    flags: list[str],
    confidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministic coaching summary for one rep.

    Returns overall sentence, top issues, one positive, one cue, and bullet feedback.
    """
    overall = float(scores.get("overall_score", 0))
    dim_scores = {k: v for k, v in scores.items() if k.endswith("_score") and k != "overall_score"}

    ranked = sorted(dim_scores.items(), key=lambda kv: kv[1])
    worst_dims = [k for k, _ in ranked[:3]]
    best_dim = ranked[-1][0] if ranked else None

    issues: list[str] = []
    for dim in worst_dims:
        val = dim_scores[dim]
        if val < 40:
            issues.append(SEVERE_ISSUE_LINES.get(dim, f"Improve {DIMENSION_LABELS.get(dim, dim)}."))
        elif val < 70:
            issues.append(ISSUE_LINES.get(dim, f"Work on {DIMENSION_LABELS.get(dim, dim)}."))

    positives: list[str] = []
    if best_dim and dim_scores.get(best_dim, 0) >= 75:
        positives.append(POSITIVE_LINES.get(best_dim, "At least one dimension looks strong."))

    if not positives:
        for dim, val in sorted(dim_scores.items(), key=lambda kv: -kv[1]):
            if val >= 80:
                positives.append(POSITIVE_LINES.get(dim, ""))
                break

    top_issue_dim = worst_dims[0] if worst_dims and dim_scores.get(worst_dims[0], 100) < 70 else None
    correction = CORRECTION_CUES.get(top_issue_dim, "Focus on one technique fix on the next rep.")

    if overall >= 80:
        overall_sentence = "Overall, this rep looks strong with only minor details to refine."
    elif overall >= 70:
        overall_sentence = "Overall, this rep is acceptable but has room to improve."
    elif overall >= 50:
        overall_sentence = "Overall, this rep needs work — address the main issues before adding load."
    else:
        overall_sentence = "Overall, this rep has significant form issues — reduce load and fix technique first."

    feedback_bullets: list[str] = [overall_sentence]
    feedback_bullets.extend(issues[:3])
    if positives:
        feedback_bullets.append(positives[0])
    feedback_bullets.append(correction)

    if confidence and confidence.get("heel_detection_confidence") == "low":
        feedback_bullets.append(
            "Note: heel position confidence is low from this camera angle — verify heel contact on video."
        )

    return {
        "overall_summary": overall_sentence,
        "top_issues": issues[:3],
        "positive_point": positives[0] if positives else None,
        "correction_cue": correction,
        "feedback": feedback_bullets,
        "worst_dimension": worst_dims[0] if worst_dims else None,
        "best_dimension": best_dim,
    }


def build_video_summary(
    rep_outputs: list[dict[str, Any]],
    analyzer_version: str,
) -> dict[str, Any]:
    """Aggregate per-rep scores into a video-level summary."""
    if not rep_outputs:
        return {
            "analyzer_version": analyzer_version,
            "num_reps": 0,
            "average_overall_score": 0.0,
            "average_scores": {},
            "best_dimension": None,
            "worst_dimension": None,
            "main_issues": [],
        }

    dim_keys = [
        "depth_score", "knee_tracking_score", "torso_control_score",
        "symmetry_score", "stability_score", "heel_control_score",
    ]
    avg_scores: dict[str, float] = {}
    for key in dim_keys:
        vals = [r["scores"][key] for r in rep_outputs if key in r.get("scores", {})]
        if vals:
            avg_scores[key] = round(sum(vals) / len(vals), 1)

    overall_vals = [r["overall_score"] for r in rep_outputs]
    avg_overall = round(sum(overall_vals) / len(overall_vals), 1)

    if avg_scores:
        worst_dim = min(avg_scores, key=avg_scores.get)
        best_dim = max(avg_scores, key=avg_scores.get)
    else:
        worst_dim = None
        best_dim = None

    all_flags: list[str] = []
    for r in rep_outputs:
        all_flags.extend(r.get("flags", []))
    # Count flag frequency, exclude severe duplicates for main_issues
    base_flags = [f for f in all_flags if not f.startswith("severe_")]
    freq: dict[str, int] = {}
    for f in base_flags:
        freq[f] = freq.get(f, 0) + 1
    main_issues = sorted(freq.keys(), key=lambda k: -freq[k])[:5]

    return {
        "analyzer_version": analyzer_version,
        "num_reps": len(rep_outputs),
        "average_overall_score": avg_overall,
        "average_scores": avg_scores,
        "best_dimension": best_dim,
        "worst_dimension": worst_dim,
        "main_issues": main_issues,
    }
