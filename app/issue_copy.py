"""Beginner-friendly copy for form issues and dimensions."""

from __future__ import annotations

from typing import Any

# Maps backend flag ids → human-readable coaching copy
ISSUE_COPY: dict[str, dict[str, str]] = {
    "shallow_depth": {
        "title": "Not enough depth",
        "explanation": "You are stopping before reaching a strong squat depth.",
        "cue": "Sit your hips lower while keeping your chest lifted.",
    },
    "severe_shallow_depth": {
        "title": "Very shallow squat",
        "explanation": "You are barely bending your knees — this limits strength gains and muscle work.",
        "cue": "Use a box or bench as a depth target and lightly touch it each rep.",
    },
    "knee_valgus": {
        "title": "Knees collapsing inward",
        "explanation": "Your knees move inward instead of tracking over your toes.",
        "cue": "Push your knees slightly outward as you descend and stand.",
    },
    "severe_knee_valgus": {
        "title": "Strong knee cave",
        "explanation": "Your knees cave in noticeably — reduce load until this improves.",
        "cue": "Try a mini-band above the knees and push out against it.",
    },
    "forward_lean": {
        "title": "Too much forward lean",
        "explanation": "Your torso leans forward too much near the bottom.",
        "cue": "Brace your core and keep your chest proud.",
    },
    "severe_forward_lean": {
        "title": "Excessive forward lean",
        "explanation": "Your chest drops significantly — this shifts stress to your lower back.",
        "cue": "Reduce weight and practice wall-facing squats for posture.",
    },
    "heel_lift": {
        "title": "Heels lifting",
        "explanation": "Your heels appear to lift during the squat.",
        "cue": "Keep pressure through your midfoot and heel.",
    },
    "severe_heel_lift": {
        "title": "Heels coming off the floor",
        "explanation": "Lifting heels reduces stability and depth potential.",
        "cue": "Widen stance slightly and stretch calves before squatting.",
    },
    "asymmetry": {
        "title": "Left/right imbalance",
        "explanation": "One side moves differently from the other.",
        "cue": "Slow down and keep both knees and hips moving evenly.",
    },
    "severe_asymmetry": {
        "title": "Strong side-to-side imbalance",
        "explanation": "One leg is doing more work than the other.",
        "cue": "Film from the front and add single-leg balance drills.",
    },
    "unstable_path": {
        "title": "Unstable movement",
        "explanation": "Your squat path is shaky or inconsistent.",
        "cue": "Control the descent and pause briefly at the bottom.",
    },
    "severe_instability": {
        "title": "Very unstable reps",
        "explanation": "The movement looks rushed or uncontrolled.",
        "cue": "Use a 3-second tempo down and a 1-second pause at the bottom.",
    },
    # Legacy mistake ids
    "insufficient_depth": {
        "title": "Not enough depth",
        "explanation": "You are stopping before reaching a strong squat depth.",
        "cue": "Sit your hips lower while keeping your chest lifted.",
    },
    "excessive_forward_lean": {
        "title": "Too much forward lean",
        "explanation": "Your torso leans forward too much near the bottom.",
        "cue": "Brace your core and keep your chest proud.",
    },
}

DIMENSION_COPY: dict[str, dict[str, str]] = {
    "depth_score": {
        "title": "Depth",
        "good": "You hit solid depth on most reps.",
        "needs_work": "You tend to stop above parallel.",
        "cue": "Sit hips back and down until thighs are closer to parallel.",
    },
    "knee_tracking_score": {
        "title": "Knee Tracking",
        "good": "Your knees track well over your feet.",
        "needs_work": "Your knees may cave or drift inward.",
        "cue": "Think 'spread the floor' with your feet.",
    },
    "torso_control_score": {
        "title": "Torso Control",
        "good": "Your chest stays relatively upright.",
        "needs_work": "Your torso leans forward at the bottom.",
        "cue": "Brace your core and look straight ahead.",
    },
    "symmetry_score": {
        "title": "Symmetry",
        "good": "Left and right sides look balanced.",
        "needs_work": "One side drops or shifts more than the other.",
        "cue": "Press evenly through both mid-feet.",
    },
    "stability_score": {
        "title": "Stability",
        "good": "Smooth, controlled movement path.",
        "needs_work": "Reps look rushed or wobbly.",
        "cue": "Slow the descent — 2–3 seconds down.",
    },
    "heel_control_score": {
        "title": "Heel Control",
        "good": "Heels stay grounded through the rep.",
        "needs_work": "Heels may lift off the floor.",
        "cue": "Keep weight through midfoot and heel.",
    },
}

FLAG_ALIASES: dict[str, str] = {
    "severe_shallow_depth": "shallow_depth",
    "severe_knee_valgus": "knee_valgus",
    "severe_forward_lean": "forward_lean",
    "severe_heel_lift": "heel_lift",
    "severe_asymmetry": "asymmetry",
    "severe_instability": "unstable_path",
    "insufficient_depth": "shallow_depth",
    "excessive_forward_lean": "forward_lean",
    "unstable_path": "unstable_path",
}


def performance_label(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Needs Work"
    if score >= 40:
        return "Poor Form"
    return "High Risk"


def score_status_label(score: float) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "OK"
    if score >= 40:
        return "Needs work"
    return "Priority fix"


def issue_copy_for(flag: str) -> dict[str, str]:
    key = FLAG_ALIASES.get(flag, flag)
    return ISSUE_COPY.get(
        key,
        {
            "title": flag.replace("_", " ").title(),
            "explanation": "This area needs attention based on your movement.",
            "cue": "Focus on one technique fix on your next set.",
        },
    )


def dimension_copy_for(dim_key: str, score: float) -> dict[str, str]:
    base = DIMENSION_COPY.get(
        dim_key,
        {
            "title": dim_key.replace("_score", "").replace("_", " ").title(),
            "good": "Looking solid here.",
            "needs_work": "Room to improve.",
            "cue": "Keep practicing with controlled reps.",
        },
    )
    explanation = base["good"] if score >= 75 else base["needs_work"]
    return {
        "title": base["title"],
        "explanation": explanation,
        "cue": base["cue"],
        "status": score_status_label(score),
    }
