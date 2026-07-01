"""Human-readable coaching messages for detected form mistakes."""

from __future__ import annotations

MESSAGES: dict[str, str] = {
    "insufficient_depth": (
        "Squat depth is limited. Your knee angle at the bottom reached {value:.0f}° "
        "(target <= {threshold:.0f}°). Try sitting back and lowering until thighs are "
        "closer to parallel."
    ),
    "excessive_forward_lean": (
        "Your torso leans too far forward ({value:.0f}° vs max {threshold:.0f}°). "
        "Keep your chest up and brace your core throughout the descent."
    ),
    "asymmetry": (
        "Left and right sides move unevenly (knee angle difference {value:.0f}°). "
        "Focus on balanced weight through both feet."
    ),
    "unstable_path": (
        "Movement looks unstable during this rep (knee angle variability {value:.1f}°). "
        "Control the descent and avoid bouncing at the bottom."
    ),
    "heel_lift": (
        "Heels may be lifting off the floor at the bottom. Keep heels planted and "
        "mobilize ankles if needed."
    ),
    "knee_valgus": (
        "Knees appear to cave inward (valgus score {value:.2f} > {threshold:.2f}). "
        "Push knees out in line with toes — a front/side angle helps confirm this."
    ),
}


def format_mistake(mistake_id: str, value: float, threshold: float) -> str:
    template = MESSAGES.get(
        mistake_id,
        "Form issue detected: {mistake_id} (value={value}, threshold={threshold})",
    )
    return template.format(mistake_id=mistake_id, value=value, threshold=threshold)
