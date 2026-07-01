"""Reusable 0–100 scoring utilities for biomechanical form analysis."""

from __future__ import annotations

from typing import Mapping


def clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    """Clip a value to [min_value, max_value]."""
    return max(min_value, min(max_value, value))


def linear_score(
    value: float,
    ideal_value: float,
    warning_value: float,
    fail_value: float,
    *,
    higher_is_worse: bool = True,
) -> float:
    """
    Map a metric to 0–100 with smooth linear segments.

    When higher_is_worse=True (default):
      - value <= ideal_value  -> 100
      - ideal < value <= warning -> linear 100 -> 70
      - warning < value <= fail  -> linear 70 -> 40
      - value > fail_value      -> linear 40 -> 0 (clamped)

    When higher_is_worse=False the ordering is reversed (higher is better).
    """
    if value != value:  # NaN
        return 50.0

    if not higher_is_worse:
        return linear_score(
            -value,
            -ideal_value,
            -warning_value,
            -fail_value,
            higher_is_worse=True,
        )

    if value <= ideal_value:
        return 100.0
    if value <= warning_value:
        span = warning_value - ideal_value
        if span <= 0:
            return 70.0
        t = (value - ideal_value) / span
        return clamp(100.0 - t * 30.0)
    if value <= fail_value:
        span = fail_value - warning_value
        if span <= 0:
            return 40.0
        t = (value - warning_value) / span
        return clamp(70.0 - t * 30.0)

    # Beyond fail: decay toward 0
    overshoot = value - fail_value
    decay = min(overshoot / max(abs(fail_value), 1e-6), 1.0)
    return clamp(40.0 - decay * 40.0)


def score_angle_range(
    value: float,
    ideal_min: float,
    ideal_max: float,
    fail_low: float | None = None,
    fail_high: float | None = None,
) -> float:
    """
    Score when the ideal range is [ideal_min, ideal_max] (inclusive).

    100 inside the ideal band; linear decay outside toward fail_low / fail_high.
  """
    if value != value:
        return 50.0

    if ideal_min <= value <= ideal_max:
        return 100.0

    if value < ideal_min:
        fail = fail_low if fail_low is not None else ideal_min - (ideal_max - ideal_min)
        span = ideal_min - fail
        if span <= 0:
            return 40.0
        t = (ideal_min - value) / span
        return clamp(100.0 - t * 60.0)

    fail = fail_high if fail_high is not None else ideal_max + (ideal_max - ideal_min)
    span = fail - ideal_max
    if span <= 0:
        return 40.0
    t = (value - ideal_max) / span
    return clamp(100.0 - t * 60.0)


def weighted_average(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Weighted mean of dimension scores; ignores unknown keys and renormalizes."""
    total_w = 0.0
    total = 0.0
    for key, weight in weights.items():
        if key not in scores or scores[key] != scores[key]:
            continue
        total_w += weight
        total += scores[key] * weight
    if total_w <= 0:
        return 0.0
    return round(total / total_w, 1)


def quality_label(score: float, good_min: float = 80.0, acceptable_min: float = 70.0) -> str:
    """Map overall score to a coarse quality label."""
    if score >= good_min:
        return "good"
    if score >= acceptable_min:
        return "acceptable"
    return "needs_work"
