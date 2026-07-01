"""Synthetic rep augmentation for development / contrastive training."""

from __future__ import annotations

import copy
import random
from typing import Any, Callable

import pandas as pd

# Biomechanically meaningful perturbations applied to rep-level aggregates.
# Each function receives the current column value and returns a worsened value.
MISTAKE_PERTURBATIONS: dict[str, dict[str, Callable[[float], float]]] = {
    "shallow_squat": {
        "rep_bottom_knee_angle": lambda v: v + 28.0,
        "rep_squat_depth_angle_at_bottom": lambda v: v + 25.0,
        "rep_knee_angle_min_at_bottom": lambda v: v + 25.0,
        "rep_knee_angle_min_min": lambda v: v + 22.0,
        "rep_hip_drop_norm_at_bottom": lambda v: v * 0.45,
    },
    "high_asymmetry": {
        "rep_knee_asymmetry_deg_at_bottom": lambda v: v + 30.0,
        "rep_knee_asymmetry_deg_mean": lambda v: v + 22.0,
        "rep_knee_asymmetry_deg_max": lambda v: v + 18.0,
    },
    "excessive_lean": {
        "rep_angle_torso_lean_at_bottom": lambda v: v + 28.0,
        "rep_angle_torso_lean_max": lambda v: v + 22.0,
        "rep_angle_torso_lean_mean": lambda v: v + 15.0,
    },
    "unstable_path": {
        "rep_knee_angle_min_std": lambda v: v + 18.0,
        "rep_knee_angle_avg_std": lambda v: v + 14.0,
        "rep_knee_angle_min_mean": lambda v: v + 12.0,
    },
    "knee_valgus": {
        "rep_knee_asymmetry_deg_at_bottom": lambda v: v + 20.0,
        "rep_angle_left_knee_at_bottom": lambda v: v - 15.0,
        "rep_angle_right_knee_at_bottom": lambda v: v + 10.0,
    },
}


def _apply_perturbation(row: dict[str, Any], mistake_id: str) -> dict[str, Any]:
    out = copy.deepcopy(row)
    rules = MISTAKE_PERTURBATIONS.get(mistake_id, {})
    for col, fn in rules.items():
        if col not in out:
            continue
        val = out.get(col)
        if val is None or (isinstance(val, float) and val != val):
            continue
        out[col] = float(fn(float(val)))
    return out


def generate_contrastive_bad_reps(
    good_rows: list[dict[str, Any]],
    variants_per_rep: int = 4,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Create labeled ``bad`` reps by perturbing real ``good`` reps toward common mistakes.

    Teaches the model what *good* looks like (from human labels) vs plausible errors.
    """
    rng = random.Random(random_state)
    mistake_ids = list(MISTAKE_PERTURBATIONS.keys())
    synthetic: list[dict[str, Any]] = []
    seq = 0

    for base in good_rows:
        if str(base.get("label", "")).lower() != "good":
            continue
        picks = mistake_ids[:]
        rng.shuffle(picks)
        for mistake_id in picks[:variants_per_rep]:
            seq += 1
            row = _apply_perturbation(base, mistake_id)
            row["source_id"] = f"synthetic_bad_{seq}"
            row["rep_id"] = seq
            row["label"] = "bad"
            row["synthetic"] = True
            row["synthetic_mistake"] = mistake_id
            synthetic.append(row)

    return synthetic


def generate_good_jitter_reps(
    good_rows: list[dict[str, Any]],
    copies_per_rep: int = 2,
    noise_scale: float = 0.03,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Small noise around good reps — same label, different source_id for split diversity."""
    rng = random.Random(random_state)
    numeric_cols = [
        c
        for c in good_rows[0].keys()
        if c not in ("source_id", "exercise", "rep_id", "label", "weak_label", "synthetic", "synthetic_mistake", "label_source")
        and isinstance(good_rows[0].get(c), (int, float))
    ]
    out: list[dict[str, Any]] = []
    seq = 0
    for base in good_rows:
        if str(base.get("label", "")).lower() != "good":
            continue
        for _ in range(copies_per_rep):
            seq += 1
            row = copy.deepcopy(base)
            row["source_id"] = f"synthetic_good_{seq}"
            row["rep_id"] = seq
            row["synthetic"] = True
            for col in numeric_cols:
                val = row.get(col)
                if val is None or (isinstance(val, float) and val != val):
                    continue
                delta = abs(float(val)) * noise_scale * rng.uniform(-1, 1)
                row[col] = float(val) + delta
            out.append(row)
    return out


def augment_labeled_reps(
    df: pd.DataFrame,
    target_size: int,
    noise_scale: float = 0.04,
    random_state: int = 42,
    mode: str = "contrastive",
    variants_per_rep: int = 4,
) -> pd.DataFrame:
    """
    Expand a small labeled set for training.

    ``contrastive`` (default): good human labels + biomechanical bad variants.
    ``noise``: legacy random perturbation (development only).
    """
    if df.empty:
        return df

    rows = df.to_dict(orient="records")
    good_rows = [r for r in rows if str(r.get("label", "")).lower() == "good"]
    bad_rows = [r for r in rows if str(r.get("label", "")).lower() == "bad"]

    if mode == "contrastive" and good_rows:
        expanded: list[dict[str, Any]] = rows[:]
        expanded.extend(generate_contrastive_bad_reps(good_rows, variants_per_rep=variants_per_rep, random_state=random_state))
        expanded.extend(generate_good_jitter_reps(good_rows, copies_per_rep=2, noise_scale=noise_scale, random_state=random_state + 1))
        expanded.extend(bad_rows)
        result = pd.DataFrame(expanded)
        if len(result) < target_size:
            # Top up with extra bad variants from good templates
            extra = generate_contrastive_bad_reps(
                good_rows,
                variants_per_rep=max(1, (target_size - len(result)) // max(len(good_rows), 1)),
                random_state=random_state + 99,
            )
            for i, row in enumerate(extra):
                row["source_id"] = f"synthetic_bad_extra_{i}"
            result = pd.concat([result, pd.DataFrame(extra)], ignore_index=True)
        return result.head(target_size) if len(result) > target_size else result

    # Legacy noise mode
    rng = random.Random(random_state)
    numeric_cols = [
        c
        for c in df.columns
        if c not in ("source_id", "exercise", "rep_id", "label", "weak_label", "label_source", "synthetic", "synthetic_mistake")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    out = rows[:]
    i = 0
    while len(out) < target_size:
        base = copy.deepcopy(rows[i % len(rows)])
        base["source_id"] = f"synthetic_{len(out)}"
        base["rep_id"] = len(out) + 1
        for col in numeric_cols:
            val = base.get(col)
            if val is None or (isinstance(val, float) and val != val):
                continue
            delta = abs(float(val)) * noise_scale * rng.uniform(-1, 1)
            base[col] = float(val) + delta
        if len(out) % 3 == 0 and base.get("label") == "good":
            base["label"] = "bad"
        elif len(out) % 5 == 0 and base.get("label") == "bad":
            base["label"] = "good"
        base["synthetic"] = True
        out.append(base)
        i += 1

    return pd.DataFrame(out)
