"""Synthetic rep augmentation for development / tests only."""

from __future__ import annotations

import copy
import random
from typing import Any

import pandas as pd


def augment_labeled_reps(
    df: pd.DataFrame,
    target_size: int,
    noise_scale: float = 0.04,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Perturb numeric rep features to reach target_size rows.

    FOR DEVELOPMENT ONLY — not a substitute for real labeled data.
    """
    if df.empty:
        return df
    rng = random.Random(random_state)
    rows: list[dict[str, Any]] = df.to_dict(orient="records")
    numeric_cols = [
        c
        for c in df.columns
        if c not in ("source_id", "exercise", "rep_id", "label", "weak_label")
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
        # flip some labels for class balance
        if len(out) % 3 == 0 and base.get("label") == "good":
            base["label"] = "bad"
        elif len(out) % 5 == 0 and base.get("label") == "bad":
            base["label"] = "good"
        base["synthetic"] = True
        out.append(base)
        i += 1

    return pd.DataFrame(out)
