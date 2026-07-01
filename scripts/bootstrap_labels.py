#!/usr/bin/env python3
"""Bootstrap rep_labels.csv from rule-based form analysis (weak labels)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.rep_dataset import RepDatasetBuilder, discover_processed_sources, load_label_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Write weak labels from Phase 6 rules.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/raw/labels/rep_labels.csv"),
    )
    parser.add_argument(
        "--overwrite-human",
        action="store_true",
        help="Replace human-labeled rows (default: human labels are preserved)",
    )
    args = parser.parse_args()

    builder = RepDatasetBuilder()
    df = builder.labeled_only(builder.build(source_ids=discover_processed_sources()))

    if df.empty:
        print("No processed reps found. Run pose → features → reps pipeline first.", file=sys.stderr)
        return 1

    weak = df[["source_id", "exercise", "rep_id", "label"]].copy()
    weak["label_source"] = "rules"

    existing = load_label_table(args.output)
    if not existing.empty and not args.overwrite_human:
        human = existing[existing.get("label_source", pd.Series(dtype=str)) == "human"]
        if human.empty and "label_source" not in existing.columns:
            human = pd.DataFrame()
        if not human.empty:
            keys = set(zip(human["source_id"], human["exercise"], human["rep_id"]))
            mask = ~weak.apply(
                lambda r: (r["source_id"], r["exercise"], r["rep_id"]) in keys,
                axis=1,
            )
            weak = pd.concat([human, weak[mask]], ignore_index=True)
            print(f"Preserved {len(human)} human label(s); filled gaps with rules.")

    if "weak_label" in df.columns:
        weak_map = {
            (str(r.source_id), str(r.exercise), int(r.rep_id)): bool(r.weak_label)
            for r in df.itertuples(index=False)
            if hasattr(r, "weak_label")
        }
        weak["weak_label"] = weak.apply(
            lambda r: weak_map.get((str(r["source_id"]), str(r["exercise"]), int(r["rep_id"]))),
            axis=1,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weak.to_csv(args.output, index=False)
    print(f"Wrote {len(weak)} labels to {args.output.resolve()}")
    print(weak.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
