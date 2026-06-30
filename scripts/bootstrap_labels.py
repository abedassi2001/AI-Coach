#!/usr/bin/env python3
"""Bootstrap rep_labels.csv from rule-based form analysis (weak labels)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.rep_dataset import RepDatasetBuilder, discover_processed_sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Write weak labels from Phase 6 rules.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/raw/labels/rep_labels.csv"),
    )
    args = parser.parse_args()

    builder = RepDatasetBuilder()
    df = builder.labeled_only(builder.build(source_ids=discover_processed_sources()))

    if df.empty:
        print("No processed reps found. Run pose → features → reps pipeline first.", file=sys.stderr)
        return 1

    out = df[["source_id", "exercise", "rep_id", "label"]].copy()
    if "weak_label" in df.columns:
        out["weak_label"] = df["weak_label"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} labels to {args.output.resolve()}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
