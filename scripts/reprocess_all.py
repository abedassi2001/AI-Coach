#!/usr/bin/env python3
"""Recompute features + reps for all processed pose recordings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.rep_dataset import discover_processed_sources


def main() -> int:
    py = sys.executable
    for sid in discover_processed_sources():
        pose = PROJECT_ROOT / "data/processed/pose" / sid / "keypoints.json"
        if not pose.exists():
            continue
        print(f"\n=== Reprocess {sid} ===")
        subprocess.run([py, "scripts/compute_features.py", str(pose)], cwd=PROJECT_ROOT, check=True)
        feat = PROJECT_ROOT / "data/processed/features" / sid / "features.csv"
        subprocess.run(
            [py, "scripts/segment_reps.py", str(feat), "--signal", "squat_depth_angle"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run([py, "scripts/analyze_form.py", sid], cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
