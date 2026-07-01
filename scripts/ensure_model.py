#!/usr/bin/env python3
"""Retrain ML checkpoint if missing or incompatible with installed sklearn."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = PROJECT_ROOT / "models/checkpoints/baseline/form_classifier.joblib"


def main() -> int:
    if MODEL.exists():
        try:
            from src.inference.model_loader import try_load_predictor

            if try_load_predictor(MODEL) is not None:
                print(f"Model OK: {MODEL}")
                return 0
        except Exception:
            pass
        print("Model incompatible or corrupt — retraining…")
    else:
        print("Model not found — training…")

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/train_classifier.py"), "--demo"]
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
