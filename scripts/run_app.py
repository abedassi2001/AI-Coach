#!/usr/bin/env python3
"""Launch the Streamlit demo app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "app" / "streamlit_app.py"


def main() -> int:
    ensure = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/ensure_model.py")],
        cwd=PROJECT_ROOT,
    )
    if ensure.returncode != 0:
        print("Warning: could not verify ML model (rule-based analysis still works).", file=sys.stderr)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.headless",
        "true",
    ]
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
