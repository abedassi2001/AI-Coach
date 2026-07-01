"""
Backward-compatible Streamlit entry.

The UI lives in ``frontend/streamlit_app.py``. This shim keeps old commands working:

    streamlit run app/streamlit_app.py

Prefer: ``python scripts/run_app.py``
"""

from __future__ import annotations

import runpy
from pathlib import Path

_FRONTEND_APP = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"

if not _FRONTEND_APP.exists():
    raise FileNotFoundError(
        f"Missing frontend app: {_FRONTEND_APP}\n"
        "Run from the repo root after pulling the latest code."
    )

runpy.run_path(str(_FRONTEND_APP), run_name="__main__")
