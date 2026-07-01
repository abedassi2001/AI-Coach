#!/usr/bin/env python3
"""Smoke-test Streamlit app flows (video buttons, cached load, summary modal)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def _ss(at: AppTest, key: str, default=None):
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default


def main() -> int:
    app_path = PROJECT_ROOT / "frontend" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run(timeout=60)
    if at.exception:
        _fail(f"App crashed on startup: {at.exception}")

    _ok("App starts without exception")

    # --- Demo tab: load cached sample_squat ---
    demo_tab = at.tabs[1]
    demo_tab.selectbox[0].set_value("sample_squat").run()
    if demo_tab.exception:
        _fail(f"Demo tab selectbox: {demo_tab.exception}")

    load_btn = next((b for b in demo_tab.button if b.key == "demo_load_cached"), None)
    if load_btn is None:
        load_btn = next((b for b in demo_tab.button if "Load cached" in (b.label or "")), None)
    if load_btn is None:
        _fail("Load cached button not found")
    load_btn.click().run()
    if at.exception:
        _fail(f"After Load cached: {at.exception}")

  # session state should have analysis complete
    if _ss(at, "analysis_status") != "complete":
        _fail(f"Expected analysis_status=complete, got {_ss(at, 'analysis_status')}")
    _ok("Load cached sets analysis_status=complete")

    # --- Annotated button on demo tab ---
    at.session_state.pending_summary_modal = False
    at.session_state.demo_choice = "sample_squat"
    at.run()
    ann_btn = next((b for b in demo_tab.button if "Annotated" in (b.label or "")), None)
    if ann_btn is None:
        _fail("Demo Annotated button not found")
    ann_btn.click().run()
    if at.exception:
        _fail(f"After Annotated click: {at.exception}")

    req = _ss(at, "video_modal_request")
    if not req or req.get("type") != "annotated":
        _fail(f"Expected video_modal_request annotated, got {req}")
    _ok("Demo Annotated queues video_modal_request")

    # Video dialog should render without nested-dialog crash
    errors = [e.value for e in at.error if e.value]
    if any("nested" in e.lower() for e in errors):
        _fail(f"Nested dialog error: {errors}")
    _ok("No nested-dialog error after Annotated click")

    # --- Upload video button after cached load ---
    at.session_state.video_modal_request = None
    at.run()
    upload_tab = at.tabs[0]
    watch_btns = [b for b in upload_tab.button if b.label and "Watch uploaded" in b.label]
    if not watch_btns:
        _fail("Watch uploaded button not found on upload tab after analysis")
    watch_btns[0].click().run()
    if at.exception:
        _fail(f"After Watch uploaded: {at.exception}")
    req2 = _ss(at, "video_modal_request")
    if not req2 or req2.get("type") != "upload":
        _fail(f"Expected upload video_modal_request, got {req2}")
    _ok("Watch uploaded queues upload modal")

    # --- Annotated from upload tab ---
    at.session_state.video_modal_request = None
    at.run()
    ann_upload = next(
        (b for b in upload_tab.button if b.label and "Watch annotated" in b.label),
        None,
    )
    if ann_upload is None:
        _fail("Watch annotated button not found on upload tab")
    ann_upload.click().run()
    if at.exception:
        _fail(f"After Watch annotated upload tab: {at.exception}")
    req3 = _ss(at, "video_modal_request")
    if not req3 or req3.get("type") != "annotated":
        _fail(f"Expected annotated request from upload tab, got {req3}")
    _ok("Watch annotated on upload tab works")

    # --- Summary modal should not open when video modal is queued ---
    if _ss(at, "pending_summary_modal") and _ss(at, "video_modal_request"):
        _fail("Both pending_summary_modal and video_modal_request set — dialog conflict risk")

    # --- Summary modal: watch buttons queue video panel (no nested dialog) ---
    at.session_state.video_modal_request = None
    at.session_state.pending_summary_modal = True
    at.run()
    if at.exception:
        _fail(f"Summary modal open: {at.exception}")
    _ok("Summary modal opens after cached load")

    # Find modal buttons - dialogs may appear in at.dialog or main
    def _all_buttons(container):
        btns = list(container.button)
        for d in getattr(container, "dialog", []) or []:
            btns.extend(d.button)
        return btns

    modal_ann = next(
        (b for b in _all_buttons(at) if b.label and "Watch annotated" in b.label),
        None,
    )
    if modal_ann is None:
        _fail("Watch annotated button not found in summary modal")
    modal_ann.click().run()
    if at.exception:
        _fail(f"Summary modal annotated click crashed: {at.exception}")
    if _ss(at, "video_modal_request", {}).get("type") != "annotated":
        _fail("Summary modal annotated did not queue video")
    _ok("Summary modal Watch annotated works (no nested dialog)")

    at.session_state.video_modal_request = None
    at.session_state.pending_summary_modal = True
    at.run()
    modal_up = next(
        (b for b in _all_buttons(at) if b.label and "Watch uploaded" in b.label),
        None,
    )
    if modal_up is None:
        _fail("Watch uploaded button not found in summary modal")
    modal_up.click().run()
    if at.exception:
        _fail(f"Summary modal upload click crashed: {at.exception}")
    if _ss(at, "video_modal_request", {}).get("type") != "upload":
        _fail("Summary modal upload did not queue video")
    _ok("Summary modal Watch uploaded works")

    # --- Full analysis panel video buttons ---
    at.session_state.video_modal_request = None
    at.session_state.pending_summary_modal = False
    at.session_state.show_full_analysis = True
    at.run()
    if at.exception:
        _fail(f"Full analysis panel: {at.exception}")

    full_ann = next(
        (b for b in _all_buttons(at) if b.label and "Watch annotated" in b.label and b.key == "full_watch_annotated"),
        None,
    )
    if full_ann is None:
        # fallback: any annotated in full section
        full_ann = next((b for b in at.button if b.key == "full_watch_annotated"), None)
    if full_ann is None:
        _fail("Full analysis Watch annotated button not found")
    full_ann.click().run()
    if at.exception:
        _fail(f"Full analysis annotated click: {at.exception}")
    if _ss(at, "video_modal_request", {}).get("type") != "annotated":
        _fail("Full analysis annotated did not queue video")
    _ok("Full analysis Watch annotated works")

    # --- Upload bytes path (pre-analysis watch) ---
    at.session_state.video_modal_request = None
    at.session_state.pending_summary_modal = False
    at.session_state.show_full_analysis = False
    at.session_state.analysis_result = None
    at.session_state.analysis_status = "idle"
    sample = PROJECT_ROOT / "data/raw/videos/sample_squat.mp4"
    at.session_state.upload_preview_bytes = sample.read_bytes()
    at.session_state.upload_preview_name = sample.name
    at.run()
    pre_btn = next(
        (b for b in at.tabs[0].button if b.key == "upload_tab_watch_raw"),
        None,
    )
    if pre_btn is None:
        _fail("Pre-analysis watch upload button missing")
    pre_btn.click().run()
    if at.exception:
        _fail(f"Pre-analysis upload watch: {at.exception}")
    if _ss(at, "video_modal_request", {}).get("type") != "upload":
        _fail("Pre-analysis upload did not open modal")
    _ok("Pre-analysis Watch uploaded works with upload bytes")

    # --- Analyze flow: starting analysis clears stale video panel ---
    at.session_state.video_modal_request = {"type": "upload"}
    at.session_state.upload_preview_bytes = sample.read_bytes()
    at.session_state.upload_preview_name = sample.name
    at.run()
    analyze_btn = next((b for b in at.tabs[0].button if "Analyze my squat" in (b.label or "")), None)
    if analyze_btn is None:
        _fail("Analyze my squat button not found")
    # Simulate what run_upload does before pipeline
    at.session_state.video_modal_request = None
    at.run()
    if _ss(at, "video_modal_request"):
        _fail("video_modal_request should be clear before analysis results")
    _ok("Analyze flow clears stale video panel state")

    # --- Legacy app/ path shim ---
    compat = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    compat.run(timeout=60)
    if compat.exception:
        _fail(f"app/streamlit_app.py compat shim: {compat.exception}")
    _ok("app/streamlit_app.py compat shim works")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
