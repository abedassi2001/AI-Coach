# Development phases — commit checklist

Each phase ends with: **files changed**, **what was implemented**, **how to test**, **commit message**.

---

## Phase 1 — Repository setup ✅

**Files changed:** Full tree under `src/`, `configs/`, `data/`, `docs/`, `tests/`, root configs.

**Implemented:** Project skeleton, README, requirements, Docker scaffold, default YAML config, dataset docs, minimal smoke test.

**How to test:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest tests/test_package.py -v
```

**Commit message:**
```
chore: initialize AI Gym Form Coach project structure

Set up src package layout, configs, docs, Docker scaffold,
requirements, and gitignore for squat-form MVP pipeline.
```

---

## Phase 2 — Video processing (next)

**Planned files:** `src/data/video_loader.py`, `src/data/frame_extractor.py`, `src/utils/config.py`, `scripts/extract_frames.py`, `tests/test_video.py`

**Commit message (suggested):**
```
feat(data): add video loading and frame extraction pipeline
```
