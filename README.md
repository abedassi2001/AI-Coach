# AI Gym Form Coach

An AI-powered **fitness form assistant** that analyzes gym exercise videos, estimates body pose, segments repetitions, detects common form mistakes, and generates actionable coaching feedback.

> **Disclaimer:** This is not a medical diagnosis tool. It provides visual technique feedback based on pose estimation and movement analysis. Consult a qualified coach or healthcare professional for injury concerns.

## MVP scope

- **Exercise:** Squat (single-exercise focus)
- **Pipeline:** Video → pose keypoints → joint angles → rep segmentation → form analysis → feedback
- **Approach:** Rule-based baseline first, then scikit-learn classifier, then sequence model (LSTM / Temporal CNN / Transformer)

## Tech stack

| Layer | Tools |
|-------|--------|
| Language | Python 3.11+ |
| Deep learning | PyTorch |
| Vision | OpenCV, MediaPipe Pose (YOLO Pose optional) |
| Tabular ML | NumPy, Pandas, scikit-learn |
| Sequence models | PyTorch (LSTM / Transformer) |
| API (later) | FastAPI |
| UI (later) | Streamlit or React |
| Deploy | Docker |
| VCS | Git / GitHub |

## Project structure

```
AI-Coach/
├── backend/                # Python CV/ML pipeline
│   ├── data/               # Video loading, frame extraction
│   ├── pose/               # MediaPipe pose estimation
│   ├── features/           # Joint angles, rep segmentation
│   ├── feedback/           # Rule engine, scoring, coaching
│   ├── inference/          # End-to-end video pipeline
│   ├── ml/                 # Classifier architectures (code)
│   ├── training/           # Dataset builders, training, metrics
│   ├── visualization/      # Skeleton overlay, evaluation videos
│   └── utils/              # Config, paths, web video helpers
├── frontend/               # Streamlit UI (Iron Form Coach)
├── models/                 # Saved weights & pose assets (gitignored)
├── configs/                # YAML configuration
├── data/                   # Raw videos, labels, processed artifacts
├── scripts/                # CLI entry points
├── tests/                  # Unit & integration tests
├── docs/                   # Architecture, phases, datasets
├── pyproject.toml
├── requirements.txt
├── Dockerfile
└── README.md
```

| Layer | Directory | Role |
|-------|-----------|------|
| **Backend** | `backend/` | Pose, features, scoring, inference, training |
| **Frontend** | `frontend/` | Streamlit upload → analyze → results UI |
| **Models** | `models/` | Checkpoints on disk (`.joblib`, `.task`) |
| **ML code** | `backend/ml/` | Classifier implementations (not weights) |

See [backend/README.md](backend/README.md), [frontend/README.md](frontend/README.md), [models/README.md](models/README.md), and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Pose extraction (Phase 3)

```bash
# From video
python scripts/extract_pose.py --video path/to/squat.mp4 --exercise squat --overlay

# From Phase 2 frames
python scripts/extract_pose.py --frames-dir data/interim/squat_clip --exercise squat
```

Output: `data/processed/pose/<video_id>/keypoints.json` (+ optional CSV and skeleton video).

## Feature engineering (Phase 4)

```bash
python scripts/compute_features.py data/processed/pose/sample_squat/keypoints.json
```

Output: `data/processed/features/<video_id>/features.csv` and `features.json`.

## Rep segmentation (Phase 5)

```bash
python scripts/segment_reps.py data/processed/features/sample_squat/features.csv
```

Output: `data/processed/reps/<video_id>/reps.json` with rep boundaries and phases.

## Form analysis (Phase 6) — continuous 0–100 scoring

Primary scoring engine: **rule-based biomechanical analysis** with per-dimension scores (not ML).

```bash
python scripts/analyze_form.py sample_squat
```

Output: `data/processed/analysis/<video_id>/form_analysis.json` with:

- Per-rep scores: `depth_score`, `knee_tracking_score`, `torso_control_score`, `symmetry_score`, `stability_score`, `heel_control_score`, `overall_score` (0–100)
- Confidence notes (pose, camera angle, heel detection)
- Flags and deterministic coaching feedback per rep
- Video-level summary (average scores, best/worst dimension)

See [docs/CONTINUOUS_SCORING.md](docs/CONTINUOUS_SCORING.md) for architecture, thresholds, and example JSON.

## ML baseline (Phase 7) — **experimental**

Binary `good`/`bad` classifier only. Labels in `data/raw/labels/rep_labels.csv` are **not** per-feature scores — use the rule engine above for coaching.

```bash
python scripts/bootstrap_labels.py
python scripts/train_classifier.py --demo
python scripts/predict_rep_quality.py sample_squat
```

See [docs/MODELS.md](docs/MODELS.md). Future: overall score regressor once human 1–100 labels are collected.

## AI coaching feedback (Phase 9) — **free by default**

Actionable cues from your analysis. **No API key or payment required.**

```bash
# Free built-in coach (default)
python scripts/generate_coaching.py sample_squat

# Optional: free local AI via Ollama (install from ollama.com)
python scripts/generate_coaching.py sample_squat --provider ollama

# With annotated video
python scripts/visualize_evaluation.py sample_squat --coaching
```

Output: `data/processed/coaching/<video_id>/coaching_report.txt`

See [docs/COACHING.md](docs/COACHING.md). Paid OpenAI is optional only if you add a key later.

## Demo app (Phase 10)

Interactive UI to upload a squat video and review the full pipeline:

```bash
python scripts/run_app.py
# or: streamlit run frontend/streamlit_app.py
# legacy: streamlit run app/streamlit_app.py  (compat shim)
```

Opens a browser at `http://localhost:8501`.

**User flow:** upload → analyze → **summary popup** (score, main issue, quick fix) → **View full analysis** for dimension cards and rep breakdown.

See [frontend/README.md](frontend/README.md) for UI architecture and screenshots placeholder.

## Development phases

| Phase | Focus | Status |
|-------|--------|--------|
| 1 | Repository setup | ✅ Done |
| 2 | Video processing | ✅ Done |
| 3 | Pose extraction | ✅ Done |
| 4 | Feature engineering | ✅ Done |
| 5 | Rep segmentation | ✅ Done |
| 6 | Rule-based form analyzer | ✅ Done |
| 7 | ML baseline classifier | ✅ Done |
| 8 | Deep learning sequence model | Pending |
| 9 | Feedback generation | ✅ Done |
| 10 | Demo app (Streamlit) | ✅ Done |

## Quick start

```bash
# Clone and enter project
cd AI-Coach

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Verify structure (after Phase 2+)
pytest tests/
```

## Git workflow

1. Work on a feature branch: `git checkout -b phase-2-video-processing`
2. Commit at the end of each phase with a clear message (see phase notes in docs)
3. Open a PR to `main` for review before merging

### Suggested first commit

```bash
git init
git add .
git commit -m "chore: initialize AI Gym Form Coach project structure

Set up src package layout, configs, docs, Docker scaffold,
requirements, and gitignore for squat-form MVP pipeline."
```

## Squat form rules (target mistakes)

- Knees collapsing inward (valgus)
- Excessive forward torso lean
- Insufficient depth
- Heels lifting
- Asymmetric movement
- Unstable bar / body path

## License

TBD — add a license file before public release.

## Author

CV / portfolio project — AI Gym Form Coach.
