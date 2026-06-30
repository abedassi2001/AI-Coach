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
├── src/                    # Application source code
│   ├── data/               # Video loading, frame extraction, dataset I/O
│   ├── pose/               # Pose estimation (MediaPipe / YOLO)
│   ├── features/           # Joint angles, normalization, movement features
│   ├── models/             # Model architectures (ML + deep learning)
│   ├── training/           # Training loops, metrics, checkpoints
│   ├── inference/          # End-to-end prediction on new videos
│   ├── feedback/           # Rule-based + model output → coaching text
│   ├── visualization/      # Skeleton overlay, charts, annotated frames
│   └── utils/              # Config loading, logging, geometry helpers
├── configs/                # YAML configuration files
├── data/                   # Local data (gitignored contents; structure tracked)
│   ├── raw/                # Original videos and labels
│   ├── interim/            # Frames and intermediate artifacts
│   └── processed/          # Keypoints, features, segmented reps
├── notebooks/              # Exploratory analysis and prototyping
├── scripts/                # CLI entry points for batch jobs
├── tests/                  # Unit and integration tests
├── app/                    # FastAPI / Streamlit demo (Phase 10)
├── docs/                   # Documentation, dataset notes, architecture
├── models/                 # Saved weights and exports (gitignored)
├── requirements.txt
├── Dockerfile
└── README.md
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline, [docs/DATASETS.md](docs/DATASETS.md) for data licensing, and [docs/EXERCISES.md](docs/EXERCISES.md) for adding new gym movements.

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

## Development phases

| Phase | Focus | Status |
|-------|--------|--------|
| 1 | Repository setup | ✅ Done |
| 2 | Video processing | ✅ Done |
| 3 | Pose extraction | ✅ Done |
| 4 | Feature engineering | ✅ Done |
| 5 | Rep segmentation | Pending |
| 6 | Rule-based form analyzer | Pending |
| 7 | ML baseline classifier | Pending |
| 8 | Deep learning sequence model | Pending |
| 9 | Feedback generation | Pending |
| 10 | Demo app | Pending |

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
