# Continuous Form Scoring

## Overview

The squat analyzer (`SquatFormAnalyzer`) produces **explainable, continuous 0–100 scores** per biomechanical dimension for each segmented rep. This is the **primary** scoring engine for the portfolio project.

The optional sklearn classifier (`form_classifier.joblib`) remains **experimental** — it only has binary `good`/`bad` labels in `rep_labels.csv` and is not used for per-feature feedback.

## Pipeline

```
Video
  → pose keypoints (MediaPipe)
  → per-frame features (angles, derived metrics)
  → rep segmentation
  → continuous rule-based scoring (this document)
  → form_analysis.json
  → coaching layer (template / GPT)
  → UI visualization
```

## Dimensions

| Score | Source metric | Interpretation |
|-------|---------------|----------------|
| `depth_score` | Bottom knee angle | Lower angle = deeper squat = higher score |
| `knee_tracking_score` | Valgus proxy from hip–knee–ankle alignment | Best from front view |
| `torso_control_score` | Torso lean at bottom | Less forward lean = higher score |
| `symmetry_score` | Left/right knee asymmetry at bottom | More balanced = higher score |
| `stability_score` | Knee-angle std dev over rep | Smoother path = higher score |
| `heel_control_score` | Heel vs ankle height at bottom | Heels down = higher score; neutral if heels not visible |
| `overall_score` | Weighted average of the above | Default weights in `configs/form_scoring/squat.yaml` |

### Score bands

- **100** — excellent
- **70–89** — acceptable / minor issue
- **40–69** — noticeable issue
- **0–39** — serious issue

Flags (e.g. `shallow_depth`) are **derived from scores**, not independent pass/fail checks.

## Configuration

- **Thresholds & weights:** `configs/form_scoring/squat.yaml`
- **Legacy rule keys:** `configs/default.yaml` → `form_rules` (still used for mistake message thresholds)

## Output schema

`data/processed/analysis/<source_id>/form_analysis.json`:

```json
{
  "analyzer_version": "0.2.0-continuous-scoring",
  "source_id": "sample_squat",
  "exercise": "squat",
  "overall_score": 72.0,
  "overall_quality": "acceptable",
  "video_summary": {
    "num_reps": 3,
    "average_overall_score": 72.0,
    "average_scores": { "depth_score": 62.0, "...": 80.0 },
    "best_dimension": "knee_tracking_score",
    "worst_dimension": "depth_score",
    "main_issues": ["shallow_depth"]
  },
  "repetitions": [
    {
      "rep_id": 1,
      "overall_score": 76.0,
      "scores": {
        "depth_score": 62.0,
        "knee_tracking_score": 84.0,
        "torso_control_score": 71.0,
        "symmetry_score": 80.0,
        "stability_score": 75.0,
        "heel_control_score": 70.0,
        "overall_score": 76.0
      },
      "confidence": {
        "pose_confidence": 0.91,
        "camera_angle_confidence": "medium",
        "heel_detection_confidence": "low"
      },
      "flags": ["shallow_depth"],
      "feedback": ["Overall, this rep is acceptable but has room to improve.", "..."],
      "coaching": {
        "overall_summary": "...",
        "top_issues": ["..."],
        "positive_point": "Your knees track well overall.",
        "correction_cue": "..."
      },
      "mistakes": [],
      "metrics": { "bottom_knee_angle": 82.0 }
    }
  ]
}
```

`form_score` is retained as a backward-compatible alias for `overall_score`.

## Key modules

| Module | Role |
|--------|------|
| `backend/feedback/scoring.py` | Generic 0–100 mapping utilities |
| `backend/feedback/squat_dimensions.py` | Per-dimension squat scorers |
| `backend/feedback/squat_metrics.py` | Raw heel / valgus measurements |
| `backend/feedback/rep_coaching.py` | Flags, coaching text, video summary |
| `backend/feedback/form_analyzer.py` | Orchestrator |

## Why rules first, not ML?

1. **Interpretability** — coaches and users see *why* a score changed.
2. **Label gap** — current CSV labels are binary per rep, not per dimension.
3. **Small dataset** — ~94 reps is insufficient for reliable multi-output ML.
4. **Camera sensitivity** — explicit confidence fields document when a metric is unreliable.

## Limitations

- Valgus detection is weak from pure side view; `camera_angle_confidence` reflects this.
- Heel lift depends on heel landmark visibility.
- Depth from 2D pose is approximate; side view is preferred.
- Rep segmentation quality directly affects per-rep scores.

## Roadmap

1. Collect human per-rep overall scores (1–100)
2. Train overall score regressor (optional calibration layer)
3. Multi-task ML for per-feature scores once dimension labels exist
4. Improve camera-angle robustness and rep segmentation
5. UI dashboard for dimension breakdown

## ML classifier (experimental)

```bash
python scripts/train_classifier.py --demo
```

Uses `data/raw/labels/rep_labels.csv` (binary only). Kept for comparison and future work — **not** the source of per-feature scores in the UI.
