# Dataset Guide

This project is an **AI fitness-form assistant**, not a medical diagnosis tool. All training and evaluation data must respect licensing and privacy.

## MVP data sources

For early development you can use:

1. **Manually recorded videos** — Record squats from the side and front angles. Label reps and form quality in a simple CSV.
2. **Public workout videos** — Only use clips where license permits redistribution and derivative ML use (e.g. Creative Commons, explicit permission).

Always document source, license, and any preprocessing in `data/raw/README.md`.

## Research datasets (future integration)

| Dataset | Description | Notes |
|---------|-------------|-------|
| [Fit3D](https://fit3d.imar.ro/) | 3D human pose during fitness exercises | Strong for pose; check license for your use case |
| [M3GYM](https://github.com/m3gym/m3gym) | Multi-modal gym exercise data | Useful for exercise recognition and form |
| Smaller exercise-form sets | Various academic releases | Good for prototyping classifiers |

## Recommended folder layout

```
data/
  raw/
    videos/           # Original uploads
    labels/           # Rep boundaries, mistake tags (CSV/JSON)
    README.md         # Provenance per file
  interim/
    frames/           # Extracted frames
    sample_frames/    # Debug samples
  processed/
    pose/             # Per-video keypoint JSON/CSV
    features/         # Per-frame angle & movement features
    reps/             # Segmented rep sequences
```

## Label schema (suggested)

```csv
video_id,rep_id,start_frame,end_frame,depth_ok,torso_lean_ok,knee_valgus,overall_quality
squat_001,1,45,120,1,1,0,good
squat_001,2,125,200,0,1,1,bad
```

## Licensing checklist

- [ ] Confirm redistribution rights
- [ ] Confirm ML training / fine-tuning allowed
- [ ] Anonymize faces if required
- [ ] Record dataset version in experiment logs
