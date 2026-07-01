# Adding exercises to AI Gym Form Coach

Pose extraction is **exercise-agnostic** — MediaPipe detects the same 33 body landmarks whether the athlete is squatting, deadlifting, or pressing.

Exercise-specific logic lives in separate config files under `configs/exercises/`.

## Add a new exercise

1. Copy `configs/exercises/_template.yaml` to `configs/exercises/<id>.yaml`
2. Define `landmark_groups` and `angles` for that movement
3. Add `form_mistakes` IDs for Phase 6 rule engine
4. Run pose extraction with `--exercise <id>` to tag outputs

```bash
python scripts/extract_pose.py --video deadlift.mp4 --exercise deadlift
```

## Architecture layers

| Layer | Scope | Location |
|-------|--------|----------|
| Pose detection | All exercises | `backend/pose/` |
| Exercise metadata | Per movement | `configs/exercises/*.yaml` |
| Feature / form rules | Per movement | `backend/features/`, Phase 4–6 |

## Squat (MVP)

See `configs/exercises/squat.yaml` for landmark groups and angle definitions used in upcoming phases.
