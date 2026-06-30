"""End-to-end inference on new exercise videos."""

from src.inference.video_pipeline import (
    PipelineResult,
    list_demo_videos,
    load_existing_result,
    run_full_pipeline,
    stage_uploaded_video,
)

__all__ = [
    "PipelineResult",
    "list_demo_videos",
    "load_existing_result",
    "run_full_pipeline",
    "stage_uploaded_video",
]
