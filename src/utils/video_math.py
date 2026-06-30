"""Pure math helpers for video frame sampling (no OpenCV dependency)."""


def compute_sample_stride(source_fps: float, target_fps: float) -> int:
    """How many source frames to skip between extractions (minimum 1)."""
    if target_fps <= 0 or source_fps <= 0:
        return 1
    if source_fps <= target_fps:
        return 1
    return max(1, int(round(source_fps / target_fps)))
