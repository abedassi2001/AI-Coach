"""Tests for video modal path resolution and browser playback prep."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.video_pipeline import load_existing_result
from src.utils.web_video import (
    ensure_browser_playable,
    find_ffmpeg,
    needs_browser_transcode,
    prepare_video_for_browser,
    resolve_playable_path,
    video_codec_fourcc,
)


def test_ffmpeg_available_via_imageio() -> None:
    assert find_ffmpeg() is not None


@pytest.mark.parametrize("source_id", ["sample_squat", "good_goblet_52106"])
def test_cached_evaluation_video_is_playable(source_id: str) -> None:
    result = load_existing_result(source_id)
    assert result is not None, f"no cached pipeline result for {source_id}"
    assert result.evaluation_video is not None
    assert result.evaluation_video.exists(), result.evaluation_video

    playable = ensure_browser_playable(result.evaluation_video)
    assert playable.exists()
    if find_ffmpeg():
        assert video_codec_fourcc(playable) in {"h264", "avc1", "H264", "AVC1"}

    data, mime = prepare_video_for_browser(playable)
    assert len(data) > 10_000
    assert mime == "video/mp4"


def test_raw_upload_bytes_playable() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/raw/videos/sample_squat.mp4"
    if not sample.exists():
        pytest.skip("sample_squat.mp4 missing")

    path = resolve_playable_path(sample.read_bytes(), filename=sample.name)
    assert path.exists()
    assert path.stat().st_size > 10_000


def test_opencv_eval_video_gets_transcoded_when_needed() -> None:
    root = Path(__file__).resolve().parents[1]
    eval_mp4 = root / "data/processed/evaluation/sample_squat/sample_squat_evaluation.mp4"
    if not eval_mp4.exists():
        pytest.skip("evaluation mp4 missing")
    if not find_ffmpeg():
        pytest.skip("ffmpeg not available")

    if needs_browser_transcode(eval_mp4):
        web = ensure_browser_playable(eval_mp4)
        assert web.name.endswith("_web.mp4")
        assert not needs_browser_transcode(web)
