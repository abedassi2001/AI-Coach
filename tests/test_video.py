"""Tests for OpenCV video loading and frame extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from src.data.frame_extractor import FrameExtractor, resize_frame
from src.data.video_loader import VideoLoader


def test_resize_frame_downscales_large_image():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = resize_frame(frame, max_width=1280, max_height=720)
    h, w = out.shape[:2]
    assert w <= 1280
    assert h <= 720
    assert w == 1280


def test_resize_frame_skips_upscale():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = resize_frame(frame, max_width=1280, max_height=720)
    assert out.shape == frame.shape


def test_video_loader_metadata(synthetic_video: Path):
    loader = VideoLoader(synthetic_video)
    meta = loader.get_metadata()
    assert meta.width == 640
    assert meta.height == 480
    assert meta.frame_count == 30
    assert meta.fps == pytest.approx(60.0, rel=0.05)
    assert meta.duration_sec == pytest.approx(0.5, rel=0.05)
    loader.close()


def test_video_loader_context_manager(synthetic_video: Path):
    with VideoLoader(synthetic_video) as loader:
        cap = loader.open()
        ok, frame = cap.read()
        assert ok
        assert frame.shape == (480, 640, 3)


def test_video_loader_missing_file():
    with pytest.raises(FileNotFoundError):
        VideoLoader("nonexistent_video.mp4")


def test_frame_extractor_saves_frames(synthetic_video: Path, tmp_path: Path):
    out_dir = tmp_path / "frames"
    extractor = FrameExtractor(target_fps=30, max_width=320, max_height=240)
    result = extractor.extract(synthetic_video, output_dir=out_dir)

    assert result.sample_stride == 2
    assert len(result.frames) == 15
    assert result.effective_fps == pytest.approx(30.0, rel=0.05)

    saved = list(out_dir.glob("frame_*.jpg"))
    assert len(saved) == 15
    assert (out_dir / "manifest.json").exists()

    img = cv2.imread(str(saved[0]))
    assert img is not None
    h, w = img.shape[:2]
    assert w <= 320 and h <= 240


def test_frame_extractor_manifest_content(synthetic_video: Path, tmp_path: Path):
    out_dir = tmp_path / "frames"
    extractor = FrameExtractor(target_fps=30)
    result = extractor.extract(synthetic_video, output_dir=out_dir)

    with (out_dir / "manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["frame_count"] == len(result.frames)
    assert manifest["frames"][0]["source_index"] == 0
    assert manifest["metadata"]["width"] == 640


def test_frame_extractor_max_frames(synthetic_video: Path, tmp_path: Path):
    out_dir = tmp_path / "frames"
    extractor = FrameExtractor(target_fps=30)
    result = extractor.extract(synthetic_video, output_dir=out_dir, max_frames=5)
    assert len(result.frames) == 5
    assert len(list(out_dir.glob("frame_*.jpg"))) == 5


def test_frame_extractor_no_save(synthetic_video: Path):
    extractor = FrameExtractor(target_fps=30)
    result = extractor.extract(
        synthetic_video,
        save_frames=False,
        save_manifest=False,
    )
    assert len(result.frames) == 15
    assert result.frames[0].file_name is None
