"""Extract, resize, and save frames from exercise videos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.data.video_loader import VideoLoader, VideoMetadata
from backend.utils.video_math import compute_sample_stride


@dataclass(frozen=True)
class ExtractedFrame:
    """One frame taken from the source video."""

    source_index: int
    output_index: int
    timestamp_sec: float
    width: int
    height: int
    file_name: str | None = None


@dataclass
class ExtractionResult:
    """Output of a full frame-extraction run."""

    metadata: VideoMetadata
    frames: list[ExtractedFrame]
    output_dir: Path
    target_fps: float
    sample_stride: int
    effective_fps: float

    def to_dict(self) -> dict:
        return {
            "metadata": asdict(self.metadata),
            "target_fps": self.target_fps,
            "sample_stride": self.sample_stride,
            "effective_fps": self.effective_fps,
            "output_dir": str(self.output_dir),
            "frame_count": len(self.frames),
            "frames": [asdict(f) for f in self.frames],
        }

    def save_manifest(self, path: Path | None = None) -> Path:
        manifest_path = path or (self.output_dir / "manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["metadata"]["path"] = str(payload["metadata"]["path"])
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return manifest_path


def resize_frame(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    """Scale down preserving aspect ratio; no upscale."""
    height, width = frame.shape[:2]
    if width <= max_width and height <= max_height:
        return frame
    scale = min(max_width / width, max_height / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class FrameExtractor:
    """Sample frames at target FPS, resize, and optionally persist to disk."""

    def __init__(
        self,
        target_fps: float = 30.0,
        max_width: int = 1280,
        max_height: int = 720,
        output_dir: str | Path | None = None,
        image_format: str = "jpg",
    ) -> None:
        self.target_fps = target_fps
        self.max_width = max_width
        self.max_height = max_height
        self.output_dir = Path(output_dir) if output_dir else None
        self.image_format = image_format.lstrip(".")

    def extract(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        save_frames: bool = True,
        save_manifest: bool = True,
        max_frames: int | None = None,
    ) -> ExtractionResult:
        """
        Extract frames from video_path.

        Frames are sampled every `sample_stride` source frames when source FPS
        exceeds target_fps. All frames are kept when source FPS is lower.
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        if save_frames and out_dir is None:
            raise ValueError("output_dir is required when save_frames=True")

        loader = VideoLoader(video_path)
        metadata = loader.get_metadata()
        stride = compute_sample_stride(metadata.fps, self.target_fps)
        effective_fps = metadata.fps / stride

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        extracted: list[ExtractedFrame] = []
        output_index = 0

        with loader:
            cap = loader.open()
            source_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if source_index % stride != 0:
                    source_index += 1
                    continue

                resized = resize_frame(frame, self.max_width, self.max_height)
                h, w = resized.shape[:2]
                timestamp = source_index / metadata.fps
                file_name: str | None = None

                if save_frames and out_dir is not None:
                    file_name = f"frame_{output_index:06d}.{self.image_format}"
                    cv2.imwrite(str(out_dir / file_name), resized)

                extracted.append(
                    ExtractedFrame(
                        source_index=source_index,
                        output_index=output_index,
                        timestamp_sec=round(timestamp, 4),
                        width=w,
                        height=h,
                        file_name=file_name,
                    )
                )
                output_index += 1
                source_index += 1

                if max_frames is not None and output_index >= max_frames:
                    break

        result = ExtractionResult(
            metadata=metadata,
            frames=extracted,
            output_dir=out_dir or Path("."),
            target_fps=self.target_fps,
            sample_stride=stride,
            effective_fps=round(effective_fps, 4),
        )
        if save_manifest and out_dir is not None:
            result.save_manifest()
        return result
