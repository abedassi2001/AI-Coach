"""Helpers for displaying videos in browser-based UIs (Streamlit, etc.)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


def mime_type_for_path(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    return MIME_BY_SUFFIX.get(suffix, "video/mp4")


def read_video_bytes(path: Path | str) -> bytes:
    return Path(path).read_bytes()


def prepare_video_for_browser(
    source: Path | bytes,
    *,
    filename: str = "video.mp4",
) -> tuple[bytes, str]:
    """
    Return (bytes, mime_type) suitable for st.video().

    Uploaded user videos are returned as-is. OpenCV-written MP4s (mp4v codec)
    are transcoded to H.264 when ffmpeg is available.
    """
    if isinstance(source, bytes):
        return source, mime_type_for_path(filename)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    playable = ensure_browser_playable(path)
    return playable.read_bytes(), mime_type_for_path(playable)


def ensure_browser_playable(path: Path) -> Path:
    """
    Re-encode OpenCV mp4v output to H.264 for HTML5 players.

    Returns the original path if ffmpeg is unavailable or transcode fails.
    """
    path = Path(path)
    if not path.exists():
        return path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return path

    web_path = path.with_name(f"{path.stem}_web{path.suffix}")
    if web_path.exists() and web_path.stat().st_mtime >= path.stat().st_mtime:
        return web_path

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(web_path),
            ],
            check=True,
            capture_output=True,
        )
        return web_path if web_path.exists() else path
    except (subprocess.CalledProcessError, OSError):
        return path


def write_browser_playable_video(
    writer_release_path: Path,
    *,
    delete_source_on_success: bool = False,
) -> Path:
    """After cv2.VideoWriter.release(), convert output for web playback."""
    playable = ensure_browser_playable(writer_release_path)
    if (
        delete_source_on_success
        and playable != writer_release_path
        and playable.exists()
    ):
        writer_release_path.unlink(missing_ok=True)
    return playable
