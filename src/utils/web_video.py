"""Helpers for displaying videos in browser-based UIs (Streamlit, etc.)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}

# HTML5 <video> plays these; OpenCV mp4v / fmp4 often do not.
_BROWSER_SAFE_CODECS = frozenset({"h264", "avc1", "H264", "AVC1"})

_CACHE_DIR = Path(tempfile.gettempdir()) / "iron_form_coach_videos"


def find_ffmpeg() -> str | None:
    """System ffmpeg, or bundled binary from imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def mime_type_for_path(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    return MIME_BY_SUFFIX.get(suffix, "video/mp4")


def read_video_bytes(path: Path | str) -> bytes:
    return Path(path).read_bytes()


def video_codec_fourcc(path: Path | str) -> str:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ""
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))


def needs_browser_transcode(path: Path) -> bool:
    if not path.exists():
        return False
    codec = video_codec_fourcc(path)
    if not codec:
        return True
    return codec not in _BROWSER_SAFE_CODECS


def ensure_browser_playable(path: Path) -> Path:
    """
    Re-encode to H.264/AAC MP4 for HTML5 players.

    Returns the original path if ffmpeg is unavailable or transcode fails.
    """
    path = Path(path)
    if not path.exists():
        return path

    web_path = path.with_name(f"{path.stem}_web{path.suffix}")
    if web_path.exists() and web_path.stat().st_mtime >= path.stat().st_mtime:
        if not needs_browser_transcode(web_path):
            return web_path

    if not needs_browser_transcode(path):
        return path

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return path

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
                "-an",
                str(web_path),
            ],
            check=True,
            capture_output=True,
        )
        if web_path.exists() and web_path.stat().st_size > 0:
            return web_path
    except (subprocess.CalledProcessError, OSError):
        pass
    return path


def materialize_bytes(source: bytes, *, filename: str = "video.mp4") -> Path:
    """Write upload bytes to a stable cache path for st.video(Path)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source).hexdigest()[:16]
    safe_name = Path(filename).name or "video.mp4"
    dest = _CACHE_DIR / f"{digest}_{safe_name}"
    if not dest.exists() or dest.stat().st_size != len(source):
        dest.write_bytes(source)
    return dest


def resolve_playable_path(
    source: Path | bytes,
    *,
    filename: str = "video.mp4",
) -> Path:
    """Return a local file path suitable for ``st.video(str(path))``."""
    if isinstance(source, bytes):
        path = materialize_bytes(source, filename=filename)
    else:
        path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return ensure_browser_playable(path)


def prepare_video_for_browser(
    source: Path | bytes,
    *,
    filename: str = "video.mp4",
) -> tuple[bytes, str]:
    """Return (bytes, mime_type) — prefer ``resolve_playable_path`` + st.video(path)."""
    playable = resolve_playable_path(source, filename=filename)
    return playable.read_bytes(), mime_type_for_path(playable)


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
