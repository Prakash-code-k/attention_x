"""
AttentionX – Helper Utilities
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def get_video_info(video_path: str) -> dict:
    """Extract video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)

        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
        )
        audio_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {}
        )
        fmt = data.get("format", {})

        duration = float(fmt.get("duration", 0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Parse FPS
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            num, den = fps_str.split("/")
            fps = round(int(num) / int(den), 2)
        except Exception:
            fps = 30.0

        return {
            "duration": round(duration, 2),
            "duration_formatted": format_duration(duration),
            "width": width,
            "height": height,
            "fps": fps,
            "aspect_ratio": f"{width}:{height}",
            "is_vertical": height > width,
            "codec": video_stream.get("codec_name", "unknown"),
            "audio_codec": audio_stream.get("codec_name", "none"),
            "size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 2),
            "bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000, 2),
        }
    except Exception as e:
        print(f"[Warning] Could not parse video info: {e}")
        return {"duration": 0, "width": 0, "height": 0, "fps": 30, "size_mb": 0}


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format: HH:MM:SS.mmm"""
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def cleanup_temp_files(job_id: str, upload_dir: str, output_dir: str, temp_dir: str):
    """Remove all files associated with a job."""
    # Remove uploaded video
    for f in Path(upload_dir).glob(f"{job_id}_*"):
        try:
            f.unlink()
        except Exception:
            pass

    # Remove temp directory
    temp_job_dir = Path(temp_dir) / job_id
    if temp_job_dir.exists():
        shutil.rmtree(str(temp_job_dir), ignore_errors=True)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """Sanitize a filename."""
    import re
    name = re.sub(r"[^\w\s\-.]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:128]


def chunk_text(text: str, max_words: int = 5) -> list[dict]:
    """
    Split transcript text into subtitle chunks of max N words.
    Returns list of dicts with 'text' key.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append({"text": chunk})
    return chunks


def interpolate_timestamps(words_data: list[dict], start: float, end: float) -> list[dict]:
    """
    Distribute timestamps proportionally across words if individual timestamps unavailable.
    words_data: list of {'text': str} dicts
    """
    if not words_data:
        return []

    total_duration = end - start
    interval = total_duration / len(words_data)

    result = []
    for i, word in enumerate(words_data):
        word_start = start + i * interval
        word_end = start + (i + 1) * interval
        result.append({
            "text": word["text"],
            "start": round(word_start, 3),
            "end": round(word_end, 3),
        })
    return result


def merge_subtitle_segments(segments: list[dict], max_words: int = 5) -> list[dict]:
    """
    Merge Whisper word-level segments into subtitle blocks of N words.
    Each segment: {'text': str, 'start': float, 'end': float}
    """
    if not segments:
        return []

    merged = []
    current_words = []
    current_start = segments[0]["start"]

    for seg in segments:
        words = seg["text"].strip().split()
        for word in words:
            current_words.append(word)
            if len(current_words) >= max_words:
                merged.append({
                    "text": " ".join(current_words),
                    "start": current_start,
                    "end": seg["end"],
                })
                current_words = []
                current_start = seg["end"]

    # Flush remaining words
    if current_words and segments:
        merged.append({
            "text": " ".join(current_words),
            "start": current_start,
            "end": segments[-1]["end"],
        })

    return merged
