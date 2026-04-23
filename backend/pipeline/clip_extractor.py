"""
AttentionX – Step 4: Clip Extraction
Extracts video clips using MoviePy based on detected peaks.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional


class ClipExtractor:
    """
    Extracts raw video clips from the source video.
    Uses FFmpeg directly for speed (avoids MoviePy re-encoding overhead).
    """

    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_clip(
        self,
        start: float,
        end: float,
        clip_index: int,
        reencode: bool = False,
    ) -> str:
        """
        Extract a clip from start to end seconds.

        Args:
            start: Start time in seconds
            end: End time in seconds
            clip_index: Index for filename
            reencode: If True, re-encode (better for further processing).
                      If False, use stream copy (faster, less quality loss).

        Returns:
            Path to extracted clip
        """
        duration = end - start
        out_path = str(Path(self.output_dir) / f"raw_clip_{clip_index:02d}.mp4")

        if reencode:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", self.video_path,
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                out_path,
            ]
        else:
            # Stream copy (fast but may have slight inaccuracy at cut points)
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", self.video_path,
                "-t", str(duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                out_path,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            # Fallback: try with re-encode
            if not reencode:
                return self.extract_clip(start, end, clip_index, reencode=True)
            raise RuntimeError(f"Clip extraction failed: {result.stderr}")

        size_mb = os.path.getsize(out_path) / (1024 * 1024) if os.path.exists(out_path) else 0
        print(f"[ClipExtractor] Clip {clip_index}: {start:.1f}s-{end:.1f}s → {size_mb:.1f}MB")
        return out_path

    def extract_all_clips(
        self,
        peaks: list[dict],
        reencode: bool = True,
    ) -> list[dict]:
        """
        Extract all clips from detected peaks.

        Returns peaks with added 'raw_clip_path' key.
        """
        clips = []
        for i, peak in enumerate(peaks):
            start = peak["clip_start"]
            end = peak["clip_end"]

            try:
                clip_path = self.extract_clip(start, end, i, reencode=reencode)
                clips.append({
                    **peak,
                    "raw_clip_path": clip_path,
                    "clip_index": i,
                    "clip_duration": round(end - start, 2),
                })
            except Exception as e:
                print(f"[ClipExtractor] Failed to extract clip {i}: {e}")
                continue

        print(f"[ClipExtractor] Extracted {len(clips)}/{len(peaks)} clips")
        return clips

    def get_video_dimensions(self) -> tuple[int, int]:
        """Get source video width and height."""
        import subprocess, json
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", self.video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                return int(s.get("width", 1920)), int(s.get("height", 1080))
        return 1920, 1080
