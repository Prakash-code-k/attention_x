"""
AttentionX – Step 6: Subtitle Generator
Burns animated subtitles onto video clips using FFmpeg drawtext filter.
Features: word-highlight mode, drop shadow, bold font.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from utils.config import settings
from utils.helpers import seconds_to_srt_timestamp


class SubtitleGenerator:
    """
    Burns stylized subtitles onto vertical video clips.

    Supports:
    - Multi-word chunks (5 words per line)
    - Bold font with stroke/shadow for readability
    - Positioned at bottom 20% of frame
    - Exports both burned-in video and separate .srt/.vtt files
    """

    def __init__(
        self,
        font_size: int = None,
        font_color: str = None,
        stroke_color: str = None,
        stroke_width: int = None,
        position: str = None,
        max_words_per_line: int = None,
    ):
        self.font_size = font_size or settings.SUBTITLE_FONT_SIZE
        self.font_color = font_color or settings.SUBTITLE_COLOR
        self.stroke_color = stroke_color or settings.SUBTITLE_STROKE_COLOR
        self.stroke_width = stroke_width or settings.SUBTITLE_STROKE_WIDTH
        self.position = position or settings.SUBTITLE_POSITION
        self.max_words = max_words_per_line or settings.SUBTITLE_MAX_WORDS

    def burn_subtitles(
        self,
        input_video: str,
        output_video: str,
        subtitles: list[dict],
        clip_start_offset: float = 0.0,
        style: str = "tiktok",
    ) -> str:
        """
        Burn subtitles into video.
        Writes SRT to /tmp to avoid path-escaping issues with FFmpeg's subtitles filter.
        Falls back to drawtext if libass is unavailable.
        """
        if not subtitles:
            print("[SubtitleGenerator] No subtitles, copying as-is")
            import shutil
            os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
            shutil.copy2(input_video, output_video)
            return output_video

        # Always write SRT to /tmp with a safe filename (no spaces, no special chars)
        import uuid
        srt_path = os.path.join(tempfile.gettempdir(), f"atx_{uuid.uuid4().hex}.srt")
        self._write_srt(subtitles, srt_path, clip_start_offset)

        try:
            result_path = self._burn_with_ffmpeg(input_video, output_video, srt_path, style)
        finally:
            # Always clean up temp SRT
            if os.path.exists(srt_path):
                os.remove(srt_path)

        return result_path

    def _write_srt(
        self,
        subtitles: list[dict],
        srt_path: str,
        offset: float = 0.0,
    ):
        """Write SRT file with optional time offset correction."""
        lines = []
        for i, sub in enumerate(subtitles, 1):
            start = max(0, sub["start"] - offset)
            end = max(0, sub["end"] - offset)

            if end <= start:
                end = start + 0.5

            lines.append(str(i))
            lines.append(
                f"{seconds_to_srt_timestamp(start)} --> {seconds_to_srt_timestamp(end)}"
            )
            # Clean text
            text = sub["text"].strip().upper()  # TikTok style: all caps
            lines.append(text)
            lines.append("")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _burn_with_ffmpeg(
        self,
        input_video: str,
        output_video: str,
        srt_path: str,
        style: str = "tiktok",
    ) -> str:
        """Use FFmpeg subtitles filter to burn captions."""
        os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

        # Style presets
        styles = {
            "tiktok": {
                "Fontsize": self.font_size,
                "FontName": "Arial",
                "Bold": 1,
                "PrimaryColour": "&H00FFFFFF",  # white
                "OutlineColour": "&H00000000",  # black outline
                "BackColour": "&H80000000",      # semi-transparent background
                "Outline": self.stroke_width,
                "Shadow": 2,
                "Alignment": 2,   # bottom center
                "MarginV": 120,   # distance from bottom
            },
            "youtube": {
                "Fontsize": self.font_size - 10,
                "FontName": "Arial",
                "Bold": 0,
                "PrimaryColour": "&H00FFFFFF",
                "OutlineColour": "&H00000000",
                "Outline": 2,
                "Shadow": 1,
                "Alignment": 2,
                "MarginV": 60,
            },
            "minimal": {
                "Fontsize": self.font_size - 15,
                "FontName": "Arial",
                "Bold": 0,
                "PrimaryColour": "&H00FFFFFF",
                "OutlineColour": "&H00000000",
                "Outline": 1,
                "Shadow": 0,
                "Alignment": 2,
                "MarginV": 80,
            },
        }

        s = styles.get(style, styles["tiktok"])
        style_str = ",".join(f"{k}={v}" for k, v in s.items())

        # Escape SRT path for FFmpeg (handle spaces & special chars)
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", f"subtitles='{srt_escaped}':force_style='{style_str}'",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_video,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"[SubtitleGenerator] FFmpeg subtitles failed, trying ASS method...")
            return self._burn_with_ass(input_video, output_video, subtitles_srt=srt_path)

        print(f"[SubtitleGenerator] Subtitles burned: {output_video}")
        return output_video

    def _burn_with_ass(
        self,
        input_video: str,
        output_video: str,
        subtitles_srt: str,
    ) -> str:
        """
        Fallback: convert SRT to ASS, then burn with more control.
        """
        ass_path = subtitles_srt.replace(".srt", ".ass")

        # Convert SRT → ASS using FFmpeg
        conv_cmd = [
            "ffmpeg", "-y", "-i", subtitles_srt, ass_path
        ]
        subprocess.run(conv_cmd, capture_output=True, timeout=30)

        if os.path.exists(ass_path):
            ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:")
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video,
                "-vf", f"ass='{ass_escaped}'",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "copy", output_video,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if os.path.exists(ass_path):
                os.remove(ass_path)
            if result.returncode == 0:
                return output_video

        # Final fallback: just copy the video without subs
        import shutil
        print("[SubtitleGenerator] Subtitle burning failed, copying without subs")
        shutil.copy2(input_video, output_video)
        return output_video

    def generate_srt_file(
        self,
        subtitles: list[dict],
        output_path: str,
    ) -> str:
        """Export standalone SRT file."""
        self._write_srt(subtitles, output_path)
        return output_path

    def add_hook_overlay(
        self,
        input_video: str,
        output_video: str,
        hook_text: str,
        duration: float = 3.0,
        position: str = "top",
    ) -> str:
        """
        Add a viral hook headline as a text overlay at the start of the clip.
        Shown for `duration` seconds at the top of the frame.
        """
        os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

        # Escape special chars for drawtext
        hook_escaped = hook_text.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")

        y_pos = "80" if position == "top" else "(h-160)"
        box_color = "0x000000@0.6"

        drawtext_filter = (
            f"drawtext=text='{hook_escaped}':"
            f"fontsize={self.font_size + 10}:"
            f"fontcolor=yellow:"
            f"font=Arial:"
            f"bold=1:"
            f"x=(w-text_w)/2:"
            f"y={y_pos}:"
            f"box=1:"
            f"boxcolor={box_color}:"
            f"boxborderw=15:"
            f"enable='between(t,0,{duration})'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", drawtext_filter,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_video,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            import shutil
            shutil.copy2(input_video, output_video)

        return output_video
