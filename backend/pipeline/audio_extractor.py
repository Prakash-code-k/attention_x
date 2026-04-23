"""
AttentionX – Step 1: Audio Extraction
Extracts audio from video using FFmpeg.
"""

import os
import subprocess
from pathlib import Path


class AudioExtractor:
    """Extracts audio track from a video file."""

    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract(self) -> str:
        """
        Extract audio as WAV (16kHz mono) for Whisper and Librosa compatibility.
        Returns path to the extracted audio file.
        """
        audio_path = str(Path(self.output_dir) / "audio.wav")

        cmd = [
            "ffmpeg",
            "-y",                       # overwrite output
            "-i", self.video_path,      # input video
            "-vn",                      # no video
            "-acodec", "pcm_s16le",     # PCM 16-bit little-endian (WAV)
            "-ar", "16000",             # 16 kHz sample rate (optimal for Whisper)
            "-ac", "1",                 # mono channel
            "-af", "loudnorm",          # normalize loudness
            audio_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for long videos
        )

        if result.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not created at {audio_path}")

        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"[AudioExtractor] Extracted audio: {audio_path} ({size_mb:.1f} MB)")
        return audio_path

    def extract_segment(self, start: float, end: float, suffix: str = "") -> str:
        """Extract a specific audio segment (for clip-level processing)."""
        out_name = f"segment_{suffix or f'{int(start)}_{int(end)}'}.wav"
        out_path = str(Path(self.output_dir) / out_name)

        cmd = [
            "ffmpeg", "-y",
            "-i", self.video_path,
            "-ss", str(start),
            "-to", str(end),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            out_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Segment extraction failed: {result.stderr}")

        return out_path
