"""
AttentionX – Configuration Settings
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─── Paths ────────────────────────────────────────────────────────────────
    BASE_DIR: str = str(Path(__file__).resolve().parent.parent)
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent / "storage" / "uploads")
    OUTPUT_DIR: str = str(Path(__file__).resolve().parent.parent / "storage" / "outputs")
    TEMP_DIR: str = str(Path(__file__).resolve().parent.parent / "storage" / "temp")

    # ─── Whisper ──────────────────────────────────────────────────────────────
    WHISPER_MODEL: str = "base"  # tiny | base | small | medium | large
    WHISPER_LANGUAGE: str = "en"
    WHISPER_DEVICE: str = "cpu"  # cpu | cuda

    # ─── Clip Settings ────────────────────────────────────────────────────────
    MIN_CLIP_DURATION: int = 15       # seconds
    MAX_CLIP_DURATION: int = 90       # seconds
    DEFAULT_CLIP_DURATION: int = 60   # seconds
    MAX_CLIPS_PER_VIDEO: int = 10

    # ─── Video Output ─────────────────────────────────────────────────────────
    OUTPUT_WIDTH: int = 1080          # 9:16 vertical
    OUTPUT_HEIGHT: int = 1920
    OUTPUT_FPS: int = 30
    OUTPUT_BITRATE: str = "5000k"
    OUTPUT_FORMAT: str = "mp4"

    # ─── Subtitle Settings ────────────────────────────────────────────────────
    SUBTITLE_FONT: str = "Arial-Bold"
    SUBTITLE_FONT_SIZE: int = 60
    SUBTITLE_COLOR: str = "white"
    SUBTITLE_STROKE_COLOR: str = "black"
    SUBTITLE_STROKE_WIDTH: int = 3
    SUBTITLE_POSITION: str = "bottom"  # top | center | bottom
    SUBTITLE_MAX_WORDS: int = 5        # words per subtitle line

    # ─── Emotion Detection ────────────────────────────────────────────────────
    EMOTION_ENERGY_WEIGHT: float = 0.4
    EMOTION_PITCH_WEIGHT: float = 0.3
    EMOTION_NLP_WEIGHT: float = 0.3
    EMOTION_PEAK_THRESHOLD: float = 0.65
    EMOTION_MIN_GAP_SECONDS: int = 30  # minimum gap between detected peaks

    # ─── Face Tracking ────────────────────────────────────────────────────────
    FACE_DETECTION_CONFIDENCE: float = 0.7
    FACE_TRACKING_SMOOTH_FACTOR: float = 0.15  # lower = smoother camera movement
    FACE_PADDING_PERCENT: float = 0.20         # padding around face

    # ─── Hook Generation ──────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    HOOK_MODEL: str = "claude-sonnet-4-20250514"
    MAX_HOOKS_PER_CLIP: int = 3

    # ─── App ──────────────────────────────────────────────────────────────────
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    MAX_UPLOAD_SIZE_MB: int = 2048  # 2 GB

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
