"""
AttentionX – Step 2: Transcription
Uses OpenAI Whisper for speech-to-text with word-level timestamps.
"""

import os
import json
from pathlib import Path
from typing import Optional
import whisper

from utils.helpers import merge_subtitle_segments


class Transcriber:
    """
    Transcribes audio using OpenAI Whisper.
    Produces word-level timestamps for precise subtitle rendering.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        language: Optional[str] = "en",
        output_dir: str = "/tmp",
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.output_dir = output_dir
        self._model = None
        os.makedirs(output_dir, exist_ok=True)

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            print(f"[Transcriber] Loading Whisper model: {self.model_name} on {self.device}")
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe audio file.

        Returns:
            {
                "text": str,               # full transcript
                "segments": [...],         # sentence-level segments with timestamps
                "words": [...],            # word-level timestamps (if available)
                "language": str,           # detected language
                "subtitles": [...],        # merged subtitle blocks (N words each)
            }
        """
        model = self._load_model()
        print(f"[Transcriber] Transcribing: {audio_path}")

        # Run Whisper with word timestamps
        result = model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            verbose=False,
            condition_on_previous_text=True,
            no_speech_threshold=0.5,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        # Extract word-level data
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", "").strip(),
                    "start": round(w.get("start", 0), 3),
                    "end": round(w.get("end", 0), 3),
                    "probability": round(w.get("probability", 0), 3),
                })

        # Build segment-level data
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "id": seg.get("id"),
                "text": seg.get("text", "").strip(),
                "start": round(seg.get("start", 0), 3),
                "end": round(seg.get("end", 0), 3),
                "avg_logprob": round(seg.get("avg_logprob", 0), 4),
                "no_speech_prob": round(seg.get("no_speech_prob", 0), 4),
            })

        # Merge into subtitle blocks (5 words each)
        subtitles = merge_subtitle_segments(segments, max_words=5)

        transcript_data = {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "segments": segments,
            "words": words,
            "subtitles": subtitles,
        }

        # Save transcript JSON
        json_path = str(Path(self.output_dir) / "transcript.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        print(f"[Transcriber] Transcript saved: {len(segments)} segments, {len(words)} words")
        return transcript_data

    def transcribe_segment(self, audio_path: str, offset: float = 0.0) -> dict:
        """
        Transcribe a short audio segment (e.g., a clip).
        offset: time offset to add to all timestamps (in seconds)
        """
        result = self.transcribe(audio_path)

        # Apply offset to all timestamps
        if offset > 0:
            for seg in result["segments"]:
                seg["start"] = round(seg["start"] + offset, 3)
                seg["end"] = round(seg["end"] + offset, 3)
            for w in result["words"]:
                w["start"] = round(w["start"] + offset, 3)
                w["end"] = round(w["end"] + offset, 3)
            for sub in result["subtitles"]:
                sub["start"] = round(sub["start"] + offset, 3)
                sub["end"] = round(sub["end"] + offset, 3)

        return result

    def get_subtitles_for_clip(
        self,
        full_subtitles: list[dict],
        clip_start: float,
        clip_end: float
    ) -> list[dict]:
        """
        Filter and re-timestamp subtitle blocks for a specific clip window.
        """
        clip_subs = []
        for sub in full_subtitles:
            sub_start = sub["start"]
            sub_end = sub["end"]

            # Check overlap with clip window
            if sub_end < clip_start or sub_start > clip_end:
                continue

            # Re-timestamp relative to clip start
            clip_subs.append({
                "text": sub["text"],
                "start": round(max(0, sub_start - clip_start), 3),
                "end": round(min(clip_end - clip_start, sub_end - clip_start), 3),
            })

        return clip_subs

    def export_srt(self, subtitles: list[dict], output_path: str) -> str:
        """Export subtitles as SRT file."""
        from utils.helpers import seconds_to_srt_timestamp

        lines = []
        for i, sub in enumerate(subtitles, 1):
            lines.append(str(i))
            lines.append(
                f"{seconds_to_srt_timestamp(sub['start'])} --> {seconds_to_srt_timestamp(sub['end'])}"
            )
            lines.append(sub["text"])
            lines.append("")

        srt_content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        return output_path

    def export_vtt(self, subtitles: list[dict], output_path: str) -> str:
        """Export subtitles as WebVTT file."""
        from utils.helpers import seconds_to_vtt_timestamp

        lines = ["WEBVTT", ""]
        for i, sub in enumerate(subtitles, 1):
            lines.append(f"cue-{i}")
            lines.append(
                f"{seconds_to_vtt_timestamp(sub['start'])} --> {seconds_to_vtt_timestamp(sub['end'])}"
            )
            lines.append(sub["text"])
            lines.append("")

        vtt_content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(vtt_content)

        return output_path
