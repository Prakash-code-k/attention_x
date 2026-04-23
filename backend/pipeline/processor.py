"""
AttentionX – Master Pipeline Processor
Orchestrates all steps: extract → transcribe → detect → crop → subtitle → hook
"""

import os
import json
import shutil
from pathlib import Path
from typing import Callable, Optional

from pipeline.audio_extractor import AudioExtractor
from pipeline.transcriber import Transcriber
from pipeline.emotion_detector import EmotionDetector
from pipeline.clip_extractor import ClipExtractor
from pipeline.face_tracker import FaceTracker
from pipeline.subtitle_generator import SubtitleGenerator
from pipeline.hook_generator import HookGenerator
from utils.config import settings
from utils.helpers import ensure_dir, safe_filename, format_duration, get_video_info


class ContentProcessor:
    """
    Full end-to-end content repurposing pipeline.

    Input: Long-form video file (16:9)
    Output: N vertical (9:16) clips with:
        - Burned-in subtitles
        - Viral hook overlays
        - Metadata (title, caption, tags, SRT file)
    """

    def __init__(
        self,
        job_id: str,
        video_path: str,
        output_dir: str,
        temp_dir: str,
        config: dict,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        self.job_id = job_id
        self.video_path = video_path
        self.output_dir = ensure_dir(output_dir)
        self.temp_dir = ensure_dir(temp_dir)
        self.config = config
        self.progress = progress_callback or (lambda p, s: None)

        self.max_clips = config.get("max_clips", 5)
        self.clip_duration = config.get("clip_duration", 60)
        self.generate_hooks = config.get("generate_hooks", True)

    def run(self) -> list[dict]:
        """
        Execute the full pipeline.
        Returns list of processed clip metadata.
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline] Starting job: {self.job_id[:8]}")
        print(f"[Pipeline] Video: {self.video_path}")
        print(f"[Pipeline] Config: {self.config}")
        print(f"{'='*60}\n")

        # ─── Step 1: Extract Audio ─────────────────────────────────────────────
        self.progress(10, "Extracting audio")
        audio_path = self._step_extract_audio()

        # ─── Step 2: Transcribe ────────────────────────────────────────────────
        self.progress(20, "Transcribing with Whisper AI")
        transcript = self._step_transcribe(audio_path)

        # ─── Step 3: Detect Emotional Peaks ───────────────────────────────────
        self.progress(35, "Detecting high-impact moments")
        peaks = self._step_detect_peaks(audio_path, transcript)

        if not peaks:
            print("[Pipeline] No peaks detected — using uniform sampling")
            peaks = self._fallback_uniform_peaks(transcript)

        # ─── Step 4: Extract Raw Clips ─────────────────────────────────────────
        self.progress(50, "Extracting viral clips")
        clips = self._step_extract_clips(peaks)

        if not clips:
            raise RuntimeError("No clips could be extracted from the video")

        # ─── Step 5: Face Tracking + Vertical Crop ────────────────────────────
        self.progress(60, "Converting to vertical (9:16) with face tracking")
        clips = self._step_face_track(clips)

        # ─── Step 6: Generate & Burn Subtitles ────────────────────────────────
        self.progress(75, "Generating and burning subtitles")
        clips = self._step_subtitles(clips, transcript)

        # ─── Step 7: Generate Viral Hooks ─────────────────────────────────────
        if self.generate_hooks:
            self.progress(88, "Generating viral hooks with AI")
            clips = self._step_generate_hooks(clips, transcript)
        else:
            clips = self._add_default_hooks(clips)

        # ─── Step 8: Add Hook Overlays to Videos ──────────────────────────────
        self.progress(93, "Adding hook overlays to videos")
        clips = self._step_add_hook_overlays(clips)

        # ─── Step 9: Export Metadata ───────────────────────────────────────────
        self.progress(97, "Exporting metadata")
        clips = self._step_export_metadata(clips)

        # ─── Cleanup ──────────────────────────────────────────────────────────
        self._cleanup_temp()

        print(f"\n[Pipeline] ✅ Complete! {len(clips)} clips ready.")
        return clips

    # ─── Step Implementations ──────────────────────────────────────────────────

    def _step_extract_audio(self) -> str:
        extractor = AudioExtractor(self.video_path, self.temp_dir)
        return extractor.extract()

    def _step_transcribe(self, audio_path: str) -> dict:
        transcriber = Transcriber(
            model_name=settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            language=settings.WHISPER_LANGUAGE,
            output_dir=self.temp_dir,
        )
        return transcriber.transcribe(audio_path)

    def _step_detect_peaks(self, audio_path: str, transcript: dict) -> list[dict]:
        detector = EmotionDetector()

        # Audio analysis
        audio_analysis = detector.analyze_audio(audio_path)

        # Score transcript segments
        segments = transcript.get("segments", [])
        if segments:
            scored = detector.score_segments_nlp(segments, audio_analysis)
            peaks = detector.detect_peaks(
                scored,
                max_clips=self.max_clips,
                clip_duration=self.clip_duration,
            )
        else:
            # Fallback: audio-only detection
            peaks = detector.detect_peaks_from_audio_only(
                audio_analysis,
                max_clips=self.max_clips,
                clip_duration=self.clip_duration,
            )

        # Enrich peaks with transcript context
        for peak in peaks:
            peak["transcript_segment"] = self._get_transcript_window(
                transcript.get("segments", []),
                peak["clip_start"],
                peak["clip_end"],
            )

        return peaks

    def _step_extract_clips(self, peaks: list[dict]) -> list[dict]:
        extractor = ClipExtractor(
            self.video_path,
            output_dir=str(Path(self.temp_dir) / "raw_clips"),
        )
        return extractor.extract_all_clips(peaks, reencode=True)

    def _step_face_track(self, clips: list[dict]) -> list[dict]:
        tracker = FaceTracker(
            output_width=settings.OUTPUT_WIDTH,
            output_height=settings.OUTPUT_HEIGHT,
        )
        tracked = []
        for clip in clips:
            output_path = str(
                Path(self.temp_dir) / "vertical_clips" /
                f"vertical_{clip['clip_index']:02d}.mp4"
            )
            ensure_dir(str(Path(self.temp_dir) / "vertical_clips"))

            try:
                tracker.process_clip(clip["raw_clip_path"], output_path)
                tracked.append({**clip, "vertical_clip_path": output_path})
            except Exception as e:
                print(f"[Pipeline] Face tracking failed for clip {clip['clip_index']}: {e}")
                tracked.append({**clip, "vertical_clip_path": clip["raw_clip_path"]})

        return tracked

    def _step_subtitles(self, clips: list[dict], transcript: dict) -> list[dict]:
        transcriber = Transcriber(output_dir=self.temp_dir)
        sub_gen = SubtitleGenerator()
        enriched = []

        for clip in clips:
            # Get subtitles for this clip window
            all_subs = transcript.get("subtitles", [])
            clip_subs = transcriber.get_subtitles_for_clip(
                all_subs, clip["clip_start"], clip["clip_end"]
            )

            # Output path for subtitled video
            sub_video_path = str(
                Path(self.temp_dir) / "subtitled_clips" /
                f"subtitled_{clip['clip_index']:02d}.mp4"
            )
            ensure_dir(str(Path(self.temp_dir) / "subtitled_clips"))

            # Burn subtitles
            try:
                sub_gen.burn_subtitles(
                    input_video=clip.get("vertical_clip_path", clip["raw_clip_path"]),
                    output_video=sub_video_path,
                    subtitles=clip_subs,
                    style="tiktok",
                )
                enriched.append({
                    **clip,
                    "subtitled_clip_path": sub_video_path,
                    "subtitles": clip_subs,
                })
            except Exception as e:
                print(f"[Pipeline] Subtitle burning failed: {e}")
                enriched.append({
                    **clip,
                    "subtitled_clip_path": clip.get("vertical_clip_path", clip["raw_clip_path"]),
                    "subtitles": clip_subs,
                })

        return enriched

    def _step_generate_hooks(self, clips: list[dict], transcript: dict) -> list[dict]:
        generator = HookGenerator()
        full_text = transcript.get("text", "")

        for clip in clips:
            clip["full_transcript"] = full_text
            # Add broader context window for hook generation
            if not clip.get("transcript_segment"):
                clip["transcript_segment"] = self._get_transcript_window(
                    transcript.get("segments", []),
                    clip["clip_start"],
                    clip["clip_end"],
                )

        return generator.generate_hooks_batch(clips)

    def _add_default_hooks(self, clips: list[dict]) -> list[dict]:
        generator = HookGenerator(api_key=None)
        for clip in clips:
            hooks_data = generator._template_fallback(
                clip.get("transcript_segment", ""),
                clip.get("keywords", []),
                clip.get("score", 0.5),
            )
            clip.update({
                "hooks": hooks_data["hooks"],
                "title": hooks_data["title"],
                "description": hooks_data["description"],
                "tags": hooks_data["tags"],
            })
        return clips

    def _step_add_hook_overlays(self, clips: list[dict]) -> list[dict]:
        sub_gen = SubtitleGenerator()
        enriched = []

        for clip in clips:
            hooks = clip.get("hooks", [])
            best_hook = hooks[0]["headline"] if hooks else ""

            input_video = clip.get("subtitled_clip_path", clip.get("vertical_clip_path", clip["raw_clip_path"]))

            # Final output filename
            final_filename = f"clip_{clip['clip_index'] + 1:02d}_attentionx.mp4"
            final_path = str(Path(self.output_dir) / final_filename)

            try:
                if best_hook:
                    sub_gen.add_hook_overlay(
                        input_video=input_video,
                        output_video=final_path,
                        hook_text=best_hook,
                        duration=3.0,
                        position="top",
                    )
                else:
                    shutil.copy2(input_video, final_path)

                enriched.append({**clip, "final_clip_path": final_path, "final_filename": final_filename})

            except Exception as e:
                print(f"[Pipeline] Hook overlay failed: {e}")
                shutil.copy2(input_video, final_path)
                enriched.append({**clip, "final_clip_path": final_path, "final_filename": final_filename})

        return enriched

    def _step_export_metadata(self, clips: list[dict]) -> list[dict]:
        """Export SRT files and metadata JSON for each clip."""
        transcriber = Transcriber(output_dir=self.output_dir)
        sub_gen = SubtitleGenerator()
        results = []

        for clip in clips:
            idx = clip["clip_index"]
            clip_subs = clip.get("subtitles", [])
            hooks = clip.get("hooks", [])

            # Export SRT
            srt_filename = f"clip_{idx + 1:02d}_subtitles.srt"
            srt_path = str(Path(self.output_dir) / srt_filename)
            if clip_subs:
                sub_gen.generate_srt_file(clip_subs, srt_path)

            # Build clean metadata dict for API response
            clip_meta = {
                "clip_number": idx + 1,
                "filename": clip.get("final_filename", f"clip_{idx+1:02d}.mp4"),
                "download_url": f"/outputs/{self.job_id}/{clip.get('final_filename')}",
                "srt_filename": srt_filename,
                "srt_url": f"/outputs/{self.job_id}/{srt_filename}",
                "duration_seconds": clip.get("clip_duration", 0),
                "duration_formatted": format_duration(clip.get("clip_duration", 0)),
                "timestamp_start": format_duration(clip["clip_start"]),
                "timestamp_end": format_duration(clip["clip_end"]),
                "score": clip.get("score", 0),
                "peak_text": clip.get("peak_text", "")[:200],
                "keywords": clip.get("keywords", []),
                "hooks": hooks[:3],
                "title": clip.get("title", ""),
                "description": clip.get("description", ""),
                "tags": clip.get("tags", [])[:10],
                "subtitles_count": len(clip_subs),
            }
            results.append(clip_meta)

        # Save full job metadata
        meta_path = str(Path(self.output_dir) / "job_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({
                "job_id": self.job_id,
                "total_clips": len(results),
                "clips": results,
            }, f, indent=2)

        return results

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_transcript_window(
        self,
        segments: list[dict],
        start: float,
        end: float,
        context_buffer: float = 5.0,
    ) -> str:
        """Extract transcript text within a time window."""
        window_segs = [
            seg for seg in segments
            if seg["end"] >= (start - context_buffer) and seg["start"] <= (end + context_buffer)
        ]
        return " ".join(seg.get("text", "") for seg in window_segs).strip()

    def _fallback_uniform_peaks(self, transcript: dict) -> list[dict]:
        """Uniformly sample clip positions if no peaks detected."""
        from utils.helpers import get_video_info
        info = get_video_info(self.video_path)
        duration = info.get("duration", 300)

        gap = duration / (self.max_clips + 1)
        peaks = []
        for i in range(self.max_clips):
            start_t = gap * (i + 1) - self.clip_duration / 2
            start_t = max(0, min(start_t, duration - self.clip_duration))
            peaks.append({
                "clip_start": start_t,
                "clip_end": start_t + self.clip_duration,
                "peak_timestamp": start_t + self.clip_duration / 2,
                "score": 0.5,
                "nlp_score": 0.0,
                "audio_score": 0.5,
                "peak_text": "",
                "keywords": [],
                "transcript_segment": self._get_transcript_window(
                    transcript.get("segments", []), start_t, start_t + self.clip_duration
                ),
            })
        return peaks

    def _cleanup_temp(self):
        """Remove temporary processing files."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
