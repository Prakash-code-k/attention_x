"""
AttentionX – Test Suite
Run with: python -m pytest tests/ -v
"""

import os
import sys
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_segments():
    """Sample transcript segments for testing."""
    return [
        {"id": 0, "text": "Today we're going to talk about something incredible.",
         "start": 0.0, "end": 5.0, "avg_logprob": -0.2},
        {"id": 1, "text": "This is the secret nobody wants you to know.",
         "start": 5.0, "end": 10.0, "avg_logprob": -0.15},
        {"id": 2, "text": "The results were shocking and unbelievable.",
         "start": 10.0, "end": 15.0, "avg_logprob": -0.18},
        {"id": 3, "text": "Let me explain the basic concept here.",
         "start": 15.0, "end": 20.0, "avg_logprob": -0.3},
        {"id": 4, "text": "Warning: this will change everything you thought you knew.",
         "start": 20.0, "end": 25.0, "avg_logprob": -0.12},
    ]


@pytest.fixture
def sample_audio_analysis():
    """Simulated audio analysis output."""
    n = 100
    times = [i * 0.5 for i in range(n)]
    # Simulate two peaks
    energy = [0.3] * n
    energy[20] = 0.9
    energy[21] = 0.85
    energy[60] = 0.88
    energy[61] = 0.82

    pitch = [0.4] * n
    pitch[20] = 0.8
    pitch[60] = 0.75

    return {
        "times": times,
        "energy_score": energy,
        "pitch_score": pitch,
        "flux_score": [0.4] * n,
        "duration": 50.0,
        "sr": 16000,
    }


# ─── Utils Tests ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_format_duration_minutes(self):
        from utils.helpers import format_duration
        assert format_duration(90) == "01:30"
        assert format_duration(3661) == "01:01:01"
        assert format_duration(0) == "00:00"

    def test_seconds_to_srt_timestamp(self):
        from utils.helpers import seconds_to_srt_timestamp
        result = seconds_to_srt_timestamp(1.5)
        assert result == "00:00:01,500"

        result = seconds_to_srt_timestamp(3661.25)
        assert result == "01:01:01,250"

    def test_seconds_to_vtt_timestamp(self):
        from utils.helpers import seconds_to_vtt_timestamp
        result = seconds_to_vtt_timestamp(1.5)
        assert result == "00:00:01.500"

    def test_merge_subtitle_segments_basic(self):
        from utils.helpers import merge_subtitle_segments
        segments = [
            {"text": "Hello world this is a test sentence here now", "start": 0.0, "end": 5.0}
        ]
        result = merge_subtitle_segments(segments, max_words=5)
        assert len(result) > 0
        for sub in result:
            words = sub["text"].split()
            assert len(words) <= 5

    def test_merge_subtitle_segments_empty(self):
        from utils.helpers import merge_subtitle_segments
        result = merge_subtitle_segments([], max_words=5)
        assert result == []

    def test_chunk_text(self):
        from utils.helpers import chunk_text
        text = "one two three four five six seven eight"
        chunks = chunk_text(text, max_words=3)
        assert len(chunks) == 3  # 3+3+2
        assert chunks[0]["text"] == "one two three"


# ─── Emotion Detector Tests ───────────────────────────────────────────────────

class TestEmotionDetector:
    def test_normalize(self):
        from pipeline.emotion_detector import EmotionDetector
        detector = EmotionDetector()
        arr = np.array([0.0, 0.5, 1.0, 2.0])
        normalized = detector._normalize(arr)
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_normalize_flat_array(self):
        from pipeline.emotion_detector import EmotionDetector
        detector = EmotionDetector()
        arr = np.array([0.5, 0.5, 0.5])
        normalized = detector._normalize(arr)
        assert np.all(normalized == 0.0)

    def test_resample_signal(self):
        from pipeline.emotion_detector import EmotionDetector
        arr = np.array([0.0, 0.5, 1.0])
        resampled = EmotionDetector._resample_signal(arr, 6)
        assert len(resampled) == 6
        assert resampled[0] == pytest.approx(0.0)
        assert resampled[-1] == pytest.approx(1.0)

    def test_nlp_scoring_keywords(self, sample_segments, sample_audio_analysis):
        from pipeline.emotion_detector import EmotionDetector
        detector = EmotionDetector()
        scored = detector.score_segments_nlp(sample_segments, sample_audio_analysis)

        # Should be sorted by combined score descending
        scores = [s["combined_score"] for s in scored]
        assert scores == sorted(scores, reverse=True)

        # "secret" and "warning" and "shocking" should have high nlp scores
        high_nlp = [s for s in scored if s["nlp_score"] > 0.5]
        assert len(high_nlp) >= 2

    def test_detect_peaks_basic(self, sample_segments, sample_audio_analysis):
        from pipeline.emotion_detector import EmotionDetector
        detector = EmotionDetector(peak_threshold=0.3, min_gap_seconds=5)
        scored = detector.score_segments_nlp(sample_segments, sample_audio_analysis)
        peaks = detector.detect_peaks(scored, max_clips=3, clip_duration=15)
        assert isinstance(peaks, list)
        assert len(peaks) <= 3

        # Peaks should be sorted by clip_start
        starts = [p["clip_start"] for p in peaks]
        assert starts == sorted(starts)

    def test_detect_peaks_no_overlap(self, sample_segments, sample_audio_analysis):
        from pipeline.emotion_detector import EmotionDetector
        detector = EmotionDetector(peak_threshold=0.1, min_gap_seconds=10)
        scored = detector.score_segments_nlp(sample_segments, sample_audio_analysis)
        peaks = detector.detect_peaks(scored, max_clips=10, clip_duration=15)

        # Verify no overlapping clips
        for i in range(len(peaks) - 1):
            assert peaks[i]["clip_end"] <= peaks[i + 1]["clip_start"]


# ─── Transcriber Tests ────────────────────────────────────────────────────────

class TestTranscriber:
    def test_get_subtitles_for_clip(self):
        from pipeline.transcriber import Transcriber
        t = Transcriber(output_dir="/tmp")

        all_subs = [
            {"text": "Hello world", "start": 5.0, "end": 7.0},
            {"text": "This is great", "start": 10.0, "end": 12.0},
            {"text": "Outside window", "start": 50.0, "end": 52.0},
        ]

        clip_subs = t.get_subtitles_for_clip(all_subs, clip_start=4.0, clip_end=15.0)

        assert len(clip_subs) == 2  # "Outside window" excluded
        # All timestamps should be relative to clip start
        for sub in clip_subs:
            assert sub["start"] >= 0

    def test_export_srt(self, tmp_path):
        from pipeline.transcriber import Transcriber
        t = Transcriber(output_dir=str(tmp_path))

        subs = [
            {"text": "Hello world", "start": 0.0, "end": 2.0},
            {"text": "How are you", "start": 2.5, "end": 4.5},
        ]

        srt_path = str(tmp_path / "test.srt")
        result = t.export_srt(subs, srt_path)

        assert os.path.exists(result)
        content = open(result).read()
        assert "HELLO WORLD" in content.upper() or "Hello world" in content
        assert "00:00:00,000" in content
        assert "-->" in content


# ─── Hook Generator Tests ─────────────────────────────────────────────────────

class TestHookGenerator:
    def test_template_fallback_basic(self):
        from pipeline.hook_generator import HookGenerator
        gen = HookGenerator(api_key=None)
        result = gen._template_fallback(
            transcript="This is an amazing discovery about science.",
            keywords=["amazing", "discovery"],
            score=0.8,
        )
        assert "hooks" in result
        assert len(result["hooks"]) == 3
        assert "title" in result
        assert "tags" in result

        for hook in result["hooks"]:
            assert "headline" in hook
            assert "caption" in hook
            assert "platform" in hook
            assert hook["platform"] in ["tiktok", "instagram", "youtube"]

    def test_template_fallback_empty(self):
        from pipeline.hook_generator import HookGenerator
        gen = HookGenerator(api_key=None)
        result = gen._template_fallback("", [], 0.3)
        assert len(result["hooks"]) == 3

    def test_hooks_without_api_key(self):
        from pipeline.hook_generator import HookGenerator
        gen = HookGenerator(api_key=None)
        # Should fallback gracefully
        result = gen.generate_hooks(
            transcript="Great content here",
            keywords=["great"],
            clip_duration=60,
            emotional_score=0.7,
        )
        assert "hooks" in result
        assert len(result["hooks"]) > 0


# ─── Config Tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_config_loads(self):
        from utils.config import settings
        assert settings.OUTPUT_WIDTH == 1080
        assert settings.OUTPUT_HEIGHT == 1920
        assert settings.OUTPUT_WIDTH / settings.OUTPUT_HEIGHT == pytest.approx(9/16, abs=0.01)

    def test_valid_paths(self):
        from utils.config import settings
        assert settings.UPLOAD_DIR
        assert settings.OUTPUT_DIR
        assert settings.TEMP_DIR

    def test_weights_sum(self):
        from utils.config import settings
        total = (settings.EMOTION_ENERGY_WEIGHT +
                 settings.EMOTION_PITCH_WEIGHT +
                 settings.EMOTION_NLP_WEIGHT)
        assert total == pytest.approx(1.0, abs=0.01)


# ─── Integration Smoke Test ───────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_mock(self, tmp_path):
        """
        Smoke test the pipeline with mocked external calls.
        Tests orchestration logic without actual video/AI processing.
        """
        from pipeline.processor import ContentProcessor

        with patch("pipeline.processor.AudioExtractor") as MockAudio, \
             patch("pipeline.processor.Transcriber") as MockTranscriber, \
             patch("pipeline.processor.EmotionDetector") as MockDetector, \
             patch("pipeline.processor.ClipExtractor") as MockExtractor, \
             patch("pipeline.processor.FaceTracker") as MockTracker, \
             patch("pipeline.processor.SubtitleGenerator") as MockSubGen, \
             patch("pipeline.processor.HookGenerator") as MockHookGen, \
             patch("pipeline.processor.get_video_info") as MockInfo:

            # Setup mocks
            MockInfo.return_value = {"duration": 120, "duration_formatted": "02:00"}
            MockAudio.return_value.extract.return_value = str(tmp_path / "audio.wav")

            MockTranscriber.return_value.transcribe.return_value = {
                "text": "Test transcript",
                "language": "en",
                "segments": [{"id": 0, "text": "Test", "start": 0.0, "end": 5.0,
                              "avg_logprob": -0.2, "no_speech_prob": 0.1}],
                "words": [],
                "subtitles": [{"text": "Test", "start": 0.0, "end": 5.0}],
            }
            MockTranscriber.return_value.get_subtitles_for_clip.return_value = []

            MockDetector.return_value.analyze_audio.return_value = {
                "times": [0.0, 1.0], "energy_score": [0.5, 0.8],
                "pitch_score": [0.4, 0.7], "flux_score": [0.3, 0.6],
                "duration": 120, "sr": 16000,
            }
            MockDetector.return_value.score_segments_nlp.return_value = [
                {"id": 0, "text": "Test", "start": 0.0, "end": 5.0,
                 "combined_score": 0.8, "nlp_score": 0.7, "audio_score": 0.9,
                 "energy_score": 0.8, "pitch_score": 0.7, "flux_score": 0.6,
                 "matched_keywords": ["test"]}
            ]
            MockDetector.return_value.detect_peaks.return_value = [
                {"clip_start": 0.0, "clip_end": 60.0, "peak_timestamp": 10.0,
                 "score": 0.8, "nlp_score": 0.7, "audio_score": 0.9,
                 "peak_text": "Test moment", "keywords": ["test"],
                 "transcript_segment": "Test content here."}
            ]

            # Create fake extracted clip
            fake_clip_path = str(tmp_path / "fake_clip.mp4")
            open(fake_clip_path, "w").close()

            MockExtractor.return_value.extract_all_clips.return_value = [
                {"clip_start": 0.0, "clip_end": 60.0, "peak_timestamp": 10.0,
                 "score": 0.8, "nlp_score": 0.7, "audio_score": 0.9,
                 "peak_text": "Test", "keywords": ["test"], "clip_index": 0,
                 "clip_duration": 60.0, "raw_clip_path": fake_clip_path,
                 "transcript_segment": "Test content."}
            ]

            MockTracker.return_value.process_clip.return_value = fake_clip_path
            MockSubGen.return_value.burn_subtitles.return_value = fake_clip_path
            MockSubGen.return_value.add_hook_overlay.return_value = fake_clip_path
            MockSubGen.return_value.generate_srt_file.return_value = str(tmp_path / "test.srt")
            open(str(tmp_path / "test.srt"), "w").close()

            MockHookGen.return_value.generate_hooks_batch.return_value = [
                {"clip_start": 0.0, "clip_end": 60.0, "clip_index": 0,
                 "clip_duration": 60.0, "score": 0.8, "nlp_score": 0.7,
                 "audio_score": 0.9, "peak_text": "Test", "keywords": ["test"],
                 "raw_clip_path": fake_clip_path,
                 "vertical_clip_path": fake_clip_path,
                 "subtitled_clip_path": fake_clip_path,
                 "subtitles": [{"text": "Test", "start": 0.0, "end": 5.0}],
                 "transcript_segment": "Test content.",
                 "hooks": [
                     {"headline": "TEST HEADLINE", "hook_text": "Test hook",
                      "caption": "Test caption #test", "platform": "tiktok",
                      "cta": "Follow"}
                 ],
                 "title": "Test Title",
                 "description": "Test description",
                 "tags": ["test", "viral"]}
            ]

            output_dir = str(tmp_path / "output")
            processor = ContentProcessor(
                job_id="test-job-id",
                video_path=str(tmp_path / "video.mp4"),
                output_dir=output_dir,
                temp_dir=str(tmp_path / "temp"),
                config={"max_clips": 1, "clip_duration": 60, "generate_hooks": True},
            )

            results = processor.run()

            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["clip_number"] == 1
            assert "filename" in results[0]
            assert "hooks" in results[0]
