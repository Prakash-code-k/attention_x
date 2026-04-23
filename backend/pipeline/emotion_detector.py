"""
AttentionX – Step 3: Emotional Peak Detection
Uses Librosa (audio energy/pitch analysis) + NLP keyword scoring
to identify the most impactful moments in a video.
"""

import numpy as np
from typing import Optional
import librosa

from utils.config import settings


# ─── High-Impact Keywords (NLP Scoring) ────────────────────────────────────────

HOOK_KEYWORDS = {
    # Surprise / revelation
    "never": 0.9, "secret": 0.9, "truth": 0.8, "finally": 0.8, "shocking": 0.95,
    "revealed": 0.85, "discovered": 0.85, "exposed": 0.85, "hidden": 0.8,

    # Strong emotion
    "incredible": 0.8, "amazing": 0.8, "unbelievable": 0.9, "crazy": 0.8,
    "insane": 0.85, "terrifying": 0.85, "heartbreaking": 0.85, "powerful": 0.75,
    "emotional": 0.7, "devastating": 0.85, "profound": 0.75,

    # Numbers & stats (high engagement)
    "billion": 0.7, "million": 0.65, "thousand": 0.55, "percent": 0.6,
    "study": 0.6, "research": 0.55, "data": 0.5, "proof": 0.7,

    # Action / urgency
    "stop": 0.7, "warning": 0.8, "must": 0.65, "urgent": 0.8, "danger": 0.8,
    "immediately": 0.75, "now": 0.6, "critical": 0.75, "breaking": 0.85,

    # Questions (engagement triggers)
    "why": 0.6, "how": 0.55, "what": 0.5, "who": 0.5,
    "imagine": 0.65, "what if": 0.7, "did you know": 0.85,

    # Success / failure
    "failed": 0.7, "success": 0.65, "mistake": 0.7, "wrong": 0.65,
    "changed": 0.7, "transformed": 0.75, "revolutionized": 0.8,

    # Personal / relatable
    "everyone": 0.6, "nobody": 0.65, "nobody knows": 0.85, "they lied": 0.9,
    "i was wrong": 0.85, "the real reason": 0.9,
}


class EmotionDetector:
    """
    Detects emotionally high-impact moments in audio/transcript.

    Scoring Formula:
        score = (energy_weight × energy_score) +
                (pitch_weight × pitch_score) +
                (nlp_weight × nlp_score)
    """

    def __init__(
        self,
        energy_weight: float = None,
        pitch_weight: float = None,
        nlp_weight: float = None,
        peak_threshold: float = None,
        min_gap_seconds: int = None,
    ):
        self.energy_weight = energy_weight or settings.EMOTION_ENERGY_WEIGHT
        self.pitch_weight = pitch_weight or settings.EMOTION_PITCH_WEIGHT
        self.nlp_weight = nlp_weight or settings.EMOTION_NLP_WEIGHT
        self.peak_threshold = peak_threshold or settings.EMOTION_PEAK_THRESHOLD
        self.min_gap_seconds = min_gap_seconds or settings.EMOTION_MIN_GAP_SECONDS

    def analyze_audio(self, audio_path: str) -> dict:
        """
        Load audio and compute energy + pitch curves.
        Returns dict with time arrays and scores.
        """
        print(f"[EmotionDetector] Loading audio: {audio_path}")

        # Load with librosa (16kHz mono)
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(y) / sr

        # ─── RMS Energy ───────────────────────────────────────────────────────
        # Compute RMS energy in 0.5-second frames
        hop_length = int(sr * 0.5)
        frame_length = int(sr * 1.0)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )

        # Normalize RMS to 0-1
        rms_norm = self._normalize(rms)

        # ─── Spectral Flux (Speech Dynamics) ──────────────────────────────────
        # Captures rapid energy changes (emphasis, excitement)
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        sf_times = librosa.frames_to_time(
            np.arange(len(spectral_flux)), sr=sr, hop_length=hop_length
        )
        sf_norm = self._normalize(spectral_flux)

        # ─── Pitch Variation ──────────────────────────────────────────────────
        # pyin requires hop_length << its internal frame_length (derived from fmin).
        # At sr=16000, fmin=C2~65Hz -> internal frame_length~2048, so hop must be <=512.
        # We use a dedicated small hop then resample the result to the RMS grid.
        pyin_hop = 512
        try:
            f0, _voiced_flag, _voiced_prob = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                hop_length=pyin_hop,
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            window_frames = max(1, int(sr / pyin_hop))
            pitch_var = np.array([
                np.std(f0[max(0, i - window_frames): i + window_frames + 1])
                for i in range(len(f0))
            ])
            pitch_norm_raw = self._normalize(pitch_var)
        except Exception as pyin_err:
            print(f"[EmotionDetector] pyin failed ({pyin_err}), using zero pitch curve")
            pitch_norm_raw = np.zeros(len(rms))

        # ─── Align to common time axis ────────────────────────────────────────
        # Resample all signals to the RMS grid (rms_times)
        energy_score = rms_norm
        pitch_score = self._resample_signal(pitch_norm_raw, len(rms))
        flux_score = self._resample_signal(sf_norm, len(rms))

        return {
            "times": rms_times.tolist(),
            "energy_score": energy_score.tolist(),
            "pitch_score": pitch_score.tolist(),
            "flux_score": flux_score.tolist(),
            "duration": duration,
            "sr": sr,
        }

    def score_segments_nlp(
        self,
        segments: list[dict],
        audio_analysis: dict,
    ) -> list[dict]:
        """
        Combine audio scores with NLP keyword scores per transcript segment.

        Returns segments annotated with:
        - nlp_score
        - audio_score
        - combined_score
        """
        times = np.array(audio_analysis["times"])
        energy = np.array(audio_analysis["energy_score"])
        pitch = np.array(audio_analysis["pitch_score"])
        flux = np.array(audio_analysis["flux_score"])

        scored = []
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            text = seg.get("text", "").lower()

            # ── Audio score for this time window ──────────────────────────────
            mask = (times >= start) & (times <= end)
            if mask.sum() > 0:
                e_score = float(np.mean(energy[mask]))
                p_score = float(np.mean(pitch[mask]))
                f_score = float(np.mean(flux[mask]))
            else:
                e_score = p_score = f_score = 0.0

            audio_score = (e_score + p_score * 0.5 + f_score * 0.5) / 2.0

            # ── NLP score ──────────────────────────────────────────────────────
            nlp_score = 0.0
            matched_keywords = []
            words = text.replace(".", " ").replace(",", " ").split()
            for word in words:
                if word in HOOK_KEYWORDS:
                    nlp_score = max(nlp_score, HOOK_KEYWORDS[word])
                    matched_keywords.append(word)

            # Check bigrams
            for i in range(len(words) - 1):
                bigram = words[i] + " " + words[i + 1]
                if bigram in HOOK_KEYWORDS:
                    nlp_score = max(nlp_score, HOOK_KEYWORDS[bigram])
                    matched_keywords.append(bigram)

            # ── Combined weighted score ────────────────────────────────────────
            combined = (
                self.energy_weight * e_score +
                self.pitch_weight * p_score +
                self.nlp_weight * min(nlp_score, 1.0)
            )

            scored.append({
                **seg,
                "energy_score": round(e_score, 4),
                "pitch_score": round(p_score, 4),
                "flux_score": round(f_score, 4),
                "nlp_score": round(nlp_score, 4),
                "audio_score": round(audio_score, 4),
                "combined_score": round(combined, 4),
                "matched_keywords": matched_keywords,
            })

        return sorted(scored, key=lambda x: x["combined_score"], reverse=True)

    def detect_peaks(
        self,
        scored_segments: list[dict],
        max_clips: int = 5,
        clip_duration: int = 60,
    ) -> list[dict]:
        """
        Select top N non-overlapping emotional peaks as clip start points.

        Returns list of peak moments:
        {start, end, score, text, keywords}
        """
        peaks = []
        used_ranges = []

        for seg in scored_segments:
            if seg["combined_score"] < self.peak_threshold:
                continue

            seg_start = seg["start"]
            seg_end = seg["end"]

            # Center clip around detected peak
            clip_start = max(0, seg_start - clip_duration * 0.25)
            clip_end = clip_start + clip_duration

            # Check no overlap with already-selected clips
            overlap = False
            for (us, ue) in used_ranges:
                if not (clip_end <= us or clip_start >= ue):
                    overlap = True
                    break

            # Check minimum gap from other peaks
            for (us, _) in used_ranges:
                if abs(clip_start - us) < self.min_gap_seconds:
                    overlap = True
                    break

            if not overlap:
                peaks.append({
                    "clip_start": round(clip_start, 2),
                    "clip_end": round(clip_end, 2),
                    "peak_timestamp": round(seg_start, 2),
                    "score": seg["combined_score"],
                    "nlp_score": seg["nlp_score"],
                    "audio_score": seg["audio_score"],
                    "peak_text": seg.get("text", ""),
                    "keywords": seg.get("matched_keywords", []),
                })
                used_ranges.append((clip_start, clip_end))

            if len(peaks) >= max_clips:
                break

        print(f"[EmotionDetector] Detected {len(peaks)} high-impact peaks")
        return sorted(peaks, key=lambda x: x["clip_start"])

    def detect_peaks_from_audio_only(
        self,
        audio_analysis: dict,
        max_clips: int = 5,
        clip_duration: int = 60,
    ) -> list[dict]:
        """
        Fallback: detect peaks from audio signal alone (no transcript needed).
        """
        times = np.array(audio_analysis["times"])
        energy = np.array(audio_analysis["energy_score"])
        pitch = np.array(audio_analysis["pitch_score"])
        flux = np.array(audio_analysis["flux_score"])

        combined = (
            self.energy_weight * energy +
            self.pitch_weight * pitch +
            self.nlp_weight * flux  # use flux as NLP proxy
        )

        duration = audio_analysis["duration"]
        peaks = []
        used_ranges = []

        # Find local maxima
        from scipy.signal import find_peaks as sp_find_peaks
        try:
            peak_indices, _ = sp_find_peaks(
                combined,
                height=self.peak_threshold,
                distance=int(self.min_gap_seconds / 0.5),  # frames
            )
        except Exception:
            peak_indices = np.argsort(combined)[-max_clips * 3:][::-1]

        for idx in peak_indices:
            t = float(times[idx])
            clip_start = max(0, t - clip_duration * 0.25)
            clip_end = min(duration, clip_start + clip_duration)

            overlap = any(
                not (clip_end <= us or clip_start >= ue)
                for (us, ue) in used_ranges
            )
            if not overlap:
                peaks.append({
                    "clip_start": round(clip_start, 2),
                    "clip_end": round(clip_end, 2),
                    "peak_timestamp": round(t, 2),
                    "score": round(float(combined[idx]), 4),
                    "nlp_score": 0.0,
                    "audio_score": round(float(energy[idx]), 4),
                    "peak_text": "",
                    "keywords": [],
                })
                used_ranges.append((clip_start, clip_end))

            if len(peaks) >= max_clips:
                break

        return sorted(peaks, key=lambda x: x["clip_start"])

    # ─── Internal Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    @staticmethod
    def _resample_signal(arr: np.ndarray, target_len: int) -> np.ndarray:
        """Resample 1D array to target length using linear interpolation."""
        if len(arr) == target_len:
            return arr
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, arr)
