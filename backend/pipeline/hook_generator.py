"""
AttentionX – Step 7: Viral Hook Generator
Uses Claude AI to generate viral hooks, titles, and captions for each clip.
"""

import os
import json
import anthropic
from typing import Optional

from utils.config import settings


HOOK_SYSTEM_PROMPT = """You are an expert viral content strategist for TikTok, Instagram Reels, and YouTube Shorts.
Your job is to craft irresistible hooks that stop people mid-scroll.

Rules for great hooks:
1. Start with tension, curiosity, or bold claim — NEVER "In this video..."
2. Use power words: Secret, Never, Finally, Everyone, Warning, Shocking
3. Create a knowledge gap — make them NEED to watch
4. Maximum 12 words for the hook headline
5. Keep captions conversational, punchy, platform-native

Output ONLY valid JSON. No markdown, no preamble."""

HOOK_USER_TEMPLATE = """Generate viral content for this video clip moment.

TRANSCRIPT EXCERPT:
"{transcript}"

DETECTED KEYWORDS: {keywords}
CLIP DURATION: {duration} seconds
EMOTIONAL INTENSITY: {score}/1.0

Generate exactly this JSON structure:
{{
  "hooks": [
    {{
      "headline": "The 10-word hook headline (all caps)",
      "hook_text": "1-2 sentence explanation that builds curiosity",
      "caption": "Full social media caption with hashtags (2-3 sentences + 5 hashtags)",
      "platform": "tiktok",
      "cta": "call-to-action phrase (e.g., 'Follow for more')"
    }},
    {{
      "headline": "Alternative hook headline",
      "hook_text": "Different angle on the same moment",
      "caption": "Alternative caption for Instagram",
      "platform": "instagram",
      "cta": "Save this for later"
    }},
    {{
      "headline": "Third option — question format",
      "hook_text": "Question-based hook that demands an answer",
      "caption": "YouTube Shorts style caption",
      "platform": "youtube",
      "cta": "Subscribe for the full story"
    }}
  ],
  "title": "SEO-optimized video title (max 60 chars)",
  "description": "2-sentence video description",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}"""


class HookGenerator:
    """
    Generates viral hooks and captions using Claude AI.
    Falls back to template-based generation if API unavailable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
    ):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.model = model or settings.HOOK_MODEL
        self._client = None

    def _get_client(self) -> Optional[anthropic.Anthropic]:
        """Get or create Anthropic client."""
        if self._client is None and self.api_key:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate_hooks(
        self,
        transcript: str,
        keywords: list[str],
        clip_duration: float,
        emotional_score: float,
    ) -> dict:
        """
        Generate viral hooks for a clip.

        Returns:
            {
                hooks: [...],
                title: str,
                description: str,
                tags: [...]
            }
        """
        client = self._get_client()

        if not client or not self.api_key:
            print("[HookGenerator] No API key — using template fallback")
            return self._template_fallback(transcript, keywords, emotional_score)

        # Build prompt
        user_prompt = HOOK_USER_TEMPLATE.format(
            transcript=transcript[:800],  # Limit context
            keywords=", ".join(keywords[:10]) if keywords else "none detected",
            duration=f"{clip_duration:.0f}",
            score=f"{emotional_score:.2f}",
        )

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=HOOK_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw = response.content[0].text.strip()

            # Parse JSON
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    raise ValueError("Could not parse JSON from response")

            print(f"[HookGenerator] Generated {len(result.get('hooks', []))} hooks via Claude")
            return result

        except Exception as e:
            print(f"[HookGenerator] API call failed: {e}. Using fallback.")
            return self._template_fallback(transcript, keywords, emotional_score)

    def generate_hooks_batch(self, clips: list[dict]) -> list[dict]:
        """
        Generate hooks for multiple clips.
        Returns clips list with 'hooks' key added.
        """
        enriched = []
        for i, clip in enumerate(clips):
            print(f"[HookGenerator] Generating hooks for clip {i+1}/{len(clips)}")

            transcript = clip.get("peak_text", "")
            # Get broader context from full transcript if available
            if "transcript_segment" in clip:
                transcript = clip["transcript_segment"]

            hooks_data = self.generate_hooks(
                transcript=transcript,
                keywords=clip.get("keywords", []),
                clip_duration=clip.get("clip_duration", 60),
                emotional_score=clip.get("score", 0.5),
            )

            enriched.append({
                **clip,
                "hooks": hooks_data.get("hooks", []),
                "title": hooks_data.get("title", ""),
                "description": hooks_data.get("description", ""),
                "tags": hooks_data.get("tags", []),
            })

        return enriched

    def _template_fallback(
        self,
        transcript: str,
        keywords: list[str],
        score: float,
    ) -> dict:
        """
        Rule-based hook generation when AI API is unavailable.
        Uses keyword patterns and templates.
        """
        # Select best keyword for hook
        top_kw = keywords[0].upper() if keywords else "THIS"
        score_pct = int(score * 100)

        # Pick template based on score
        if score >= 0.8:
            headline = f"NOBODY TALKS ABOUT {top_kw} (AND THEY SHOULD)"
            hook_text = "This moment changes everything you thought you knew."
        elif score >= 0.6:
            headline = f"THE TRUTH ABOUT {top_kw} THEY HIDE FROM YOU"
            hook_text = "Most people miss this completely."
        else:
            headline = f"YOU NEED TO HEAR THIS ABOUT {top_kw}"
            hook_text = "Pay close attention to what happens next."

        # Extract first sentence of transcript as teaser
        sentences = transcript.split(". ") if transcript else [""]
        teaser = sentences[0][:100] if sentences else ""

        return {
            "hooks": [
                {
                    "headline": headline,
                    "hook_text": hook_text,
                    "caption": (
                        f"{hook_text} {teaser}\n\n"
                        f"#viral #trending #mustwatch #facts #{top_kw.lower()}"
                    ),
                    "platform": "tiktok",
                    "cta": "Follow for more eye-opening content 👆",
                },
                {
                    "headline": f"WAIT FOR THE {top_kw} PART 👀",
                    "hook_text": "The part that made everyone stop scrolling.",
                    "caption": (
                        f"The moment that changes everything. {teaser}\n\n"
                        f"#reels #trending #mindblown #educational #{top_kw.lower()}"
                    ),
                    "platform": "instagram",
                    "cta": "Save this 🔖",
                },
                {
                    "headline": f"WHY DOES NOBODY TALK ABOUT {top_kw}?",
                    "hook_text": "This is why you can't stop watching.",
                    "caption": f"I had to share this. {teaser} Full video in description.",
                    "platform": "youtube",
                    "cta": "Subscribe and hit the bell 🔔",
                },
            ],
            "title": f"The Hidden Truth About {top_kw.title()} (Full Breakdown)",
            "description": f"In this clip, we break down the most important moment. {teaser}",
            "tags": keywords[:5] + ["viral", "trending", "shorts", "fyp", "educational"],
        }
