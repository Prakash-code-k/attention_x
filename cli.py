"""
AttentionX – Command Line Interface
Process videos directly from the terminal without the web UI.

Usage:
    python cli.py --input my_podcast.mp4 --clips 5 --duration 60
    python cli.py --input lecture.mp4 --clips 3 --no-hooks --output ./results
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from pipeline.processor import ContentProcessor
from utils.config import settings
from utils.helpers import get_video_info, format_duration


def print_banner():
    print("""
╔═══════════════════════════════════════════════════╗
║  ⚡ AttentionX – Content Repurposing Engine CLI   ║
╚═══════════════════════════════════════════════════╝
""")


def print_progress(progress: int, stage: str):
    bar_len = 40
    filled = int(bar_len * progress / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r[{bar}] {progress:3d}% │ {stage:<35}", end="", flush=True)
    if progress == 100:
        print()


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="AttentionX – Automated Content Repurposing Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --input podcast.mp4
  python cli.py --input lecture.mp4 --clips 3 --duration 45
  python cli.py --input video.mp4 --output ./my_clips --no-hooks
  python cli.py --input video.mp4 --whisper-model small
        """
    )

    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--clips", "-n", type=int, default=5, help="Max clips to generate (default: 5)")
    parser.add_argument("--duration", "-d", type=int, default=60, help="Clip duration in seconds (default: 60)")
    parser.add_argument("--no-hooks", action="store_true", help="Skip AI hook generation")
    parser.add_argument("--whisper-model", default=settings.WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--language", default="en", help="Audio language code (default: en)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # ─── Validate input ────────────────────────────────────────────────────────
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    # ─── Video info ────────────────────────────────────────────────────────────
    print(f"📹 Input: {input_path.name}")
    info = get_video_info(str(input_path))
    print(f"   Duration:   {info.get('duration_formatted', 'unknown')}")
    print(f"   Resolution: {info.get('width', 0)}×{info.get('height', 0)}")
    print(f"   Size:       {info.get('size_mb', 0):.1f} MB")
    print()

    # ─── Setup directories ─────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    output_dir = Path(args.output) / job_id
    temp_dir = Path("./temp_processing") / job_id

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️  Settings: {args.clips} clips × {args.duration}s each")
    print(f"🤖 Whisper model: {args.whisper_model}")
    print(f"🪝 Hook generation: {'disabled' if args.no_hooks else 'enabled'}")
    print()
    print("─" * 55)
    print("🚀 Starting pipeline...")
    print()

    # Override whisper model if specified
    settings.WHISPER_MODEL = args.whisper_model
    settings.WHISPER_LANGUAGE = args.language

    # ─── Run pipeline ─────────────────────────────────────────────────────────
    config = {
        "max_clips": args.clips,
        "clip_duration": args.duration,
        "generate_hooks": not args.no_hooks,
    }

    processor = ContentProcessor(
        job_id=job_id,
        video_path=str(input_path),
        output_dir=str(output_dir),
        temp_dir=str(temp_dir),
        config=config,
        progress_callback=print_progress,
    )

    try:
        clips = processor.run()
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ─── Output results ────────────────────────────────────────────────────────
    print()
    print("─" * 55)
    print(f"✅ Done! Generated {len(clips)} clips\n")

    if args.json:
        print(json.dumps({"job_id": job_id, "clips": clips}, indent=2))
    else:
        for clip in clips:
            print(f"🎬 Clip {clip['clip_number']} │ "
                  f"{clip.get('timestamp_start', '')} → {clip.get('timestamp_end', '')} │ "
                  f"Score: {clip.get('score', 0):.0%}")
            print(f"   📌 {clip.get('title', 'No title')}")

            hooks = clip.get("hooks", [])
            if hooks:
                print(f"   🪝 \"{hooks[0].get('headline', '')}\"")

            print(f"   📁 {output_dir / clip.get('filename', '')}")
            print()

    print(f"📦 All files saved to: {output_dir.resolve()}")
    print(f"📊 Metadata: {output_dir / 'job_metadata.json'}")


if __name__ == "__main__":
    main()
