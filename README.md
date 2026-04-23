# ⚡ AttentionX — Automated Content Repurposing Engine

<div align="center">

![AttentionX Banner](https://via.placeholder.com/900x200/0a0a0f/7c3aed?text=⚡+AttentionX+—+Turn+Long+Videos+Into+Viral+Clips)

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange?style=flat-square)](https://github.com/openai/whisper)
[![MediaPipe](https://img.shields.io/badge/Google-MediaPipe-red?style=flat-square)](https://mediapipe.dev)
[![Claude AI](https://img.shields.io/badge/Anthropic-Claude-purple?style=flat-square)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Transform 60-minute podcasts and lectures into 10 viral vertical clips — fully automated.**

[🚀 Quick Start](#-quick-start) • [🎬 Demo](#-demo) • [📖 Docs](#-api-documentation) • [🏆 Hackathon Edge](#-hackathon-winning-edge)

</div>

---

## 🎯 What Is AttentionX?

AttentionX is a fully automated AI pipeline that takes a long-form video (lecture, podcast, interview, webinar) and produces platform-ready **9:16 vertical short clips** complete with:

- ✅ **Smart clip selection** — Detects emotionally charged, high-impact moments using audio energy + NLP
- ✅ **Face-tracked vertical crop** — Converts 16:9 → 9:16 automatically keeping the speaker centered
- ✅ **Burned-in subtitles** — TikTok-style bold captions with word-level timestamps
- ✅ **Viral hooks** — Claude AI generates 3 platform-specific hooks per clip (TikTok, Instagram, YouTube)
- ✅ **One-click pipeline** — Upload → Process → Download. Zero manual editing

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Streamlit)                   │
│              Upload Video ──► Track Progress ──► Download        │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP REST API
┌────────────────────────────▼────────────────────────────────────┐
│                      FASTAPI BACKEND                             │
│   POST /upload ──► Background Job ──► GET /status ──► /results  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    PROCESSING PIPELINE                           │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  STEP 1  │   │  STEP 2  │   │  STEP 3  │   │  STEP 4  │     │
│  │  FFmpeg  │──►│  Whisper │──►│ Librosa  │──►│  FFmpeg  │     │
│  │  Audio   │   │   STT    │   │ + NLP    │   │   Clip   │     │
│  │ Extract  │   │Transcribe│   │Detection │   │ Extract  │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                     │
│  │  STEP 5  │   │  STEP 6  │   │  STEP 7  │                     │
│  │MediaPipe │──►│  FFmpeg  │──►│  Claude  │                     │
│  │  Face    │   │Subtitle  │   │   Hook   │                     │
│  │ Tracking │   │ Burning  │   │Generator │                     │
│  └──────────┘   └──────────┘   └──────────┘                     │
│                                                                  │
│                         ▼                                        │
│         OUTPUT: Vertical MP4 + SRT + JSON Metadata               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Technology | Detail |
|---|---|---|
| 🎵 Audio Extraction | FFmpeg | WAV 16kHz mono, loudness normalized |
| 📝 Transcription | OpenAI Whisper | Word-level timestamps, multi-language |
| 💥 Emotion Detection | Librosa + NLP | RMS energy + pitch variance + keyword scoring |
| ✂️ Clip Extraction | FFmpeg | Lossless stream copy or re-encode |
| 📱 Face Tracking | MediaPipe | Smooth 16:9→9:16 crop with exponential smoothing |
| 💬 Subtitles | FFmpeg drawtext | TikTok-style bold burned-in captions |
| 🪝 Viral Hooks | Claude AI | 3 platform-specific hooks per clip |
| 🔄 Async Processing | FastAPI + BackgroundTasks | Non-blocking, real-time status polling |
| 📦 Export | JSON + SRT + MP4 | Ready to upload to any platform |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required system tools
python3.11+
ffmpeg          # brew install ffmpeg  |  sudo apt install ffmpeg
git
```

### Option 1: Local Setup (Recommended for Demo)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/attentionx.git
cd attentionx

# 2. Run the setup script (handles everything)
chmod +x start.sh
./start.sh
```

**That's it!** Open http://localhost:8501 in your browser.

---

### Option 2: Manual Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/attentionx.git
cd attentionx

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. Create storage directories
mkdir -p storage/uploads storage/outputs storage/temp

# 6. Start backend (Terminal 1)
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 7. Start frontend (Terminal 2)
cd frontend
streamlit run app.py --server.port 8501
```

### Option 3: Docker

```bash
# 1. Clone and configure
git clone https://github.com/yourusername/attentionx.git
cd attentionx
cp .env.example .env
# Edit .env with your API key

# 2. Run with Docker Compose
docker-compose up --build

# Access:
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
attentionx/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .env.example
├── 📄 .gitignore
├── 📄 Dockerfile
├── 📄 docker-compose.yml
├── 📄 start.sh
│
├── 🔧 backend/
│   ├── main.py                    # FastAPI app + endpoints
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── processor.py           # Master pipeline orchestrator
│   │   ├── audio_extractor.py     # Step 1: FFmpeg audio extraction
│   │   ├── transcriber.py         # Step 2: Whisper STT
│   │   ├── emotion_detector.py    # Step 3: Librosa + NLP scoring
│   │   ├── clip_extractor.py      # Step 4: Clip extraction
│   │   ├── face_tracker.py        # Step 5: MediaPipe face tracking
│   │   ├── subtitle_generator.py  # Step 6: Subtitle burning
│   │   └── hook_generator.py      # Step 7: Claude AI hooks
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Pydantic settings
│       └── helpers.py             # Utility functions
│
├── 🎨 frontend/
│   └── app.py                     # Streamlit UI
│
└── 📦 storage/                    # Auto-created at runtime
    ├── uploads/                   # Raw uploaded videos
    ├── outputs/                   # Processed clips + metadata
    └── temp/                      # Intermediate files (auto-cleaned)
```

---

## 🔑 Getting an API Key

AttentionX uses Claude AI for viral hook generation. To enable this:

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account (free tier available)
3. Generate an API key
4. Add it to your `.env` file:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

> **Note:** Hook generation works without an API key using our built-in template engine. The AI-powered hooks are simply higher quality.

---

## 📖 API Documentation

Once running, visit **http://localhost:8000/docs** for interactive Swagger UI.

### Key Endpoints

```
POST   /upload              Upload video + start processing
GET    /status/{job_id}     Poll processing progress
GET    /results/{job_id}    Get completed results
GET    /download/{job_id}/{filename}  Download a clip
DELETE /jobs/{job_id}       Clean up job data
GET    /jobs                List all jobs
GET    /health              Service health check
```

### Example API Call

```python
import requests

# Upload video
with open("my_podcast.mp4", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/upload",
        files={"file": ("my_podcast.mp4", f, "video/mp4")},
        params={"max_clips": 5, "clip_duration": 60, "generate_hooks": True}
    )
    job_id = resp.json()["job_id"]

# Poll for completion
import time
while True:
    status = requests.get(f"http://localhost:8000/status/{job_id}").json()
    print(f"Progress: {status['progress']}% - {status['stage']}")
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(3)

# Get results
results = requests.get(f"http://localhost:8000/results/{job_id}").json()
for clip in results["clips"]:
    print(f"Clip {clip['clip_number']}: {clip['title']}")
    print(f"Download: {clip['download_url']}")
```

---

## 🎬 Demo

### Sample Output

**Input:** 60-minute tech podcast (16:9 horizontal)

**Output** (5 clips, processed in ~8 minutes on CPU):

```
Clip 1 — 04:23 → 05:23 | Score: 87%
  Title: "The Hidden Cost of Technical Debt Nobody Talks About"
  Hook:  "EVERY STARTUP FAILS FOR THIS EXACT REASON"
  Tags:  #startup #techdebt #programming #viral #coding

Clip 2 — 18:41 → 19:41 | Score: 82%
  Title: "Why 90% of Developers Get This Completely Wrong"
  Hook:  "THEY NEVER TEACH YOU THIS IN COMPUTER SCIENCE"
  Tags:  #developer #coding #javascript #learn #shorts

Clip 3 — 34:12 → 35:12 | Score: 79%
  Title: "The Real Reason Your Code Is Slow"
  Hook:  "WARNING: THIS MISTAKE COSTS COMPANIES MILLIONS"
  Tags:  #performance #coding #software #engineering #fyp
```

### Sample SRT Output

```srt
1
00:00:00,000 --> 00:00:01,200
THE TRUTH IS

2
00:00:01,200 --> 00:00:02,800
NOBODY TALKS ABOUT

3
00:00:02,800 --> 00:00:04,400
TECHNICAL DEBT UNTIL

4
00:00:04,400 --> 00:00:06,000
IT'S TOO LATE
```

---

## ⚙️ Configuration Reference

All settings are in `.env` or `backend/utils/config.py`:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | `tiny/base/small/medium/large` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MAX_CLIPS_PER_VIDEO` | `10` | Max clips to generate |
| `DEFAULT_CLIP_DURATION` | `60` | Clip length in seconds |
| `OUTPUT_WIDTH` | `1080` | Output video width |
| `OUTPUT_HEIGHT` | `1920` | Output video height |
| `EMOTION_PEAK_THRESHOLD` | `0.65` | Min score to select as peak |
| `FACE_TRACKING_SMOOTH_FACTOR` | `0.15` | Camera smoothness (lower = smoother) |
| `SUBTITLE_FONT_SIZE` | `60` | Caption font size |

---

## 🚀 Deployment

### Free Deployment Options

#### Railway (Recommended)
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

#### Render
```bash
# Connect GitHub repo to render.com
# Set build command: pip install -r requirements.txt
# Set start command: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

#### Local Network Demo
```bash
# Expose local server (for demos without deployment)
# Install ngrok: https://ngrok.com
ngrok http 8501  # Frontend
ngrok http 8000  # Backend API
```

---

## 🏆 Hackathon Winning Edge

### 3 Unique Features That Set Us Apart

**1. 🧠 Multi-Signal Emotion Detection**
We don't just look for loud moments. Our proprietary scoring combines:
- Audio RMS energy (volume peaks)
- Spectral flux (rapid speech dynamics)
- Pitch variation (emotional vocal range)
- NLP keyword scoring (power words database of 60+ terms)

This creates a **composite impact score** that identifies moments humans would actually find compelling.

**2. 📱 Smooth Face-Tracked Vertical Crop**
Unlike naive center-crop tools, we use MediaPipe face detection + exponential position smoothing to create a natural "camera operator" feel. The crop window follows the speaker without jarring jumps.

**3. 🪝 Platform-Native AI Hooks**
Hooks aren't generic — they're tailored per platform:
- **TikTok**: FOMO + shock + curiosity gap
- **Instagram**: Lifestyle + aspiration + save-worthy
- **YouTube Shorts**: Question-based + SEO-optimized

### Demo Script (2-Minute Pitch)

```
"Let me show you a problem every content creator faces.

You spend 3 hours recording a podcast.
You get 200 views.
Meanwhile, a 60-second clip from that same podcast
could get 2 million views on TikTok.

The problem isn't your content. It's distribution.
AttentionX solves this automatically.

[LIVE DEMO: Upload a 10-min video]

Watch as our AI:
1. Finds the 5 most emotionally compelling moments — not just the loudest ones
2. Converts horizontal video to vertical with face tracking
3. Burns TikTok-style captions
4. Generates platform-specific viral hooks using Claude AI

[Show results in 60 seconds on a pre-processed video]

What took an editor 4 hours now takes 8 minutes.
We're not replacing creators — we're giving them a 10x multiplier.

Thank you."
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition
- [Google MediaPipe](https://mediapipe.dev) — Face detection
- [Librosa](https://librosa.org) — Audio analysis
- [FFmpeg](https://ffmpeg.org) — Video processing
- [Anthropic Claude](https://anthropic.com) — AI hook generation

---

<div align="center">
Built with ❤️ for hackathons | ⚡ AttentionX
</div>
