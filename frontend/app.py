"""
AttentionX - Streamlit Frontend
Drag-and-drop video upload + real-time processing dashboard
"""

import streamlit as st
import requests
import time
import json
import os
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
POLL_INTERVAL = 2  # seconds

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttentionX - AI Content Repurposer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    * { font-family: 'Space Grotesk', sans-serif; }

    .main { background: #0a0a0f; }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
    }

    /* Header */
    .atx-header {
        text-align: center;
        padding: 2rem 0 1rem;
        background: linear-gradient(90deg, #7c3aed, #ec4899, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .atx-subheader {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }

    .clip-card {
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.2s;
    }

    .hook-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px;
    }

    .keyword-chip {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #f59e0b;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
    }

    .score-bar {
        height: 8px;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        border-radius: 4px;
    }

    /* Pipeline stage indicator */
    .stage-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(124, 58, 237, 0.1);
        border-radius: 8px;
        font-size: 0.9rem;
        color: #a78bfa;
        border-left: 3px solid #7c3aed;
    }

    /* Upload zone */
    .uploadedFile {
        background: rgba(124, 58, 237, 0.1) !important;
        border: 1px dashed rgba(124, 58, 237, 0.5) !important;
        border-radius: 12px !important;
    }

    /* Sidebar */
    .css-1d391kg { background: #0f0f1a; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #ec4899) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.4) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #ec4899, #f59e0b) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── State Init ────────────────────────────────────────────────────────────────
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "results" not in st.session_state:
    st.session_state.results = None
if "processing" not in st.session_state:
    st.session_state.processing = False


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="atx-header">AttentionX</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="atx-subheader">AI-Powered Content Repurposing Engine • '
    'Turn 60-min videos into 10 viral clips in minutes</div>',
    unsafe_allow_html=True,
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Processing Settings")
    st.markdown("---")

    max_clips = st.slider("Max Clips to Generate", 1, 10, 5)
    clip_duration = st.slider("Clip Duration (seconds)", 15, 90, 60)
    generate_hooks = st.toggle("Generate Viral Hooks (AI)", value=True)

    st.markdown("---")
    st.markdown("### Pipeline Stages")
    stages = [
        ("", "Audio Extraction"),
        ("", "Whisper Transcription"),
        ("", "Emotion Detection"),
        ("", "Clip Extraction"),
        ("", "Face Tracking (9:16)"),
        ("", "Subtitle Burning"),
        ("", "Hook Generation"),
    ]
    for icon, stage in stages:
        st.markdown(f"{icon} {stage}")

    st.markdown("---")
    st.markdown("### 🔗 API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health.get("status") == "healthy":
            st.success("Backend Connected")
            gpu_status = "GPU" if health.get("gpu_available") else "CPU"
            st.info(f"Running on: {gpu_status}")
    except Exception:
        st.error("Backend Offline")
        st.markdown(f"Start with: `cd backend && uvicorn main:app`")


# ─── Main Content ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem"></div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #a78bfa">16:9 → 9:16</div>
        <div style="color: #64748b; font-size: 0.85rem">Face-tracked vertical crop</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem"></div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #ec4899">AI Detection</div>
        <div style="color: #64748b; font-size: 0.85rem">Emotion + energy peaks</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2rem"></div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b">Viral Hooks</div>
        <div style="color: #64748b; font-size: 0.85rem">Claude AI-powered captions</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Upload Section ────────────────────────────────────────────────────────────
if not st.session_state.job_id:
    st.markdown("##Upload Your Video")

    uploaded_file = st.file_uploader(
        "Drop your podcast, lecture, or long-form video here",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Max 2GB. Supported: MP4, MOV, AVI, MKV, WebM",
    )

    if uploaded_file is not None:
        st.markdown(f"**File:** `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.1f} MB)")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            if st.button("Start Processing", use_container_width=True):
                with st.spinner("Uploading video..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/upload",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            params={
                                "max_clips": max_clips,
                                "clip_duration": clip_duration,
                                "generate_hooks": generate_hooks,
                            },
                            timeout=60,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.job_id = data["job_id"]
                            st.session_state.processing = True

                            info = data.get("video_info", {})
                            st.success(f"Video uploaded! Job ID: `{data['job_id'][:8]}...`")
                            st.info(
                                f"Duration: {info.get('duration_formatted', 'N/A')} | "
                                f"Resolution: {info.get('width', 0)}×{info.get('height', 0)} | "
                                f"FPS: {info.get('fps', 0)}"
                            )
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Make sure it's running!")

        with col_b:
            st.info(
                f"Will generate up to **{max_clips} clips** "
                f"of **{clip_duration}s** each"
            )


# ─── Processing Status ─────────────────────────────────────────────────────────
elif st.session_state.job_id and not st.session_state.results:
    job_id = st.session_state.job_id

    st.markdown("## ⚙️ Processing Your Video")
    st.markdown(f"**Job ID:** `{job_id[:8]}...`")

    # Status container
    status_container = st.container()
    progress_bar = st.progress(0)
    stage_text = st.empty()
    cancel_col, _ = st.columns([1, 3])

    with cancel_col:
        if st.button("Cancel Job"):
            try:
                requests.delete(f"{API_URL}/jobs/{job_id}", timeout=10)
            except Exception:
                pass
            st.session_state.job_id = None
            st.session_state.results = None
            st.rerun()

    # Polling loop
    for _ in range(600):  # Max 20 minutes
        try:
            status_resp = requests.get(f"{API_URL}/status/{job_id}", timeout=10)
            if status_resp.status_code != 200:
                st.error("Job not found. Resetting...")
                st.session_state.job_id = None
                st.rerun()
                break

            status = status_resp.json()
            progress = status.get("progress", 0)
            stage = status.get("stage", "Processing...")
            job_status = status.get("status", "processing")

            progress_bar.progress(progress / 100)
            stage_text.markdown(
                f'<div class="stage-indicator">⏳ {stage} ({progress}%)</div>',
                unsafe_allow_html=True,
            )

            if job_status == "completed":
                # Fetch results
                results_resp = requests.get(f"{API_URL}/results/{job_id}", timeout=10)
                if results_resp.status_code == 200:
                    st.session_state.results = results_resp.json()
                    st.session_state.processing = False
                    st.balloons()
                    st.rerun()
                break

            elif job_status == "failed":
                st.error(f"Processing failed: {status.get('error', 'Unknown error')}")
                with st.expander("Error Details"):
                    st.code(status.get("error", ""))
                if st.button("Try Again"):
                    st.session_state.job_id = None
                    st.rerun()
                break

        except Exception as e:
            st.warning(f"Status check failed: {e}")

        time.sleep(POLL_INTERVAL)


# ─── Results Section ───────────────────────────────────────────────────────────
elif st.session_state.results:
    results = st.session_state.results
    clips = results.get("clips", [])

    st.markdown("##Your Viral Clips Are Ready!")

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Clips Generated", len(clips))
    with mc2:
        total_duration = sum(c.get("duration_seconds", 0) for c in clips)
        st.metric("Total Duration", f"{total_duration:.0f}s")
    with mc3:
        avg_score = sum(c.get("score", 0) for c in clips) / max(len(clips), 1)
        st.metric("Avg Impact Score", f"{avg_score:.0%}")
    with mc4:
        total_hooks = sum(len(c.get("hooks", [])) for c in clips)
        st.metric("Hooks Generated", total_hooks)

    st.markdown("---")

    # New job button
    if st.button("Process Another Video"):
        st.session_state.job_id = None
        st.session_state.results = None
        st.rerun()

    # ─── Clip Cards ───────────────────────────────────────────────────────────
    for i, clip in enumerate(clips):
        with st.expander(
            f"Clip {clip['clip_number']} — "
            f"{clip.get('timestamp_start', '')} to {clip.get('timestamp_end', '')} — "
            f"Score: {clip.get('score', 0):.0%}",
            expanded=(i == 0),
        ):
            col_left, col_right = st.columns([1, 1])

            with col_left:
                # Score bar
                score_pct = int(clip.get("score", 0) * 100)
                st.markdown(f"**Impact Score: {score_pct}%**")
                st.markdown(
                    f'<div class="score-bar" style="width: {score_pct}%"></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # Keywords
                kws = clip.get("keywords", [])
                if kws:
                    st.markdown("**Detected Keywords:**")
                    chips = " ".join(
                        f'<span class="keyword-chip">{kw}</span>' for kw in kws[:8]
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                # Peak transcript
                peak_text = clip.get("peak_text", "")
                if peak_text:
                    st.markdown("**Peak Moment:**")
                    st.info(f'"{peak_text}"')

                # Download buttons
                st.markdown("**Downloads:**")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    video_url = f"{API_URL}{clip.get('download_url', '')}"
                    st.markdown(
                        f'<a href="{video_url}" target="_blank">'
                        f'<button style="background:linear-gradient(135deg,#7c3aed,#ec4899);'
                        f'color:white;border:none;border-radius:8px;padding:8px 16px;'
                        f'cursor:pointer;font-weight:600;">📹 Download MP4</button></a>',
                        unsafe_allow_html=True,
                    )
                with dl_col2:
                    srt_url = f"{API_URL}{clip.get('srt_url', '')}"
                    st.markdown(
                        f'<a href="{srt_url}" target="_blank">'
                        f'<button style="background:rgba(124,58,237,0.2);'
                        f'color:#a78bfa;border:1px solid rgba(124,58,237,0.4);'
                        f'border-radius:8px;padding:8px 16px;cursor:pointer;font-weight:600;">'
                        f'Download SRT</button></a>',
                        unsafe_allow_html=True,
                    )

            with col_right:
                # Title
                title = clip.get("title", "")
                if title:
                    st.markdown(f"**Title:** {title}")

                # Hooks
                hooks = clip.get("hooks", [])
                if hooks:
                    st.markdown("**🪝 Viral Hooks:**")
                    for hook in hooks[:3]:
                        platform_emoji = {"tiktok": "", "instagram": "", "youtube": ""}.get(
                            hook.get("platform", ""), ""
                        )
                        st.markdown(
                            f'<span class="hook-badge">{platform_emoji} '
                            f'{hook.get("platform", "").upper()}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**{hook.get('headline', '')}**")
                        st.markdown(f"_{hook.get('hook_text', '')}_")
                        st.text_area(
                            f"Caption ({hook.get('platform', '').title()}) — copy & paste:",
                            value=hook.get("caption", ""),
                            height=110,
                            key=f"caption_{i}_{hook.get('platform', '')}",
                        )
                        st.markdown("---")

                # Tags
                tags = clip.get("tags", [])
                if tags:
                    st.markdown("**Tags:**")
                    st.markdown(
                        " ".join(f'<span class="keyword-chip">#{t}</span>' for t in tags[:8]),
                        unsafe_allow_html=True,
                    )

    # ─── Export All Metadata ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Export All Metadata")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.download_button(
            "Download Full JSON",
            data=json.dumps(results, indent=2),
            file_name=f"attentionx_{results.get('job_id', 'results')[:8]}.json",
            mime="application/json",
        ):
            st.success("Downloaded!")

    with col_exp2:
        # Build markdown summary
        md_lines = [f"# AttentionX Results\n", f"**Job:** {results.get('job_id', '')[:8]}\n"]
        for clip in clips:
            md_lines.append(f"\n## Clip {clip['clip_number']}")
            md_lines.append(f"**Timestamp:** {clip.get('timestamp_start')} → {clip.get('timestamp_end')}")
            md_lines.append(f"**Title:** {clip.get('title', '')}")
            for h in clip.get("hooks", [])[:1]:
                md_lines.append(f"**Headline:** {h.get('headline', '')}")
                md_lines.append(f"**Caption:**\n{h.get('caption', '')}")
        md_content = "\n".join(md_lines)

        if st.download_button(
            "Download Markdown Summary",
            data=md_content,
            file_name="attentionx_summary.md",
            mime="text/markdown",
        ):
            st.success("Downloaded!")


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 1rem 0">
    Made For Hackathon
    </div>
    """,
    unsafe_allow_html=True,
)