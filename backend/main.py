"""
AttentionX – Automated Content Repurposing Engine
FastAPI Backend – Main Entry Point
"""

import os
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from pipeline.processor import ContentProcessor
from utils.config import settings
from utils.helpers import get_video_info, cleanup_temp_files

# ─── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="AttentionX API",
    description="Automated Content Repurposing Engine – Turn long-form videos into viral clips",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static File Serving ────────────────────────────────────────────────────────
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

# ─── In-Memory Job Tracker ──────────────────────────────────────────────────────
jobs: dict[str, dict] = {}


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AttentionX",
        "status": "running",
        "version": "1.0.0",
        "message": "🎬 Automated Content Repurposing Engine is live!"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "gpu_available": _check_gpu()}


@app.post("/upload", tags=["Processing"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_clips: Optional[int] = 5,
    clip_duration: Optional[int] = 60,
    generate_hooks: Optional[bool] = True,
):
    """
    Upload a video file for processing.
    Returns a job_id to track progress.
    """
    # Validate file type
    allowed_types = ["video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: mp4, mpeg, mov, avi, webm"
        )

    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    upload_path = Path(settings.UPLOAD_DIR) / f"{job_id}_{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get video metadata
    video_info = get_video_info(str(upload_path))

    # Initialize job tracker
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "upload_path": str(upload_path),
        "video_info": video_info,
        "progress": 0,
        "stage": "Queued",
        "clips": [],
        "error": None,
        "config": {
            "max_clips": max_clips,
            "clip_duration": clip_duration,
            "generate_hooks": generate_hooks,
        }
    }

    # Start background processing
    background_tasks.add_task(process_video_job, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Video '{file.filename}' uploaded successfully. Processing started.",
        "video_info": video_info,
        "estimated_time_seconds": max(30, video_info.get("duration", 60) * 2),
    }


@app.get("/status/{job_id}", tags=["Processing"])
async def get_job_status(job_id: str):
    """Get the current processing status of a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "clips": job["clips"] if job["status"] == "completed" else [],
        "error": job["error"],
    }


@app.get("/results/{job_id}", tags=["Results"])
async def get_results(job_id: str):
    """Get full results for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Current status: {job['status']}"
        )

    return {
        "job_id": job_id,
        "status": "completed",
        "filename": job["filename"],
        "clips": job["clips"],
        "total_clips": len(job["clips"]),
        "download_base_url": f"/outputs/{job_id}/",
    }


@app.get("/download/{job_id}/{clip_filename}", tags=["Results"])
async def download_clip(job_id: str, clip_filename: str):
    """Download a specific processed clip."""
    clip_path = Path(settings.OUTPUT_DIR) / job_id / clip_filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(str(clip_path), media_type="video/mp4", filename=clip_filename)


@app.delete("/jobs/{job_id}", tags=["Management"])
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    # Cleanup files
    cleanup_temp_files(job_id, settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.TEMP_DIR)

    del jobs[job_id]
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/jobs", tags=["Management"])
async def list_jobs():
    """List all jobs."""
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": jid,
                "status": j["status"],
                "filename": j["filename"],
                "progress": j["progress"],
            }
            for jid, j in jobs.items()
        ]
    }


# ─── Background Processing Task ────────────────────────────────────────────────

async def process_video_job(job_id: str):
    """Background task that runs the full processing pipeline."""
    job = jobs[job_id]

    def update_progress(progress: int, stage: str):
        jobs[job_id]["progress"] = progress
        jobs[job_id]["stage"] = stage
        print(f"[{job_id[:8]}] {progress}% – {stage}")

    try:
        jobs[job_id]["status"] = "processing"
        update_progress(5, "Initializing pipeline")

        processor = ContentProcessor(
            job_id=job_id,
            video_path=job["upload_path"],
            output_dir=str(Path(settings.OUTPUT_DIR) / job_id),
            temp_dir=str(Path(settings.TEMP_DIR) / job_id),
            config=job["config"],
            progress_callback=update_progress,
        )

        clips = await asyncio.get_running_loop().run_in_executor(
            None, processor.run
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["clips"] = clips
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "Complete"
        print(f"[{job_id[:8]}] ✅ Processing complete. {len(clips)} clips generated.")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["stage"] = "Failed"
        print(f"[{job_id[:8]}] ❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )
