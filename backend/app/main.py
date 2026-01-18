import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models import (
    JobCreateRequest,
    JobResponse,
    JobStatusResponse,
    JobStatus,
)
from app.jobs import JobManager
from app.pipeline import run_pipeline


WORK_DIR = Path("/tmp/surfboard_volume_jobs")

job_manager = JobManager(WORK_DIR)


async def cleanup_task():
    """Background task to cleanup expired jobs every hour."""
    while True:
        await asyncio.sleep(3600)
        cleaned = await job_manager.cleanup_expired_jobs()
        if cleaned > 0:
            print(f"Cleaned up {cleaned} expired jobs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_task_handle = asyncio.create_task(cleanup_task())
    yield
    # Shutdown
    cleanup_task_handle.cancel()
    try:
        await cleanup_task_handle
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Surfboard Volume Calculator API",
    description="Compute surfboard volumes from multi-view images using photogrammetry",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/jobs", response_model=JobResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    images: Annotated[list[UploadFile], File(description="Surfboard images")],
    request_data: Annotated[str, Form(description="JSON job request data")],
):
    """
    Create a new volume calculation job.

    Upload multiple images of the surfboard along with job configuration.
    Returns a job_id that can be used to poll for results.
    """
    import json

    try:
        request = JobCreateRequest.model_validate_json(request_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {e}")

    if len(images) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 images required for reconstruction"
        )

    # Validate click point image index
    if request.board_click_point.image_index >= len(images):
        raise HTTPException(
            status_code=400,
            detail=f"board_click_point.image_index ({request.board_click_point.image_index}) "
            f"exceeds number of images ({len(images)})"
        )

    job = await job_manager.create_job(request)

    # Save uploaded images
    images_dir = job.work_dir / "images"
    for i, image in enumerate(images):
        ext = Path(image.filename).suffix if image.filename else ".jpg"
        image_path = images_dir / f"image_{i:03d}{ext}"
        content = await image.read()
        image_path.write_bytes(content)
        job.add_image(image_path)

    # Start processing in background
    background_tasks.add_task(run_pipeline, job)

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress="Job created, queued for processing",
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get job status and results.

    Poll this endpoint to check if processing is complete.
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        errors=job.errors,
    )


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Lightweight status check for a job.

    Use this for frequent polling to minimize data transfer.
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Cancel and cleanup a job.

    Removes all associated data including images and intermediate results.
    """
    deleted = await job_manager.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": "Job deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "jobs_count": len(job_manager.list_jobs())}
