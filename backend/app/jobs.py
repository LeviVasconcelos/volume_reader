import asyncio
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.models import (
    JobStatus,
    JobResult,
    JobCreateRequest,
    ScaleMethod,
)


class Job:
    def __init__(
        self,
        job_id: str,
        request: JobCreateRequest,
        work_dir: Path,
    ):
        self.job_id = job_id
        self.request = request
        self.work_dir = work_dir
        self.status = JobStatus.PENDING
        self.progress: Optional[str] = None
        self.result: Optional[JobResult] = None
        self.errors: list[str] = []
        self.created_at = datetime.utcnow()
        self.image_paths: list[Path] = []

    def add_image(self, image_path: Path):
        self.image_paths.append(image_path)

    def set_processing(self, progress: str = "Starting pipeline"):
        self.status = JobStatus.PROCESSING
        self.progress = progress

    def set_progress(self, progress: str):
        self.progress = progress

    def set_completed(self, result: JobResult):
        self.status = JobStatus.COMPLETED
        self.result = result
        self.progress = "Completed"

    def set_partial(self, result: JobResult, warnings: list[str]):
        self.status = JobStatus.PARTIAL
        self.result = result
        self.progress = "Completed with warnings"

    def set_failed(self, errors: list[str]):
        self.status = JobStatus.FAILED
        self.errors = errors
        self.progress = "Failed"

    def is_expired(self, retention_hours: int = 48) -> bool:
        return datetime.utcnow() > self.created_at + timedelta(hours=retention_hours)


class JobManager:
    def __init__(self, base_work_dir: Path):
        self.base_work_dir = base_work_dir
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, request: JobCreateRequest) -> Job:
        async with self._lock:
            job_id = str(uuid.uuid4())
            work_dir = self.base_work_dir / job_id
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "images").mkdir(exist_ok=True)

            job = Job(job_id=job_id, request=request, work_dir=work_dir)
            self.jobs[job_id] = job
            return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    async def delete_job(self, job_id: str) -> bool:
        async with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.work_dir.exists():
                    shutil.rmtree(job.work_dir)
                del self.jobs[job_id]
                return True
            return False

    async def cleanup_expired_jobs(self, retention_hours: int = 48):
        async with self._lock:
            expired_ids = [
                job_id
                for job_id, job in self.jobs.items()
                if job.is_expired(retention_hours)
            ]
            for job_id in expired_ids:
                job = self.jobs[job_id]
                if job.work_dir.exists():
                    shutil.rmtree(job.work_dir)
                del self.jobs[job_id]
            return len(expired_ids)

    def list_jobs(self) -> list[str]:
        return list(self.jobs.keys())
