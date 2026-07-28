"""In-memory job store and single-worker OCR queue.

IMPORTANT: the store lives in process memory — the app must run with a
single uvicorn worker (no --workers N). The one-thread executor doubles
as the job queue, guaranteeing a single OCR job at a time.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app import config


@dataclass
class Job:
    id: str
    original_name: str
    directory: Path
    total_pages: int
    recognizer: str = "tesseract"  # tesseract (default) | paddle
    layout: Optional[str] = None  # None -> service default (config.LAYOUT_ENGINE)
    mode: str = "ocr"  # ocr | compare (OCR-free layout comparison)
    conf: Optional[float] = None  # layout detector confidence override
    reading_order: Optional[str] = None  # xycut | native (paddle only) override
    region_psm: Optional[int] = None  # Tesseract block segmentation override
    status: str = "queued"  # queued | processing | done | error
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    done_pages: int = 0
    pages: Optional[list] = None  # list[Page] when done
    page_sizes: List[dict] = field(default_factory=list)
    error: Optional[dict] = None

    def public_state(self, queue_position: Optional[int] = None) -> dict:
        state = {
            "job_id": self.id,
            "status": self.status,
            "recognizer": self.recognizer,
            "layout": self.layout,
            "mode": self.mode,
            "progress": {
                "done_pages": self.done_pages,
                "total_pages": self.total_pages,
            },
            "error": self.error,
        }
        if queue_position is not None:
            state["queue_position"] = queue_position
        if self.status == "done":
            state["pages"] = self.page_sizes
        return state


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        # one worker == one OCR job at a time on the small CPU Space
        self.executor = ThreadPoolExecutor(max_workers=1)

    def create(
        self,
        original_name: str,
        directory: Path,
        total_pages: int,
        recognizer: str = "tesseract",
        layout: Optional[str] = None,
        mode: str = "ocr",
        conf: Optional[float] = None,
        reading_order: Optional[str] = None,
        region_psm: Optional[int] = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            original_name=original_name,
            directory=directory,
            total_pages=total_pages,
            recognizer=recognizer,
            layout=layout,
            mode=mode,
            conf=conf,
            reading_order=reading_order,
            region_psm=region_psm,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def queued_count(self) -> int:
        with self._lock:
            return sum(
                1
                for job in self._jobs.values()
                if job.status in ("queued", "processing")
            )

    def queue_position(self, job_id: str) -> Optional[int]:
        """0 = running or next; None when not queued."""
        with self._lock:
            queued = [
                job
                for job in sorted(
                    self._jobs.values(), key=lambda j: j.created_at
                )
                if job.status in ("queued", "processing")
            ]
        for position, job in enumerate(queued):
            if job.id == job_id:
                return position
        return None

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(job.directory, ignore_errors=True)
        return True

    def cleanup_expired(self) -> None:
        """Lazy TTL sweep, called on submit and poll."""
        now = time.time()
        with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if job.finished_at is not None
                and now - job.finished_at > config.JOB_TTL_S
            ]
            jobs = [self._jobs.pop(job_id) for job_id in expired]
        for job in jobs:
            shutil.rmtree(job.directory, ignore_errors=True)


store = JobStore()
