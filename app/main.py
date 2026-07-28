"""FastAPI app for the Armenian OCR HuggingFace Space.

Single-worker only: jobs are stored in process memory (see app/jobs.py).
"""

from __future__ import annotations

import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app import config, ocr_service
from app.jobs import store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "static"

MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "json": "application/json",
    "alto": "application/xml",
    "pdf": "application/pdf",
}
DOWNLOAD_EXTENSIONS = {
    "txt": ".txt",
    "json": ".json",
    "alto": ".xml",
    "pdf": ".pdf",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ocr_service.warmup_async()
    yield
    store.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Armenian OCR", lifespan=lifespan)


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code, detail={"code": code, "message": message}
    )


def _parse_conf(raw: str):
    """Optional layout confidence override, clamped to [0.05, 0.95]."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return max(0.05, min(0.95, float(raw)))
    except ValueError:
        return None


def _parse_psm(raw: str):
    """Optional Tesseract region PSM override, restricted to the modes we use."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        psm = int(raw)
    except ValueError:
        return None
    return psm if psm in (3, 4, 6) else None


def _parse_reading_order(raw: str, layout):
    """Optional reading-order override. `native` only applies to the paddle
    layout; anything else falls back to the X-Y cut (returns None → service
    default)."""
    raw = (raw or "").strip().lower()
    if raw == "native":
        return "native" if layout == "paddle" else "xycut"
    if raw == "xycut":
        return "xycut"
    return None


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/style.css")
async def style():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
async def script():
    return FileResponse(
        STATIC_DIR / "app.js", media_type="text/javascript"
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": ocr_service.models_loaded.is_set(),
        "engines": {
            "tesseract": {"available": True, "default": True},
            "paddle": {
                "available": ocr_service.paddle_available(),
                "default": False,
            },
        },
        "layouts": {
            name: {
                "available": ocr_service.layout_available(name),
                "default": ocr_service.DEFAULT_LAYOUT == name,
            }
            for name in ocr_service.LAYOUTS
        },
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile,
    recognizer: str = Form(default="tesseract"),
    layout: str = Form(default=""),
    mode: str = Form(default="ocr"),
    conf: str = Form(default=""),
    reading_order: str = Form(default=""),
    region_psm: str = Form(default=""),
):
    store.cleanup_expired()

    mode = "compare" if str(mode).strip().lower() == "compare" else "ocr"
    recognizer = ocr_service.normalize_recognizer(recognizer)
    layout = ocr_service.normalize_layout(layout) if layout.strip() else None
    conf_val = _parse_conf(conf)
    order_val = _parse_reading_order(reading_order, layout)
    psm_val = _parse_psm(region_psm)

    # OCR needs a working recognizer; compare mode runs no recognition.
    if (
        mode == "ocr"
        and recognizer == "paddle"
        and not ocr_service.paddle_available()
    ):
        raise _error(
            422,
            "recognizer_unavailable",
            "PaddleOCR backend not installed — install the paddle extra "
            "(pip install \"hye-open-ocr[paddle]\"). The paddle-calfa-tiny "
            "model itself is downloaded automatically.",
        )

    if store.queued_count() >= config.MAX_QUEUE:
        raise _error(429, "queue_full", "Too many jobs queued, retry later.")

    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise _error(
            422,
            "unsupported_type",
            f"Unsupported file type '{suffix}'. Allowed: "
            + ", ".join(sorted(config.ALLOWED_EXTENSIONS)),
        )

    directory = Path(tempfile.mkdtemp(prefix="ocrjob_"))
    target = directory / f"input{suffix}"
    size = 0
    with target.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                target.unlink(missing_ok=True)
                directory.rmdir()
                raise _error(
                    413,
                    "too_large",
                    f"File exceeds {config.MAX_UPLOAD_MB} MB.",
                )
            output.write(chunk)

    # validate content and count pages
    try:
        if suffix == ".pdf":
            from armenian_ocr.documents import count_pdf_pages

            total_pages = count_pdf_pages(target)
            if total_pages > config.MAX_PDF_PAGES:
                raise _error(
                    422,
                    "too_many_pages",
                    f"PDF has {total_pages} pages, maximum is "
                    f"{config.MAX_PDF_PAGES}.",
                )
        else:
            from PIL import Image

            with Image.open(target) as probe:
                probe.verify()
            total_pages = 1
    except HTTPException:
        import shutil

        shutil.rmtree(directory, ignore_errors=True)
        raise
    except Exception:
        import shutil

        shutil.rmtree(directory, ignore_errors=True)
        raise _error(422, "invalid_file", "File could not be read.")

    job = store.create(
        original_name=file.filename or f"document{suffix}",
        directory=directory,
        total_pages=total_pages,
        recognizer=recognizer,
        layout=layout,
        mode=mode,
        conf=conf_val,
        reading_order=order_val,
        region_psm=psm_val,
    )
    store.executor.submit(ocr_service.run_job, job)
    logger.info(
        "job %s queued (%s, %d pages, mode=%s, layout=%s, recognizer=%s)",
        job.id,
        job.original_name,
        total_pages,
        mode,
        layout,
        recognizer,
    )

    return {
        "job_id": job.id,
        "total_pages": total_pages,
        "recognizer": recognizer,
        "layout": layout,
        "mode": mode,
        "queue_position": store.queue_position(job.id),
    }


def _get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise _error(404, "job_expired", "Unknown or expired job.")
    return job


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    store.cleanup_expired()
    job = _get_job(job_id)
    return job.public_state(queue_position=store.queue_position(job_id))


@app.get("/api/jobs/{job_id}/result")
async def job_result(job_id: str):
    job = _get_job(job_id)
    if job.status != "done":
        raise _error(409, "not_ready", f"Job is {job.status}.")

    from armenian_ocr.export import pages_to_dict

    return JSONResponse(pages_to_dict(job.pages))


@app.get("/api/jobs/{job_id}/layout")
async def job_layout(job_id: str):
    """Per-page layout-comparison candidates (compare-mode jobs only)."""
    job = _get_job(job_id)
    if job.status != "done":
        raise _error(409, "not_ready", f"Job is {job.status}.")
    pages = []
    for index in range(job.total_pages):
        path = job.directory / f"layout_{index}.json"
        if path.exists():
            pages.append(json.loads(path.read_text()))
    return JSONResponse({"pages": pages})


@app.get("/api/jobs/{job_id}/pages/{index}/image")
async def page_image(job_id: str, index: int):
    job = _get_job(job_id)
    path = job.directory / f"page_{index}.png"
    if not path.exists():
        raise _error(404, "page_not_found", "Page image not available.")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/jobs/{job_id}/download/{format_name}")
async def download(job_id: str, format_name: str):
    job = _get_job(job_id)
    if job.status != "done":
        raise _error(409, "not_ready", f"Job is {job.status}.")
    if format_name not in MEDIA_TYPES:
        raise _error(422, "unknown_format", f"Unknown format {format_name}.")

    extension = DOWNLOAD_EXTENSIONS[format_name]
    target = job.directory / f"export{extension}"
    if not target.exists():  # lazy export, cached in the job directory
        from armenian_ocr.export import export_pages

        images = None
        if format_name == "pdf":
            images = [
                ocr_service.load_page_image(job, index)
                for index in range(job.total_pages)
            ]
        export_pages(
            job.pages,
            [format_name],
            job.directory,
            stem="export",
            images=images,
            source_name=job.original_name,
        )

    stem = Path(job.original_name).stem or "document"
    return FileResponse(
        target,
        media_type=MEDIA_TYPES[format_name],
        filename=f"{stem}{extension}",
    )


@app.delete("/api/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    job = store.get(job_id)
    if job is not None and job.status in ("queued", "processing"):
        # the worker thread may still be using the files; let TTL clean it
        job.finished_at = job.created_at
        return
    store.delete(job_id)
