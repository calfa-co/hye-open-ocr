"""Model download, pipeline singleton and the per-job OCR worker."""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from app import config
from app.jobs import Job

logger = logging.getLogger("app")

# Recognizer engines the app can serve. Tesseract is the always-available
# default; Paddle is built on demand and may be unavailable (no trained
# model / no paddlepaddle), in which case the job errors gracefully.
RECOGNIZERS = ("tesseract", "paddle")
DEFAULT_RECOGNIZER = "tesseract"

# Layout (region) detectors. Both are region-level; the OCR flow uses one
# (config.LAYOUT_ENGINE), the comparison flow runs several side by side.
LAYOUTS = ("yolo", "paddle")
DEFAULT_LAYOUT = (
    config.LAYOUT_ENGINE if config.LAYOUT_ENGINE in LAYOUTS else "yolo"
)

# Low detection threshold used by the layout comparison so every candidate
# region is returned with its score; the frontend confidence slider then filters
# them live (a preview of what a higher threshold would keep).
COMPARE_PREVIEW_CONF = 0.05

# Caches (all guarded by _lock):
#  _pipelines   (layout, reading_order, recognizer) -> OcrPipeline
#  _layout_engines (layout, reading_order)          -> LayoutEngine
#  _detectors   layout                               -> detector (.detect())
_pipelines: dict = {}
_layout_engines: dict = {}
_detectors: dict = {}
_lock = threading.Lock()
models_loaded = threading.Event()


def normalize_recognizer(name) -> str:
    name = (name or DEFAULT_RECOGNIZER).strip().lower()
    return name if name in RECOGNIZERS else DEFAULT_RECOGNIZER


def normalize_layout(name) -> str:
    name = (name or DEFAULT_LAYOUT).strip().lower()
    return name if name in LAYOUTS else DEFAULT_LAYOUT


def _reading_order_for(layout: str, override=None) -> str:
    """Reading-order strategy for the OCR flow.

    Uses ``override`` when the caller (a per-job request) supplies one, else the
    deploy-time ``config.READING_ORDER``. ``native`` is only meaningful for
    paddle; anything else falls back to the X-Y cut."""
    order = (override or config.READING_ORDER or "xycut").strip().lower()
    if layout == "paddle" and order in ("xycut", "native"):
        return order
    return "xycut"


def _build_layout_engine(layout: str, reading_order: str, conf=None):
    import torch

    torch.set_num_threads(max(1, (torch.get_num_threads() or 2)))
    if layout == "paddle":
        from armenian_ocr.layout_paddle import PaddleDocLayoutEngine

        kwargs = {"reading_order": reading_order}
        if conf is not None:
            kwargs["conf"] = conf
        return PaddleDocLayoutEngine(**kwargs)

    from armenian_ocr.layout_yolo import YoloLayoutEngine

    logger.info("downloading / loading DocLayout-YOLO…")
    kwargs = {"reading_order": reading_order}
    if conf is not None:
        kwargs["conf"] = conf
    return YoloLayoutEngine(**kwargs)


def _get_layout_engine(layout: str, reading_order: str, conf=None):
    """Cached layout engine (called under _lock)."""
    key = (layout, reading_order, conf)
    engine = _layout_engines.get(key)
    if engine is None:
        engine = _build_layout_engine(layout, reading_order, conf)
        _layout_engines[key] = engine
    return engine


def _get_detector(layout: str):
    """Cached detector exposing ``detect()`` for the comparison flow (called
    under _lock). May raise RuntimeError if a backend is missing."""
    detector = _detectors.get(layout)
    if detector is None:
        if layout == "paddle":
            from armenian_ocr.layout_paddle import PaddleDocLayoutDetector

            detector = PaddleDocLayoutDetector()
        else:
            from armenian_ocr.layout_yolo import YoloLayoutEngine

            detector = YoloLayoutEngine()
        _detectors[layout] = detector
    return detector


def _build_recognizer(name: str, region_psm=None):
    if name == "paddle":
        # raises RuntimeError with a clear message when unavailable
        from armenian_ocr.recognition.paddle import PaddleRecognizer

        return PaddleRecognizer(block_pad=8)

    from armenian_ocr import models
    from armenian_ocr.recognition.tesseract import TesseractRecognizer

    kwargs = {
        "tessdata_dir": models.get_tessdata_dir(),
        "timeout": config.PAGE_TIMEOUT_S,
    }
    if region_psm is not None:
        kwargs["region_psm"] = region_psm
    return TesseractRecognizer(**kwargs)


def get_pipeline(
    recognizer: str = DEFAULT_RECOGNIZER,
    layout=None,
    conf=None,
    region_psm=None,
    reading_order=None,
):
    """Return (and cache) the pipeline for ``(layout, recognizer)``.

    ``conf`` overrides the layout detector's confidence threshold,
    ``reading_order`` the reading-order strategy (``xycut``/``native``; the
    latter paddle-only), and ``region_psm`` the Tesseract block segmentation
    mode; all are folded into the cache key so a tuned run does not clobber the
    default pipeline.

    Building a Paddle recognizer or the Paddle layout engine may raise
    ``RuntimeError`` when unavailable; the job worker surfaces that as a job
    error.
    """
    recognizer = normalize_recognizer(recognizer)
    layout = normalize_layout(layout)
    reading_order = _reading_order_for(layout, reading_order)
    # psm only affects the Tesseract recognizer
    if recognizer != "tesseract":
        region_psm = None
    key = (layout, reading_order, recognizer, conf, region_psm)
    with _lock:
        pipeline = _pipelines.get(key)
        if pipeline is None:
            from armenian_ocr.pipeline import OcrPipeline

            engine = _get_layout_engine(layout, reading_order, conf)
            pipeline = OcrPipeline(
                layout_engine=engine,
                recognizer=_build_recognizer(recognizer, region_psm),
            )
            _pipelines[key] = pipeline
            if recognizer == DEFAULT_RECOGNIZER and layout == DEFAULT_LAYOUT:
                models_loaded.set()
            logger.info(
                "pipeline ready: layout=%s order=%s recognizer=%s",
                layout,
                reading_order,
                recognizer,
            )
        return pipeline


def compare_page(image, detectors=LAYOUTS, orders=("xycut", "native"), conf=None):
    """Run the OCR-free layout comparison.

    With the default ``conf`` this reuses the cached detectors (their default
    thresholds). When ``conf`` is given (the low-threshold region preview),
    fresh detectors are built at that threshold so every candidate region comes
    back with its score for client-side filtering."""
    from armenian_ocr.compare import compare_layouts

    if conf is not None:
        return compare_layouts(
            image, detectors=detectors, orders=orders, conf=conf
        )
    instances: dict = {}
    with _lock:
        for name in detectors:
            try:
                instances[name] = _get_detector(name)
            except RuntimeError:
                pass  # unavailable -> compare_layouts records the error
    return compare_layouts(
        image, detectors=detectors, orders=orders, instances=instances
    )


def paddle_available() -> bool:
    """Whether the Paddle recognizer can be built.

    The paddle-calfa-tiny model auto-downloads (github.com/calfa-co/hye-paddle),
    so availability just requires the paddle backend to be importable. Cheap
    best-effort probe for the frontend — it does not instantiate PaddleOCR."""
    import importlib.util

    if importlib.util.find_spec("paddleocr") is None:
        return False
    if importlib.util.find_spec("paddle") is None:
        return False
    return True


def layout_available(layout: str) -> bool:
    """Whether a layout detector's backend is importable (for /api/health)."""
    import importlib.util

    if layout == "paddle":
        # PP-DocLayoutV3 runs through PaddleX's layout_analysis predictor,
        # which loads the model via transformers (torch).
        return all(
            importlib.util.find_spec(pkg) is not None
            for pkg in ("paddlex", "transformers")
        )
    # yolo
    return importlib.util.find_spec("doclayout_yolo") is not None


def warmup_async() -> None:
    def _warm() -> None:
        try:
            get_pipeline(DEFAULT_RECOGNIZER, DEFAULT_LAYOUT)
        except Exception as error:  # missing backend for a paddle default
            logger.warning("warmup skipped (%s)", error)

    threading.Thread(target=_warm, daemon=True).start()


def _downscale(image: np.ndarray) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= config.MAX_IMAGE_SIDE:
        return image
    scale = config.MAX_IMAGE_SIDE / longest
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _save_preview(image: np.ndarray, path: Path) -> None:
    import cv2

    height, width = image.shape[:2]
    if width > config.PREVIEW_MAX_WIDTH:
        scale = config.PREVIEW_MAX_WIDTH / width
        image = cv2.resize(
            image,
            (config.PREVIEW_MAX_WIDTH, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _run_compare_job(job: Job) -> None:
    """OCR-free layout comparison: store per-page candidates as JSON."""
    from armenian_ocr.documents import iter_pages

    try:
        job.status = "processing"
        source = next(job.directory.glob("input.*"))
        for index, (image, _dpi) in enumerate(
            iter_pages(source, dpi=config.PDF_DPI)
        ):
            start = time.time()
            image = _downscale(image)
            np.save(job.directory / f"page_{index}.npy", image)
            _save_preview(image, job.directory / f"page_{index}.png")

            # detect at a low threshold so every candidate region comes back
            # with its score; the client filters live with the confidence
            # slider to preview what a higher threshold would keep/drop.
            candidates = compare_page(image, conf=COMPARE_PREVIEW_CONF)
            payload = {
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "candidates": [c.to_dict() for c in candidates],
            }
            (job.directory / f"layout_{index}.json").write_text(
                json.dumps(payload)
            )
            job.page_sizes.append(
                {
                    "index": index,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
            )
            job.done_pages = index + 1
            logger.info(
                "compare job %s: page %d/%d in %.1fs",
                job.id,
                index + 1,
                job.total_pages,
                time.time() - start,
            )
        job.status = "done"
    except Exception as error:  # noqa: BLE001 — terminal state must be set
        logger.error(
            "compare job %s failed: %s", job.id, traceback.format_exc()
        )
        job.error = {"code": "processing_failed", "message": str(error)}
        job.status = "error"
    finally:
        job.finished_at = time.time()


def run_job(job: Job) -> None:
    """Executed in the job queue thread; always reaches a terminal state."""
    if getattr(job, "mode", "ocr") == "compare":
        _run_compare_job(job)
        return

    from armenian_ocr.documents import iter_pages

    try:
        recognizer = getattr(job, "recognizer", DEFAULT_RECOGNIZER)
        layout = getattr(job, "layout", None)
        conf = getattr(job, "conf", None)
        region_psm = getattr(job, "region_psm", None)
        reading_order = getattr(job, "reading_order", None)
        try:
            pipeline = get_pipeline(
                recognizer, layout, conf, region_psm, reading_order
            )
        except RuntimeError as error:
            # e.g. Paddle selected but no trained model / no backend
            job.error = {"code": "engine_unavailable", "message": str(error)}
            job.status = "error"
            return
        job.status = "processing"

        source = next(job.directory.glob("input.*"))
        pages = []
        for index, (image, page_dpi) in enumerate(
            iter_pages(source, dpi=config.PDF_DPI)
        ):
            start = time.time()
            image = _downscale(image)

            # full-resolution page image kept on disk for the pdf export
            np.save(job.directory / f"page_{index}.npy", image)
            _save_preview(image, job.directory / f"page_{index}.png")

            page = pipeline.process_image(image, dpi=page_dpi)
            pages.append(page)
            job.page_sizes.append(
                {
                    "index": index,
                    "width": page.width,
                    "height": page.height,
                }
            )
            job.done_pages = index + 1
            logger.info(
                "job %s: page %d/%d in %.1fs",
                job.id,
                index + 1,
                job.total_pages,
                time.time() - start,
            )

        job.pages = pages
        job.status = "done"
    except Exception as error:  # noqa: BLE001 — terminal state must be set
        logger.error("job %s failed: %s", job.id, traceback.format_exc())
        job.error = {"code": "processing_failed", "message": str(error)}
        job.status = "error"
    finally:
        job.finished_at = time.time()


def load_page_image(job: Job, index: int) -> np.ndarray:
    return np.load(job.directory / f"page_{index}.npy")
