"""App configuration — every limit is overridable by environment variable."""

from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Layout engine + reading order for the OCR flow. The CLI requires an explicit
# --layout on every run; a long-running service instead resolves it here so it
# stays bootable (default `yolo`, the current behaviour). Set ARMENIAN_OCR_LAYOUT
# = yolo|paddle (and optionally ARMENIAN_OCR_READING_ORDER = xycut|native) to
# change it at deploy time. Layout *comparison* ignores these (it runs all).
LAYOUT_ENGINE = os.environ.get("ARMENIAN_OCR_LAYOUT", "yolo")
READING_ORDER = os.environ.get("ARMENIAN_OCR_READING_ORDER", "xycut")

MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 30)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 20)
MAX_QUEUE = _int("MAX_QUEUE", 5)
PAGE_TIMEOUT_S = _int("PAGE_TIMEOUT_S", 300)
JOB_TTL_S = _int("JOB_TTL_S", 1800)
PDF_DPI = _int("PDF_DPI", 200)
PREVIEW_MAX_WIDTH = _int("PREVIEW_MAX_WIDTH", 1400)
MAX_IMAGE_SIDE = _int("MAX_IMAGE_SIDE", 4000)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}
