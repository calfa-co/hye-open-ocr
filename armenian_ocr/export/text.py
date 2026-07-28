"""Plain-text export in reading order."""

from __future__ import annotations

from typing import List

from armenian_ocr.types import Page

PAGE_SEPARATOR = "\f\n"  # form feed, as produced by pdftotext


def pages_to_text(pages: List[Page]) -> str:
    return PAGE_SEPARATOR.join(page.text for page in pages)
