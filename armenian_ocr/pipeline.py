"""The OCR pipeline: region layout analysis + recognition.

Thin orchestrator over two pluggable components (see
`armenian_ocr.types`): a `LayoutEngine` producing text regions in reading
order, and a `Recognizer` producing a `Paragraph` (lines + words) per
region.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import time
from typing import List, Optional, Union

import numpy as np

from armenian_ocr.types import LayoutEngine, Page, Recognizer

logger = logging.getLogger(__name__)


class OcrPipeline:
    def __init__(
        self,
        layout_engine: Optional[LayoutEngine] = None,
        recognizer: Optional[Recognizer] = None,
        *,
        device: str = "cpu",
        dpi: int = 300,
        max_workers: Optional[int] = None,
    ):
        if layout_engine is None:
            from armenian_ocr.layout_yolo import YoloLayoutEngine

            layout_engine = YoloLayoutEngine(device=device)
        if recognizer is None:
            from armenian_ocr import models
            from armenian_ocr.recognition.tesseract import TesseractRecognizer

            recognizer = TesseractRecognizer(
                tessdata_dir=models.get_tessdata_dir(),
                dpi=dpi,
                max_workers=max_workers,
            )

        self.layout_engine = layout_engine
        self.recognizer = recognizer

    def process_image(
        self, image: np.ndarray, dpi: Optional[int] = None
    ) -> Page:
        """Run layout analysis + recognition on an RGB uint8 page image."""
        height, width = image.shape[:2]

        start = time()
        regions = self.layout_engine.analyze(image)
        logger.info(
            "layout: %.2fs, %d regions", time() - start, len(regions)
        )

        start = time()
        paragraphs = self.recognizer.recognize(image, regions)
        logger.info(
            "recognition: %.2fs, %d lines",
            time() - start,
            sum(len(p.lines) for p in paragraphs),
        )

        # keep only paragraphs that produced text
        paragraphs = [p for p in paragraphs if p.lines]
        return Page(
            width=width, height=height, paragraphs=paragraphs, dpi=dpi
        )

    def process_document(
        self, path: Union[str, Path], dpi: int = 300
    ) -> List[Page]:
        """OCR an image or PDF file; returns one Page per page."""
        from armenian_ocr.documents import iter_pages

        pages = []
        for index, (image, page_dpi) in enumerate(iter_pages(path, dpi=dpi)):
            logger.info("processing page %d", index + 1)
            pages.append(self.process_image(image, dpi=page_dpi))
        return pages
