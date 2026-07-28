"""Core data structures and interfaces of the OCR pipeline.

The pipeline is deliberately modular: a layout engine produces the page
structure (paragraphs / lines / word boxes, in reading order) and a
recognizer turns line boxes into text. Swapping either implementation
only requires satisfying the corresponding protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

Box = Tuple[int, int, int, int]
"""Axis-aligned box in page pixel coordinates: (x1, y1, x2, y2)."""


@dataclass
class Word:
    box: Box
    text: str = ""
    confidence: Optional[float] = None  # 0-100 (Tesseract scale)
    # tight detection polygon in page coords (e.g. PaddleOCR `det` quad),
    # hugging skewed text; None when only an axis-aligned box is available.
    poly: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "box": list(self.box),
            "text": self.text,
            "confidence": self.confidence,
        }
        if self.poly is not None:
            data["poly"] = [list(point) for point in self.poly]
        return data


@dataclass
class Line:
    box: Box
    words: List[Word] = field(default_factory=list)
    text: str = ""
    confidence: Optional[float] = None
    # tight detection polygon in page coords (see Word.poly).
    poly: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "box": list(self.box),
            "text": self.text,
            "confidence": self.confidence,
            "words": [word.to_dict() for word in self.words],
        }
        if self.poly is not None:
            data["poly"] = [list(point) for point in self.poly]
        return data


@dataclass
class Paragraph:
    box: Box
    lines: List[Line] = field(default_factory=list)
    label: str = "text"  # layout class (e.g. from the detector)
    # tight layout polygon (page coords) when the detector emits one
    # (PP-DocLayoutV3); None when the region is a plain axis-aligned box.
    poly: Optional[List[List[int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "box": list(self.box),
            "label": self.label,
            "lines": [line.to_dict() for line in self.lines],
        }
        if self.poly is not None:
            data["poly"] = [list(point) for point in self.poly]
        return data


@dataclass
class Page:
    width: int
    height: int
    paragraphs: List[Paragraph] = field(default_factory=list)
    dpi: Optional[int] = None

    @property
    def lines(self) -> List[Line]:
        return [line for paragraph in self.paragraphs for line in paragraph.lines]

    @property
    def text(self) -> str:
        """Page text in reading order: lines joined by newlines, paragraphs
        separated by a blank line. Lines with no recognized text are skipped."""
        blocks = []
        for paragraph in self.paragraphs:
            texts = [line.text for line in paragraph.lines if line.text]
            if texts:
                blocks.append("\n".join(texts))
        return "\n\n".join(blocks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
        }


@dataclass
class Region:
    """A text-bearing area of the page, in reading order.

    `lines` holds pre-segmented line boxes when the layout engine does
    line segmentation itself (e.g. word-box grouping); it is None when
    the engine only finds blocks and delegates line/word segmentation to
    the recognizer (e.g. DocLayout-YOLO regions → Tesseract with psm 4).
    """

    box: Box
    label: str = "text"
    lines: Optional[List[Box]] = None
    # tight multi-point layout polygon in page coords (e.g. PP-DocLayoutV3),
    # hugging skewed/curved blocks; None when the region is a plain box. The
    # axis-aligned `box` is always kept for consumers that need a rectangle.
    poly: Optional[List[List[int]]] = None
    # detector confidence in [0, 1] when available (used for a client-side
    # confidence preview); None when the detector does not expose a score.
    score: Optional[float] = None


@runtime_checkable
class LayoutEngine(Protocol):
    """Turns a page image into text regions, in reading order."""

    def analyze(self, image: np.ndarray) -> List["Region"]:
        """Detect text regions on an RGB uint8 image, ordered for reading.

        A region may carry pre-segmented line boxes (`Region.lines`) or
        leave them to the recognizer (`Region.lines is None`).
        """
        ...


@runtime_checkable
class Recognizer(Protocol):
    """Recognizes the text of layout regions on a page image."""

    def recognize(
        self, image: np.ndarray, regions: Sequence["Region"]
    ) -> List[Paragraph]:
        """Return one Paragraph per region, in the same order.

        `image` is the full page (RGB uint8); implementations crop as
        needed. Lines and words carry page coordinates.
        """
        ...
