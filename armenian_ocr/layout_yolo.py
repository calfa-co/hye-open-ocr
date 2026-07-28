"""DocLayout-YOLO layout engine: detect text regions, order them for reading.

Regions are handed to the recognizer without pre-segmented lines, so
Tesseract does its own line/word segmentation inside each block. Reading
order uses a recursive X-Y cut on the region rectangles (see
`armenian_ocr.reading_order`): horizontal bands top to bottom, then columns
left to right, recursively.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from armenian_ocr.preprocess import enhance_contrast
from armenian_ocr.reading_order import READING_ORDERS
from armenian_ocr.types import Box, Region

# DocStructBench classes that do NOT carry OCR-able text. Everything else
# is treated as text: on historical Armenian, YOLO frequently mislabels a
# slanted text block as `isolate_formula` or a dense column as `table`, so
# only pure images are skipped.
NON_TEXT_LABELS = frozenset({"figure"})


def _area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(a: Box, b: Box) -> int:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, ix) * max(0, iy)


def _drop_contained(
    regions: List[Region], threshold: float = 0.6
) -> List[Region]:
    """Drop a region mostly covered by a larger kept one.

    Prevents double OCR when the detector nests boxes (e.g. a whole-page
    block plus a small block for its first lines).
    """
    kept: List[Region] = []
    for region in sorted(regions, key=lambda r: _area(r.box), reverse=True):
        region_area = _area(region.box)
        if region_area == 0:
            continue
        if any(
            _intersection_area(region.box, keep.box) / region_area > threshold
            for keep in kept
        ):
            continue
        kept.append(region)
    # Return in the ORIGINAL input order: a detector's native order is the
    # `native` reading-order strategy, so it must survive de-duplication.
    # Only the kept *set* comes from the area-sorted pass above (YOLO re-sorts
    # via order_regions anyway, so its result is unchanged).
    kept_ids = {id(region) for region in kept}
    return [region for region in regions if id(region) in kept_ids]


class YoloLayoutEngine:
    """Region detection with DocLayout-YOLO."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        *,
        conf: float = 0.2,
        imgsz: int = 1024,
        preprocess: bool = True,
        skip_labels: Sequence[str] = tuple(NON_TEXT_LABELS),
        reading_order: str = "xycut",
    ):
        if weights is None:
            from armenian_ocr import models

            weights = models.get_yolo_weights()

        from doclayout_yolo import YOLOv10

        self._model = YOLOv10(str(weights))
        self._names = self._model.names
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.preprocess = preprocess
        self.skip_labels = frozenset(skip_labels)
        # DocLayout-YOLO has no native order of its own, so only the X-Y cut is
        # meaningful; anything else (e.g. "native") falls back to it.
        self.reading_order = reading_order if reading_order == "xycut" else "xycut"

    def detect(self, image: np.ndarray) -> List[Region]:
        """Detect layout regions on the page — **without** reading order.

        Kept separate from `analyze()` so the reading-order strategy is
        swappable and the layout can be compared OCR-free (see
        `armenian_ocr.compare`). Regions come back in the detector's own
        order; `analyze()` applies the X-Y cut.
        """
        height, width = image.shape[:2]
        detection_input = (
            enhance_contrast(image) if self.preprocess else image
        )
        result = self._model.predict(
            detection_input,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]

        regions: List[Region] = []
        for xyxy, cls, score in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
        ):
            label = self._names[int(cls)]
            if label in self.skip_labels:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            box = (
                max(0, x1),
                max(0, y1),
                min(width, x2),
                min(height, y2),
            )
            if box[2] > box[0] and box[3] > box[1]:
                regions.append(
                    Region(
                        box=box, label=label, lines=None, score=float(score)
                    )
                )

        regions = _drop_contained(regions)

        if not regions:
            # nothing detected: hand the whole page to the recognizer and
            # let Tesseract do full-page segmentation (never worse than
            # plain Tesseract)
            return [Region(box=(0, 0, width, height), label="page")]

        return regions

    def analyze(self, image: np.ndarray) -> List[Region]:
        return READING_ORDERS[self.reading_order](self.detect(image), image.shape)
