"""OCR-free layout comparison: run several detectors × reading orders.

Region detection and reading order are cheap next to recognition, so this
runs them **without any OCR** and returns one candidate per
``(detector, reading-order)`` pairing — for eyeballing which detector segments
Armenian best and which reading order threads the page correctly. Each
candidate carries its ordered regions (for a canvas overlay in the app / JSON)
and, optionally, a rendered overlay image (for the CLI).

`native` reading order is only meaningful for a detector that emits regions in
its own order (PP-DocLayoutV3), so it is skipped for YOLO. A detector that
cannot be built (e.g. paddle not installed) yields a single candidate carrying
the error instead of raising, so a partial comparison still succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from armenian_ocr.reading_order import READING_ORDERS
from armenian_ocr.types import Region
from armenian_ocr.visualize import draw_regions

# Reading orders that make sense per detector. YOLO has no native order, so
# only the X-Y cut applies; PP-DocLayoutV3 also offers its own (native) order.
DETECTOR_ORDERS: Dict[str, Sequence[str]] = {
    "yolo": ("xycut",),
    "paddle": ("xycut", "native"),
}


@dataclass
class LayoutCandidate:
    """One (detector, reading-order) result of a layout comparison."""

    name: str  # "<detector>.<order>", e.g. "paddle.native"
    detector: str
    order: str
    regions: List[Region] = field(default_factory=list)
    overlay: Optional[np.ndarray] = None  # RGB, only when overlay=True
    error: Optional[str] = None  # set when the detector could not run

    def to_dict(self) -> Dict[str, Any]:
        """JSON-able form for the app (boxes + labels in reading order)."""
        return {
            "name": self.name,
            "detector": self.detector,
            "order": self.order,
            "error": self.error,
            "regions": [
                {
                    "box": list(region.box),
                    "label": region.label,
                    **(
                        {"poly": [list(pt) for pt in region.poly]}
                        if region.poly
                        else {}
                    ),
                    **(
                        {"score": region.score}
                        if region.score is not None
                        else {}
                    ),
                }
                for region in self.regions
            ],
        }


def _build_detector(name: str, device: str = "cpu", *, conf: Optional[float] = None):
    """Construct a detector exposing ``detect(image) -> List[Region]``.

    Imports are local so this module needs neither torch nor paddle at import
    time. ``conf`` overrides the detector's confidence threshold when given.
    Raises on an unknown name or a missing backend (caught by the caller).
    """
    if name == "yolo":
        from armenian_ocr.layout_yolo import YoloLayoutEngine

        kwargs = {"device": device}
        if conf is not None:
            kwargs["conf"] = conf
        return YoloLayoutEngine(**kwargs)
    if name == "paddle":
        from armenian_ocr.layout_paddle import PaddleDocLayoutDetector

        kwargs = {"device": device}
        if conf is not None:
            kwargs["conf"] = conf
        return PaddleDocLayoutDetector(**kwargs)
    raise ValueError(f"unknown detector {name!r}")


def compare_layouts(
    image: np.ndarray,
    detectors: Sequence[str] = ("yolo", "paddle"),
    orders: Sequence[str] = ("xycut", "native"),
    *,
    device: str = "cpu",
    overlay: bool = False,
    instances: Optional[Dict[str, Any]] = None,
    conf: Optional[float] = None,
) -> List[LayoutCandidate]:
    """Detect layout with each ``detectors`` and order it each ``orders`` way.

    Each detector runs **once**; every valid order is then applied to its
    regions (no re-detection, no OCR). ``instances`` maps a detector name to a
    pre-built object with ``.detect()`` (so the app can reuse cached engines);
    missing ones are built with ``device``. When ``conf`` is given, fresh
    detectors are built at that threshold (cached ``instances`` are ignored) —
    used for the low-threshold region preview that returns every candidate with
    its score. Returns candidates in ``detectors × orders`` order; a detector
    that fails contributes one candidate with ``error`` set (and no regions).
    """
    instances = instances or {}
    candidates: List[LayoutCandidate] = []

    for det_name in detectors:
        try:
            if conf is not None:
                detector = _build_detector(det_name, device, conf=conf)
            else:
                detector = instances.get(det_name) or _build_detector(
                    det_name, device
                )
            regions = detector.detect(image)
        except Exception as error:  # missing backend / model / bad name
            candidates.append(
                LayoutCandidate(
                    name=det_name,
                    detector=det_name,
                    order="",
                    error=str(error),
                )
            )
            continue

        valid = DETECTOR_ORDERS.get(det_name, ("xycut",))
        for order_name in orders:
            if order_name not in valid or order_name not in READING_ORDERS:
                continue
            ordered = READING_ORDERS[order_name](regions, image.shape)
            candidate = LayoutCandidate(
                name=f"{det_name}.{order_name}",
                detector=det_name,
                order=order_name,
                regions=ordered,
                overlay=draw_regions(image, ordered) if overlay else None,
            )
            candidates.append(candidate)

    return candidates
