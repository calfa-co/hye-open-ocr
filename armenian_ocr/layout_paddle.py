"""PP-DocLayoutV3 layout engine: an alternative region detector to YOLO.

PP-DocLayoutV3 is a **region/block** detector (same granularity as
DocLayout-YOLO): page in, labelled block boxes out — text blocks, titles,
tables, figures, headers, footers… It does **not** find text lines (those come
from the recognizer's own text-detection model). So this is a drop-in
alternative to `YoloLayoutEngine`, and the two can be compared side by side
(see `armenian_ocr.compare`).

Unlike a plain object detector, PP-DocLayoutV3 predicts **tight multi-point
polygons** and a **native reading order** in a single pass, which is what makes
it robust on poor-quality printed scans (skew, curvature near the binding,
lighting). We run it through PaddleX's ``layout_analysis`` predictor with
``layout_shape_mode="poly"`` (``paddlex.create_model("PP-DocLayoutV3")``,
CPU-runnable), keep the polygon on each ``Region`` (``Region.poly``) alongside
the axis-aligned ``box``, and order regions by the model's own reading order.

Detection and reading order are kept separate (mirroring `layout_yolo`):
`PaddleDocLayoutDetector.detect()` returns regions in the model's own order,
and `PaddleDocLayoutEngine.analyze()` applies a reading-order strategy
(`xycut` by default; `native` keeps PP-DocLayoutV3's own order). `paddlex` is
imported lazily, so importing this module never requires it; if it is missing,
construction raises a clear ``RuntimeError`` and the caller can stay on YOLO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from armenian_ocr._paddle_common import (
    silence_paddle_logs,
    to_paddle_device,
)
from armenian_ocr.layout_yolo import _drop_contained
from armenian_ocr.preprocess import enhance_contrast
from armenian_ocr.reading_order import READING_ORDERS
from armenian_ocr.types import Region

ENV_PADDLE_LAYOUT_DIR = "ARMENIAN_OCR_PADDLE_LAYOUT_DIR"

# PP-DocLayoutV3 classes that carry NO OCR-able body text and are dropped
# before recognition. Kept permissive (like YOLO, which skips only `figure`):
# on degraded scans dense text blocks get mislabelled table/formula/etc., so we
# only drop unambiguous non-text. Override via the ctor.
NON_TEXT_LABELS = frozenset(
    {"image", "figure", "chart", "seal", "header_image", "footer_image"}
)

_UNAVAILABLE = (
    "PP-DocLayoutV3 layout engine not available — install the paddle extra "
    "(pip install '.[paddle]') so PaddleX/transformers can load PP-DocLayoutV3, "
    "or set ARMENIAN_OCR_PADDLE_LAYOUT_DIR to a local model directory"
)


def _poly_to_points(
    polygon, width: int, height: int
) -> Optional[List[List[int]]]:
    """Clip a PP-DocLayoutV3 ``polygon_points`` array to int page coords."""
    if polygon is None:
        return None
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    return [
        [
            int(round(min(max(0.0, x), width))),
            int(round(min(max(0.0, y), height))),
        ]
        for x, y in points
    ]


class PaddleDocLayoutDetector:
    """Region detection with PP-DocLayoutV3 (PaddleX ``layout_analysis``).

    `detect()` returns regions in the model's own reading order (its ``order``
    field); it carries the tight polygon on ``Region.poly``. The ``native``
    strategy is therefore a no-op and ``xycut`` re-orders with our recursive
    cut on the axis-aligned boxes.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        *,
        model_name: str = "PP-DocLayoutV3",
        conf: float = 0.5,
        preprocess: bool = False,
        skip_labels: Sequence[str] = tuple(NON_TEXT_LABELS),
    ):
        model_dir = model_dir or os.environ.get(ENV_PADDLE_LAYOUT_DIR)

        try:
            from paddlex import create_model
        except Exception as error:  # paddlex / paddlepaddle missing
            raise RuntimeError(_UNAVAILABLE) from error
        silence_paddle_logs()

        kwargs = dict(model_name=model_name, device=to_paddle_device(device))
        if model_dir:
            kwargs["model_dir"] = str(model_dir)
        try:
            self._model = create_model(**kwargs)
        except Exception as error:  # backend broken / bad model dir
            raise RuntimeError(_UNAVAILABLE) from error

        self.device = device
        self.conf = conf
        self.preprocess = preprocess
        self.skip_labels = frozenset(skip_labels)

    def _predict(self, image: np.ndarray):
        # layout_shape_mode="poly" makes PP-DocLayoutV3 emit tight polygons;
        # pass our confidence as the detector threshold when supported.
        try:
            return list(
                self._model.predict(
                    image,
                    layout_shape_mode="poly",
                    threshold=self.conf,
                    batch_size=1,
                )
            )
        except TypeError:
            return list(
                self._model.predict(image, layout_shape_mode="poly")
            )

    def detect(self, image: np.ndarray) -> List[Region]:
        height, width = image.shape[:2]
        src = enhance_contrast(image) if self.preprocess else image
        raw = self._predict(src)

        ordered: List[tuple] = []  # (native order rank, Region)
        for result in raw:
            for box in result["boxes"]:
                label = box.get("label", "text")
                score = box.get("score")
                if score is not None and score < self.conf:
                    continue
                if label in self.skip_labels:
                    continue
                coord = box.get("coordinate")
                if coord is None or len(coord) < 4:
                    continue
                x1, y1, x2, y2 = (int(round(float(v))) for v in coord[:4])
                clipped = (
                    max(0, x1),
                    max(0, y1),
                    min(width, x2),
                    min(height, y2),
                )
                if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    continue
                poly = _poly_to_points(
                    box.get("polygon_points"), width, height
                )
                ordered.append(
                    (
                        box.get("order"),
                        Region(
                            box=clipped,
                            label=label,
                            poly=poly,
                            score=(
                                float(score) if score is not None else None
                            ),
                        ),
                    )
                )

        # native reading order = the model's own `order` (None sinks last)
        ordered.sort(key=lambda item: (item[0] is None, item[0] or 0))
        regions = _drop_contained([region for _, region in ordered])
        if not regions:
            return [Region(box=(0, 0, width, height), label="page")]
        return regions


class PaddleDocLayoutEngine:
    """`LayoutEngine` = PP-DocLayoutV3 detector + a reading-order strategy.

    `analyze()` returns regions in reading order (default `xycut` for parity
    with `YoloLayoutEngine`; pass `reading_order="native"` to keep
    PP-DocLayoutV3's own order). Extra keyword args go to the detector.
    """

    def __init__(
        self,
        detector: Optional[PaddleDocLayoutDetector] = None,
        *,
        reading_order: str = "xycut",
        **detector_kwargs,
    ):
        if reading_order not in READING_ORDERS:
            raise ValueError(
                f"unknown reading_order {reading_order!r}; "
                f"choose from {sorted(READING_ORDERS)}"
            )
        self.detector = detector or PaddleDocLayoutDetector(**detector_kwargs)
        self.reading_order = reading_order

    def detect(self, image: np.ndarray) -> List[Region]:
        return self.detector.detect(image)

    def analyze(self, image: np.ndarray) -> List[Region]:
        regions = self.detector.detect(image)
        return READING_ORDERS[self.reading_order](regions, image.shape)
