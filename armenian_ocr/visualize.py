"""Render a visual debug overlay of layout regions and reading order.

Draws each detected region as a coloured box (by layout label), a numbered
badge at its top-left giving the reading-order rank, and a path through
region centres showing the reading sequence. Layout-only, so it needs no
OCR and is fast enough to sweep a whole corpus.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from armenian_ocr.types import Region

# RGB colours per layout label (matched by substring, so 'plain text',
# 'figure_caption', etc. resolve sensibly)
_LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "title": (211, 47, 47),
    "text": (46, 125, 50),
    "caption": (21, 101, 192),
    "table": (239, 108, 0),
    "formula": (123, 31, 162),
    "figure": (120, 120, 120),
    "abandon": (150, 150, 150),
    "page": (0, 131, 143),
}
_DEFAULT_COLOR = (46, 125, 50)
_PATH_COLOR = (30, 30, 30)


def color_for(label: str) -> Tuple[int, int, int]:
    label = label.lower()
    for key, color in _LABEL_COLORS.items():
        if key in label:
            return color
    return _DEFAULT_COLOR


def draw_regions(
    image: np.ndarray,
    regions: Sequence[Region],
    *,
    show_order: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Return a copy of `image` (RGB) with the region overlay drawn.

    `regions` are assumed to be in reading order (as returned by a
    LayoutEngine); their index is the reading-order rank.
    """
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    scale = max(1.0, max(canvas.shape[:2]) / 1000.0)
    thickness = max(2, int(2 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale

    centers: List[Tuple[int, int]] = []
    for region in regions:
        x1, y1, x2, y2 = region.box
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        color = color_for(region.label)
        # a tight layout polygon (PP-DocLayoutV3) hugs skewed/curved blocks;
        # fall back to the axis-aligned box when there is none.
        poly = getattr(region, "poly", None)
        if poly and len(poly) >= 3:
            points = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], True, color, thickness)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if show_labels:
            _label(canvas, region.label, (x1, y1), color, font, font_scale,
                   thickness)

    if show_order and len(centers) > 1:
        for start, end in zip(centers[:-1], centers[1:]):
            cv2.arrowedLine(
                canvas, start, end, _PATH_COLOR, thickness,
                tipLength=0.02,
            )

    if show_order:
        for rank, (region, center) in enumerate(zip(regions, centers), 1):
            _badge(canvas, str(rank), region.box[:2], color_for(region.label),
                   font, font_scale, scale)

    return canvas


def _label(canvas, text, top_left, color, font, font_scale, thickness):
    x, y = top_left
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    y_text = max(th + 4, y - 4)
    cv2.rectangle(
        canvas, (x, y_text - th - 4), (x + tw + 6, y_text + 2), color, -1
    )
    cv2.putText(
        canvas, text, (x + 3, y_text - 2), font, font_scale,
        (255, 255, 255), max(1, thickness // 2), cv2.LINE_AA,
    )


def _badge(canvas, text, anchor, color, font, font_scale, scale):
    radius = int(16 * scale)
    cx, cy = anchor[0] + radius, anchor[1] + radius
    cv2.circle(canvas, (cx, cy), radius, color, -1)
    cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), max(1, int(scale)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    cv2.putText(
        canvas, text, (cx - tw // 2, cy + th // 2), font, font_scale,
        (255, 255, 255), max(1, int(2 * scale) // 2), cv2.LINE_AA,
    )
