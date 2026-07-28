"""Reading-order estimation via a recursive X-Y cut on region boxes.

The projection-based breakpoint heuristic (`moving_average`,
`get_horizontal_breakpoints`, `get_vertical_breakpoints`) is drawn from
Portmind's Armenian OCR project (where it was validated on word boxes) and
updated here. The same detectors are applied recursively to a filled mask of
the layout regions: the page is
cut into horizontal bands (top to bottom), and any band that admits no
horizontal cut is cut into columns (left to right); each resulting piece
is recursively re-cut. This X-Y-cut recursion correctly handles a
full-width masthead sitting above a multi-column body and stacked
sub-blocks inside a single column, which a fixed two-level cut does not.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.signal import find_peaks

from armenian_ocr.types import Box, Region

# Non-body "furniture" (page numbers, running headers/footers, side notes)
# read after the main content. DocLayout-YOLO tags these `abandon` (matched by
# substring below); PP-DocLayoutV3 uses explicit class names, listed here so
# the same X-Y-cut rule sinks them without relabelling. A detector that emits
# none of these (e.g. YOLO) is unaffected.
FURNITURE_LABELS = frozenset(
    {"header", "footer", "page_number", "aside_text"}
)


# --------------------------------------------------------------------------
# Projection-based breakpoint detectors (drawn from Portmind's armenian-ocr,
# updated here).
# --------------------------------------------------------------------------
def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate moving average values

    Args:
        values: Input values
        window_size: Window size

    Returns:
        Moving average values
    """
    return np.convolve(values, np.ones(window_size), "same") / window_size


def get_horizontal_breakpoints(
    word_box_heatmap: np.ndarray, window_size: int = 10
) -> List[int]:
    """Find horizontal breakpoints using detected boxes.

    Args:
        word_box_heatmap: Image with colored detection boxes
        window_size: Moving average window size (used for smoothing the amounts of pixels covered by each horizontal
            line)

    Returns:
        Horizontal breakpoints
    """
    horizontal_whites = np.where(
        moving_average(
            (word_box_heatmap != 255).sum(axis=1), window_size
        ).astype(int)
        == 0
    )[0]
    y_breakpoints, window = [], []

    for index in horizontal_whites:
        if len(window) == 0:
            window.append(index)
        else:
            if window[-1] != index - 1:
                y_breakpoints.append(window[len(window) // 2])
                window = [index]
            else:
                window.append(index)
    if len(window) == 0:
        return []
    else:
        y_breakpoints.append(window[len(window) // 2])
    return y_breakpoints


def get_vertical_breakpoints(
    word_box_heatmap: np.ndarray, divisor: int = 4, window_size: int = 100
) -> np.ndarray:
    """Find vertical breakpoints using detected boxes

    Args:
        word_box_heatmap: Image with colored detection boxes
        divisor: What fraction change compared to image height should be considered as breakpoint
            (for example, if divisor is 4 if a prominence of image height / 4 occurs a break line will be added)
        window_size: Moving average window size (used for smoothing the amounts of pixels covered by each vertical
            line)

    Returns:
        Vertical breakpoints
    """
    values = (word_box_heatmap != 255).sum(axis=0)
    moving_averages = moving_average(
        values=values, window_size=window_size
    )  # to make smoother
    breakpoints = find_peaks(
        -moving_averages, prominence=word_box_heatmap.shape[0] // divisor
    )[0]
    return breakpoints


# --------------------------------------------------------------------------
# Recursive X-Y cut over region boxes.
# --------------------------------------------------------------------------
def _center(box: Box) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _sort_by_position(regions: List[Region]) -> List[Region]:
    """Base-case order: top-to-bottom then left-to-right by center."""
    return sorted(regions, key=lambda r: (_center(r.box)[1], _center(r.box)[0]))


def _split_by_edges(
    regions: List[Region], edges: List[int], axis: int
) -> List[List[Region]]:
    """Partition regions into the slices delimited by `edges` (which must
    start at the sub-rectangle's low bound and end past its high bound),
    assigning each region to a slice by its center on `axis`
    (0 = x / columns, 1 = y / bands). Empty slices are dropped.
    """
    groups: List[List[Region]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        group = [
            region
            for region in regions
            if low <= _center(region.box)[axis] < high
        ]
        if group:
            groups.append(group)
    return groups


def _xy_cut(
    regions: List[Region],
    mask: np.ndarray,
    rect: Tuple[int, int, int, int],
) -> List[Region]:
    """Recursively order the regions whose centers fall inside `rect`.

    `rect` is (x0, y0, x1, y1) in absolute image coordinates; `mask` is the
    full-page uint8 mask (255 background, 0 on region boxes). Breakpoint
    detectors run on the sub-mask `mask[y0:y1, x0:x1]`, so cuts are found
    relative to the current window and shifted back to absolute.
    """
    if len(regions) <= 1:
        return list(regions)

    x0, y0, x1, y1 = rect
    sub = mask[y0:y1, x0:x1]
    sub_height, sub_width = sub.shape[:2]
    if sub_height <= 0 or sub_width <= 0:
        return _sort_by_position(regions)

    # 1) Try to split into horizontal bands (top to bottom).
    h_breaks = get_horizontal_breakpoints(
        sub, window_size=max(5, sub_height // 200)
    )
    interior_h = sorted(
        y0 + int(b) for b in h_breaks if 0 < int(b) < sub_height
    )
    if interior_h:
        band_edges = [y0, *interior_h, y1]
        bands = _split_by_edges(regions, band_edges, axis=1)
        # A genuine split reduces the group; otherwise fall through so we
        # don't recurse forever on the same set.
        if len(bands) > 1:
            ordered: List[Region] = []
            for band in bands:  # already top to bottom
                by0 = min(r.box[1] for r in band)
                by1 = max(r.box[3] for r in band)
                band_rect = (x0, max(y0, by0), x1, min(y1, by1))
                ordered.extend(_xy_cut(band, mask, band_rect))
            return ordered

    # 2) No usable horizontal cut: try to split into columns (left to right).
    v_breaks = get_vertical_breakpoints(
        sub, divisor=4, window_size=max(15, sub_width // 60)
    )
    interior_v = sorted(
        x0 + int(b) for b in v_breaks if 0 < int(b) < sub_width
    )
    if interior_v:
        col_edges = [x0, *interior_v, x1 + 1]
        columns = _split_by_edges(regions, col_edges, axis=0)
        if len(columns) > 1:
            ordered = []
            for column in columns:  # already left to right
                cx0 = min(r.box[0] for r in column)
                cx1 = max(r.box[2] for r in column)
                col_rect = (max(x0, cx0), y0, min(x1, cx1), y1)
                ordered.extend(_xy_cut(column, mask, col_rect))
            return ordered

    # 3) No cut in either direction: order by position.
    return _sort_by_position(regions)


def order_regions(
    regions: List[Region],
    image_shape: Tuple[int, ...],
    furniture_labels: Iterable[str] = FURNITURE_LABELS,
) -> List[Region]:
    """Order regions for reading with a recursive X-Y cut.

    Builds a mask of the region boxes and recursively cuts it into
    horizontal bands (top to bottom); a band with no horizontal cut is cut
    into columns (left to right), and every piece is re-cut. Reuses the
    projection-based breakpoint detectors so the strategy matches the one
    validated on word boxes — only the granularity (regions) differs, which
    keeps it robust to the loosely-overlapping boxes DocLayout-YOLO
    produces (where a plain empty-gap cut fails).

    Regions whose label marks them as non-body furniture — the label contains
    "abandon" (DocLayout-YOLO) or is in ``furniture_labels`` (PP-DocLayoutV3's
    header / footer / page_number / aside_text) — are moved to the end, since
    they are read after the main content.
    """
    if len(regions) <= 1:
        return list(regions)

    height, width = image_shape[0], image_shape[1]
    mask = np.full((height, width), 255, dtype=np.uint8)
    for region in regions:
        rx0, ry0, rx1, ry1 = region.box
        mask[ry0:ry1, rx0:rx1] = 0

    ordered = _xy_cut(list(regions), mask, (0, 0, width, height))

    # Push running headers/footers and page numbers to the end. Only do
    # this when it is unambiguous (some but not all regions are furniture),
    # so a page of pure body text is left exactly as ordered.
    furniture_set = frozenset(furniture_labels)
    is_furniture = [
        ("abandon" in r.label.lower()) or (r.label in furniture_set)
        for r in ordered
    ]
    if any(is_furniture) and not all(is_furniture):
        body = [r for r, furniture in zip(ordered, is_furniture) if not furniture]
        furniture = [r for r, f in zip(ordered, is_furniture) if f]
        ordered = body + furniture

    return ordered


# --------------------------------------------------------------------------
# Reading-order strategies (selectable / comparable).
# --------------------------------------------------------------------------
def _xycut(
    regions: List[Region], image_shape: Tuple[int, ...]
) -> List[Region]:
    """Recursive X-Y cut (the default; drawn from Portmind, updated). Works on
    any regions."""
    return order_regions(regions, image_shape)


def _native(
    regions: List[Region], image_shape: Tuple[int, ...]
) -> List[Region]:
    """Identity: keep the detector's own (native) reading order.

    Only meaningful for a detector that emits regions already ordered (e.g.
    PP-DocLayoutV3's pointer network); for others it is a no-op.
    """
    return list(regions)


# name -> strategy(regions, image_shape) -> ordered regions.
# Any LayoutEngine / the comparison view resolves an order by name here.
READING_ORDERS = {"xycut": _xycut, "native": _native}
"""Selectable reading-order strategies, keyed by CLI/app name."""
