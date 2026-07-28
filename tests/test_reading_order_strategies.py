"""Reading-order strategy registry + PP-DocLayoutV3 furniture handling.

Synthetic `Region` objects, no models. Covers the `xycut` / `native`
strategies and that PP-DocLayoutV3's furniture labels (header / footer /
page_number / aside_text) sink to the end under the X-Y cut without being
relabelled.
"""

from armenian_ocr.reading_order import READING_ORDERS, order_regions
from armenian_ocr.types import Region


def region(x1, y1, x2, y2, label="text"):
    return Region(box=(x1, y1, x2, y2), label=label)


def test_registry_exposes_xycut_and_native():
    assert {"xycut", "native"} <= set(READING_ORDERS)


def test_native_is_identity_copy():
    regions = [
        region(0, 300, 100, 400),
        region(0, 0, 100, 100),
        region(0, 150, 100, 250),
    ]
    ordered = READING_ORDERS["native"](regions, (500, 500))
    assert ordered == regions  # same order
    assert ordered is not regions  # but a fresh list


def test_xycut_reorders_top_to_bottom():
    regions = [
        region(0, 300, 100, 400),
        region(0, 0, 100, 100),
        region(0, 150, 100, 250),
    ]
    ordered = READING_ORDERS["xycut"](regions, (500, 500))
    tops = [r.box[1] for r in ordered]
    assert tops == sorted(tops)


def test_paddle_furniture_labels_sink_to_end():
    """header / page_number (PP-DocLayoutV3 furniture) read after the body."""
    header = region(400, 20, 600, 60, label="header")
    body = [region(100, 120, 900, 500), region(100, 520, 900, 900)]
    footer = region(450, 940, 550, 980, label="page_number")

    ordered = order_regions([header, *body, footer], (1000, 1000))

    body_positions = [ordered.index(r) for r in body]
    assert ordered.index(header) > max(body_positions)
    assert ordered.index(footer) > max(body_positions)
    # labels are preserved (not rewritten to "abandon_*")
    assert {r.label for r in ordered} == {"header", "text", "page_number"}


def test_yolo_abandon_substring_still_works():
    """The original DocLayout-YOLO furniture rule is unchanged."""
    header = region(400, 20, 600, 60, label="abandon")
    body = [region(100, 120, 900, 900)]
    ordered = order_regions([header, *body], (1000, 1000))
    assert ordered[-1] is header
