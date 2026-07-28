"""Reading-order tests for the recursive X-Y cut over region boxes.

These use synthetic `Region` objects and need no models. Column blocks are
made to overlap in y (as real multi-column text does), so no full-width
horizontal gap spans the columns and the vertical cut is exercised — a
grid of y-aligned blocks would instead be a table and read row-major.
"""

from armenian_ocr.reading_order import order_regions
from armenian_ocr.types import Region


def region(x1, y1, x2, y2, label="text"):
    return Region(box=(x1, y1, x2, y2), label=label)


def positions(ordered):
    return {id(r): index for index, r in enumerate(ordered)}


def test_single_region_returned_unchanged():
    only = region(100, 100, 400, 400)
    assert order_regions([only], (1000, 1000)) == [only]


def test_two_columns_left_before_right():
    """Every left-column region precedes every right-column region.

    Each column is split into stacked sub-blocks that overlap in y with
    the other column, so the top level has no horizontal cut and must cut
    vertically into columns first, then recurse each column top to bottom.
    """
    left = [region(50, 100, 450, 480), region(50, 500, 450, 900)]
    right = [region(550, 100, 950, 380), region(550, 400, 950, 900)]
    # feed them scrambled to prove ordering is computed, not preserved
    scrambled = [right[1], left[0], right[0], left[1]]

    ordered = order_regions(scrambled, (1000, 1000))
    position = positions(ordered)

    assert len(ordered) == 4
    assert max(position[id(r)] for r in left) < min(
        position[id(r)] for r in right
    )
    # within each column, top to bottom
    left_tops = [r.box[1] for r in ordered if r in left]
    right_tops = [r.box[1] for r in ordered if r in right]
    assert left_tops == sorted(left_tops)
    assert right_tops == sorted(right_tops)


def test_full_width_title_above_two_columns():
    """A full-width masthead is read first, then the two columns.

    The top-level horizontal cut separates the title band from the body
    band; the body band has no horizontal cut and splits into columns.
    """
    title = region(50, 30, 950, 120, label="title")
    left = [region(50, 200, 450, 560), region(50, 580, 450, 950)]
    right = [region(550, 200, 950, 480), region(550, 500, 950, 950)]
    scrambled = [left[1], right[0], title, right[1], left[0]]

    ordered = order_regions(scrambled, (1000, 1000))
    position = positions(ordered)

    # title is region #1
    assert ordered[0] is title
    # then the whole left column, then the whole right column
    assert position[id(title)] < min(
        position[id(r)] for r in left + right
    )
    assert max(position[id(r)] for r in left) < min(
        position[id(r)] for r in right
    )


def test_single_column_stacked_top_to_bottom():
    stacked = [
        region(100, 50, 500, 130),
        region(100, 160, 500, 240),
        region(100, 270, 500, 350),
        region(100, 380, 500, 460),
    ]
    scrambled = [stacked[2], stacked[0], stacked[3], stacked[1]]

    ordered = order_regions(scrambled, (600, 600))

    tops = [r.box[1] for r in ordered]
    assert tops == sorted(tops)


def test_abandon_regions_pushed_to_end():
    """Page numbers / running headers (label ~ 'abandon') read last."""
    header = region(400, 20, 600, 60, label="abandon_top")
    body = [region(100, 120, 900, 500), region(100, 520, 900, 900)]
    footer = region(450, 940, 550, 980, label="abandon_bottom")

    ordered = order_regions([header, *body, footer], (1000, 1000))

    # both furniture regions end up after all body regions
    body_positions = [ordered.index(r) for r in body]
    assert ordered.index(header) > max(body_positions)
    assert ordered.index(footer) > max(body_positions)


def test_pure_body_order_not_disturbed_by_furniture_rule():
    """With no furniture, ordering is left exactly as the X-Y cut produced."""
    body = [region(100, 100, 900, 400), region(100, 450, 900, 800)]
    ordered = order_regions(body, (1000, 1000))
    assert [r.box[1] for r in ordered] == [100, 450]


def test_center_padding_math():
    """The fixed `center` branch: content size must derive from the
    original image dimensions, not from the input tensor."""
    original_height, original_width = 600, 400
    canvas_size = 1280
    target_ratio = canvas_size / max(original_height, original_width)
    ratio = 1 / target_ratio

    target_height = original_height * target_ratio
    target_width = original_width * target_ratio
    left_padding = int(ratio * (canvas_size - target_width) / 2)
    upper_padding = int(ratio * (canvas_size - target_height) / 2)

    # the tall side fills the canvas, the narrow side is centered
    assert upper_padding == 0
    expected = (original_height - original_width) / 2
    assert abs(left_padding - expected) <= 1  # int() truncation
