"""OCR-free layout comparison — fake detectors, no models.

Covers the candidate set (native only pairs with paddle), that `native`
preserves the detector's own order while `xycut` reorders, the graceful
error candidate when a detector cannot run, and the JSON serialisation.
"""

import numpy as np

from armenian_ocr.compare import LayoutCandidate, compare_layouts
from armenian_ocr.types import Region


class FakeDetector:
    def __init__(self, regions):
        self._regions = regions

    def detect(self, image):
        return list(self._regions)


class BoomDetector:
    def detect(self, image):
        raise RuntimeError("paddle backend missing")


def test_candidate_set_and_native_only_for_paddle():
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    yolo = [Region(box=(0, 300, 100, 400)), Region(box=(0, 0, 100, 100))]
    paddle = [Region(box=(0, 0, 100, 100)), Region(box=(0, 300, 100, 400))]
    instances = {"yolo": FakeDetector(yolo), "paddle": FakeDetector(paddle)}

    candidates = compare_layouts(
        image,
        detectors=("yolo", "paddle"),
        orders=("xycut", "native"),
        instances=instances,
    )

    # yolo has no native order -> only xycut; paddle gets both
    assert [c.name for c in candidates] == [
        "yolo.xycut",
        "paddle.xycut",
        "paddle.native",
    ]

    native = next(c for c in candidates if c.name == "paddle.native")
    assert [r.box for r in native.regions] == [r.box for r in paddle]

    yolo_xycut = next(c for c in candidates if c.name == "yolo.xycut")
    tops = [r.box[1] for r in yolo_xycut.regions]
    assert tops == sorted(tops)  # xycut reordered top-to-bottom


def test_unavailable_detector_yields_error_candidate():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    instances = {
        "yolo": FakeDetector([Region(box=(0, 0, 50, 50))]),
        "paddle": BoomDetector(),
    }
    candidates = compare_layouts(
        image,
        detectors=("yolo", "paddle"),
        orders=("xycut",),
        instances=instances,
    )
    assert any(c.name == "yolo.xycut" and not c.error for c in candidates)
    errored = [c for c in candidates if c.error]
    assert len(errored) == 1
    assert errored[0].detector == "paddle"
    assert not errored[0].regions


def test_candidate_to_dict():
    candidate = LayoutCandidate(
        name="paddle.native",
        detector="paddle",
        order="native",
        regions=[
            Region(box=(1, 2, 3, 4), label="title"),
            Region(
                box=(5, 6, 9, 10),
                label="text",
                poly=[[5, 6], [9, 6], [9, 10], [5, 10]],
                score=0.87,
            ),
        ],
    )
    data = candidate.to_dict()
    assert data["name"] == "paddle.native"
    assert data["detector"] == "paddle"
    # a plain box has no "poly"/"score" key; a scored region with a polygon
    # carries both (score drives the client-side confidence preview)
    assert data["regions"][0] == {"box": [1, 2, 3, 4], "label": "title"}
    assert data["regions"][1]["poly"] == [[5, 6], [9, 6], [9, 10], [5, 10]]
    assert data["regions"][1]["score"] == 0.87
    assert "score" not in data["regions"][0]
