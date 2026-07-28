"""PaddleDocLayoutDetector / PaddleDocLayoutEngine with a mocked paddlex.

No PaddlePaddle/transformers install and no model download: `paddlex` is faked
in sys.modules so `create_model(...).predict(...)` returns canned
PP-DocLayoutV3-shaped results (boxes with coordinate / order / polygon_points).
Covers box parsing/filtering/clipping, polygon capture, native reading-order
sort, the device cuda→gpu translation, the whole-page fallback, and the
missing-backend RuntimeError.
"""

import sys
import types
from unittest import mock

import numpy as np
import pytest

from armenian_ocr.types import Region


def _fake_paddlex(boxes, capture):
    module = types.ModuleType("paddlex")

    class _Model:
        def predict(self, image, **kwargs):
            capture["predict_kwargs"] = kwargs
            return [{"boxes": boxes}]

    def create_model(**kwargs):
        capture.update(kwargs)
        return _Model()

    module.create_model = create_model
    return module


def _detector(boxes, capture, **kwargs):
    module = _fake_paddlex(boxes, capture)
    with mock.patch.dict(sys.modules, {"paddlex": module}):
        from armenian_ocr.layout_paddle import PaddleDocLayoutDetector

        return PaddleDocLayoutDetector(**kwargs)


# order is deliberately out of geometric order to prove we sort by it
BOXES = [
    {"cls_id": 0, "label": "text", "score": 0.9,
     "coordinate": [50, 100, 450, 480], "order": 2,
     "polygon_points": [[50, 100], [450, 105], [448, 480], [52, 478]]},
    {"cls_id": 1, "label": "image", "score": 0.95,
     "coordinate": [500, 100, 900, 480], "order": 3,
     "polygon_points": [[500, 100], [900, 100], [900, 480], [500, 480]]},
    {"cls_id": 2, "label": "header", "score": 0.8,
     "coordinate": [400, 20, 600, 60], "order": 0,
     "polygon_points": [[400, 20], [600, 22], [599, 60], [401, 59]]},
    {"cls_id": 3, "label": "text", "score": 0.1,
     "coordinate": [0, 0, 10, 10], "order": 4,
     "polygon_points": [[0, 0], [10, 0], [10, 10], [0, 10]]},
    {"cls_id": 4, "label": "paragraph_title", "score": 0.85,
     "coordinate": [60, 70, 450, 95], "order": 1,
     "polygon_points": [[60, 70], [450, 71], [449, 95], [61, 94]]},
]


def test_detect_parses_filters_and_keeps_polygon():
    capture = {}
    detector = _detector(BOXES, capture, conf=0.5)
    regions = detector.detect(np.zeros((1000, 1000, 3), dtype=np.uint8))

    labels = [r.label for r in regions]
    assert "image" not in labels  # non-text dropped
    assert labels.count("text") == 1  # low-score text dropped
    assert "header" in labels and "paragraph_title" in labels
    # poly is preserved as int page-coord points and requests poly mode
    assert capture["predict_kwargs"].get("layout_shape_mode") == "poly"
    for region in regions:
        assert isinstance(region, Region)
        assert region.lines is None
        assert region.poly is not None and len(region.poly) == 4
        assert all(isinstance(v, int) for v in region.box)
        assert all(isinstance(c, int) for pt in region.poly for c in pt)


def test_detect_sorts_by_native_reading_order():
    # native order = the model's `order` field (0=header, 1=title, 2=text);
    # detect() sorts by it, so that becomes the `native` reading order.
    capture = {}
    detector = _detector(BOXES, capture)
    regions = detector.detect(np.zeros((1000, 1000, 3), dtype=np.uint8))
    assert [r.label for r in regions] == ["header", "paragraph_title", "text"]


def test_device_cuda_becomes_gpu_and_model_name_explicit():
    capture = {}
    _detector(BOXES, capture, device="cuda")
    assert capture.get("device") == "gpu"
    assert capture.get("model_name") == "PP-DocLayoutV3"


def test_empty_detection_falls_back_to_whole_page():
    capture = {}
    detector = _detector([], capture)
    regions = detector.detect(np.zeros((600, 800, 3), dtype=np.uint8))
    assert len(regions) == 1
    assert regions[0].label == "page"
    assert regions[0].box == (0, 0, 800, 600)


def test_engine_reading_order_and_validation():
    capture = {}
    module = _fake_paddlex(BOXES, capture)
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    with mock.patch.dict(sys.modules, {"paddlex": module}):
        from armenian_ocr.layout_paddle import PaddleDocLayoutEngine

        with pytest.raises(ValueError):
            PaddleDocLayoutEngine(reading_order="bogus")

        native = PaddleDocLayoutEngine(reading_order="native").analyze(image)
        xycut = PaddleDocLayoutEngine(reading_order="xycut").analyze(image)

    assert [r.label for r in native] == ["header", "paragraph_title", "text"]
    assert xycut[-1].label == "header"  # furniture sunk to the end by xycut


def test_missing_paddle_raises_runtimeerror():
    empty = types.ModuleType("paddlex")  # no create_model attribute
    with mock.patch.dict(sys.modules, {"paddlex": empty}):
        from armenian_ocr.layout_paddle import PaddleDocLayoutDetector

        with pytest.raises(RuntimeError):
            PaddleDocLayoutDetector()
