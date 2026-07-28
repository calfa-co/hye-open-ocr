"""PaddleRecognizer with a mocked paddleocr — no model, no download.

Covers page-coordinate box mapping and the 0-1 -> 0-100 confidence rescale,
the stock-rec fallback (model *name* instead of dir), and the hard error when
neither a custom model nor the stock fallback is available.
"""

import sys
import types
from unittest import mock

import numpy as np
import pytest

from armenian_ocr.types import Region


class _FakeOCRResult:
    def __init__(self, data):
        self._data = data

    @property
    def json(self):
        return {"res": self._data}


def _fake_paddleocr(predict_data, capture):
    module = types.ModuleType("paddleocr")

    class PaddleOCR:
        def __init__(self, **kwargs):
            capture.update(kwargs)

        def predict(self, crop):
            return [_FakeOCRResult(predict_data)]

    module.PaddleOCR = PaddleOCR
    return module


def _recognizer(predict_data, capture, **kwargs):
    module = _fake_paddleocr(predict_data, capture)
    with mock.patch.dict(sys.modules, {"paddleocr": module}):
        from armenian_ocr.recognition.paddle import PaddleRecognizer

        return PaddleRecognizer(**kwargs)


def test_recognize_maps_box_and_rescales_confidence(tmp_path):
    data = {
        "rec_texts": ["Բարև"],
        "rec_scores": [0.87],
        "rec_polys": [[[10, 20], [110, 20], [110, 50], [10, 50]]],
    }
    capture = {}
    recognizer = _recognizer(data, capture, rec_model_dir=str(tmp_path))

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    paragraphs = recognizer.recognize(image, [Region(box=(0, 0, 150, 100))])

    assert len(paragraphs) == 1
    lines = paragraphs[0].lines
    assert len(lines) == 1
    assert lines[0].text == "Բարև"
    assert lines[0].confidence == 87.0  # 0.87 -> 87.0
    # the det quad is preserved on the line/word (page coords), not flattened
    assert lines[0].poly == [[10, 20], [110, 20], [110, 50], [10, 50]]
    assert lines[0].words[0].poly == lines[0].poly
    # custom model dir was passed (not a stock model name)
    assert capture.get("text_recognition_model_dir") == str(tmp_path)
    assert "text_recognition_model_name" not in capture
    # tighter det tuning is passed through to PaddleOCR
    assert capture.get("text_det_unclip_ratio") == 1.5
    assert capture.get("text_det_limit_side_len") == 1600
    assert capture.get("text_det_limit_type") == "max"


def test_empty_detection_result_does_not_crash(tmp_path):
    # a region with no detected lines comes back as empty numpy arrays;
    # `arr or []` used to raise "truth value of an empty array is ambiguous".
    data = {
        "rec_texts": np.array([]),
        "rec_scores": np.array([]),
        "rec_polys": np.array([]),
    }
    capture = {}
    recognizer = _recognizer(data, capture, rec_model_dir=str(tmp_path))

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    paragraphs = recognizer.recognize(image, [Region(box=(0, 0, 50, 50))])

    assert len(paragraphs) == 1
    assert paragraphs[0].lines == []  # no lines, no crash


def test_stock_fallback_uses_model_name(monkeypatch):
    monkeypatch.delenv("ARMENIAN_OCR_PADDLE_REC_DIR", raising=False)
    capture = {}
    _recognizer({}, capture, allow_stock_fallback=True)
    assert "text_recognition_model_name" in capture
    assert "text_recognition_model_dir" not in capture


def test_auto_downloads_model_when_no_dir(tmp_path, monkeypatch):
    # no explicit dir / env / stock: the recognizer auto-downloads
    # paddle-calfa-tiny (mocked here so no network) and passes it as a dir.
    monkeypatch.delenv("ARMENIAN_OCR_PADDLE_REC_DIR", raising=False)
    monkeypatch.delenv("ARMENIAN_OCR_PADDLE_ALLOW_STOCK", raising=False)
    capture = {}
    with mock.patch(
        "armenian_ocr.models.get_paddle_rec_dir", return_value=tmp_path
    ) as fetch:
        _recognizer({}, capture, allow_stock_fallback=False)
    fetch.assert_called_once()
    assert capture.get("text_recognition_model_dir") == str(tmp_path)


def test_download_failure_raises(monkeypatch):
    # if the auto-download itself fails and there is no fallback, construction
    # raises a clear RuntimeError (before paddleocr is imported).
    monkeypatch.delenv("ARMENIAN_OCR_PADDLE_REC_DIR", raising=False)
    monkeypatch.delenv("ARMENIAN_OCR_PADDLE_ALLOW_STOCK", raising=False)
    from armenian_ocr.recognition.paddle import PaddleRecognizer

    with mock.patch(
        "armenian_ocr.models.get_paddle_rec_dir",
        side_effect=RuntimeError("offline"),
    ):
        with pytest.raises(RuntimeError):
            PaddleRecognizer(allow_stock_fallback=False)
