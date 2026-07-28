"""PaddleOCR recognizer using the open Armenian ``paddle-calfa-tiny`` model.

The model (a PP-OCRv6-tiny recognizer) is **downloaded automatically** from
``github.com/calfa-co/hye-paddle`` on first use and cached locally; point
``ARMENIAN_OCR_PADDLE_REC_DIR`` at a local inference model to override it. The
recognizer then behaves like the reference ``TesseractRecognizer``:

- for each layout region (already in reading order) the block is cropped
  with a little padding;
- PaddleOCR runs detection + recognition **on the crop**, so it segments
  its own lines inside the block (mirroring the Tesseract ``--psm 4`` flow);
- returned line polygons / word boxes are mapped back to page coordinates
  and assembled into one ``Paragraph`` per region, lines sorted top-to-bottom.

Setting ``ARMENIAN_OCR_PADDLE_ALLOW_STOCK=1`` (or ``allow_stock_fallback=True``)
falls back to a stock PP-OCR recognizer selected by name — its transcription of
Armenian glyphs is meaningless, but layout → det → rec → export runs end to end.

`paddleocr` is imported lazily so importing this module never requires Paddle to
be installed. If the Paddle backend (``paddleocr`` / ``paddlepaddle``, in the
``paddle`` extra) is missing, or the model download fails, construction raises a
clear ``RuntimeError`` and the pipeline can stay on Tesseract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from armenian_ocr._paddle_common import (
    iter_paddle_results,
    silence_paddle_logs,
)
from armenian_ocr.types import Box, Line, Paragraph, Region, Word

ENV_REC_DIR = "ARMENIAN_OCR_PADDLE_REC_DIR"
ENV_DET_DIR = "ARMENIAN_OCR_PADDLE_DET_DIR"
ENV_REC_NAME = "ARMENIAN_OCR_PADDLE_REC_NAME"
ENV_ALLOW_STOCK = "ARMENIAN_OCR_PADDLE_ALLOW_STOCK"
ENV_STOCK_REC_NAME = "ARMENIAN_OCR_PADDLE_STOCK_REC"

# Text-detection (`det`) tuning — controls how tight the per-line boxes are.
# The lines the `det` model finds inside each region crop were coarse/merged
# on degraded scans at PaddleOCR defaults (unclip_ratio=2.0 fattens boxes;
# limit_side_len=960 downscales wide region crops). Tighter defaults below,
# all overridable by env for per-corpus calibration.
ENV_DET_UNCLIP = "ARMENIAN_OCR_PADDLE_DET_UNCLIP_RATIO"
ENV_DET_LIMIT = "ARMENIAN_OCR_PADDLE_DET_LIMIT_SIDE_LEN"
ENV_DET_BOX_THRESH = "ARMENIAN_OCR_PADDLE_DET_BOX_THRESH"
DEFAULT_DET_UNCLIP = 1.5  # < 2.0 default: tighter lines, less vertical merge
DEFAULT_DET_LIMIT = 1600  # > 960 default: don't downscale wide region crops

# Stock PP-OCR recognizer used only to exercise the pipeline before the
# Armenian model exists (mobile = light on CPU). It does NOT know Armenian.
DEFAULT_STOCK_REC = "PP-OCRv5_mobile_rec"

# The Paddle backend (paddleocr + paddlepaddle) ships in the optional `paddle`
# extra; without it the recognizer cannot be built.
_BACKEND_MISSING = (
    "PaddleOCR backend not installed — install the paddle extra: "
    "pip install \"hye-open-ocr[paddle]\" (installs paddleocr + paddlepaddle; "
    "GPU users install paddlepaddle-gpu instead)."
)
# The paddle-calfa-tiny model auto-downloads from github.com/calfa-co/hye-paddle;
# this is raised only if that download fails (offline / GitHub unreachable).
_MODEL_DOWNLOAD_FAILED = (
    "could not download the paddle-calfa-tiny model "
    "(github.com/calfa-co/hye-paddle) — check your network, or set "
    "ARMENIAN_OCR_PADDLE_REC_DIR to a local inference directory"
)


def _union(boxes: Sequence[Box]) -> Box:
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return (min(xs1), min(ys1), max(xs2), max(ys2))


def _poly_to_box(poly, ox: int, oy: int) -> Box:
    """Axis-aligned page-coordinate box from a crop-local polygon/box."""
    points = np.asarray(poly, dtype=float).reshape(-1, 2)
    x1 = int(round(points[:, 0].min())) + ox
    y1 = int(round(points[:, 1].min())) + oy
    x2 = int(round(points[:, 0].max())) + ox
    y2 = int(round(points[:, 1].max())) + oy
    return (x1, y1, x2, y2)


def _poly_to_points(poly, ox: int, oy: int) -> Optional[List[List[int]]]:
    """Crop-local detection polygon -> tight page-coord quad ([[x,y], …]).

    Returns None for a plain 2-point box (``rec_boxes``), which carries no
    shape — only ``det`` polygons (``rec_polys``/``dt_polys``, 4+ points) do.
    """
    points = np.asarray(poly, dtype=float).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    return [[int(round(x)) + ox, int(round(y)) + oy] for x, y in points]


def _env_number(name: str, default, cast):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        return default


def _first_nonempty(*candidates):
    """First candidate that is present and non-empty, else ``[]``.

    PaddleOCR fields come back as numpy arrays, so `a or b` is unsafe (an empty
    or multi-element array raises "truth value … ambiguous"); use an explicit
    length check via ``len()`` instead.
    """
    for candidate in candidates:
        if candidate is not None and len(candidate) > 0:
            return candidate
    return []


def _rec_model_name_from_dir(rec_dir: str) -> Optional[str]:
    """Read ``Global.model_name`` from an exported ``inference.yml``.

    PaddleX asserts ``text_recognition_model_name`` matches this value when a
    custom model dir is passed, so we auto-detect it (overridable via the
    ``rec_model_name`` kwarg / ``ARMENIAN_OCR_PADDLE_REC_NAME``).
    """
    yml = Path(rec_dir) / "inference.yml"
    if not yml.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        name = (data.get("Global") or {}).get("model_name")
        if name:
            return str(name)
    except Exception:
        pass
    # minimal fallback: scan for a top-level "model_name:" line
    try:
        for line in yml.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("model_name:"):
                return stripped.split(":", 1)[1].strip().strip("\"'")
    except Exception:
        return None
    return None


class PaddleRecognizer:
    def __init__(
        self,
        rec_model_dir: Optional[Union[str, Path]] = None,
        det_model_dir: Optional[Union[str, Path]] = None,
        *,
        lang: str = "armenian",
        block_pad: int = 8,
        use_textline_orientation: bool = False,
        rec_score_thresh: float = 0.0,
        rec_model_name: Optional[str] = None,
        allow_stock_fallback: Optional[bool] = None,
        stock_rec_model_name: Optional[str] = None,
        det_unclip_ratio: Optional[float] = DEFAULT_DET_UNCLIP,
        det_limit_side_len: Optional[int] = DEFAULT_DET_LIMIT,
        det_box_thresh: Optional[float] = None,
    ):
        rec_model_dir = rec_model_dir or os.environ.get(ENV_REC_DIR)
        det_model_dir = det_model_dir or os.environ.get(ENV_DET_DIR)
        det_unclip_ratio = _env_number(ENV_DET_UNCLIP, det_unclip_ratio, float)
        det_limit_side_len = _env_number(ENV_DET_LIMIT, det_limit_side_len, int)
        det_box_thresh = _env_number(ENV_DET_BOX_THRESH, det_box_thresh, float)
        if allow_stock_fallback is None:
            allow_stock_fallback = os.environ.get(
                ENV_ALLOW_STOCK, ""
            ).strip().lower() in ("1", "true", "yes")
        stock_rec_model_name = (
            stock_rec_model_name
            or os.environ.get(ENV_STOCK_REC_NAME)
            or DEFAULT_STOCK_REC
        )

        # No explicit dir and no override env: fetch the open paddle-calfa-tiny
        # model (github.com/calfa-co/hye-paddle), cached locally — mirrors how
        # the Tesseract model is auto-downloaded.
        if not rec_model_dir and not allow_stock_fallback:
            from armenian_ocr import models

            try:
                rec_model_dir = str(models.get_paddle_rec_dir())
            except RuntimeError as error:
                raise RuntimeError(_MODEL_DOWNLOAD_FAILED) from error

        have_custom = bool(rec_model_dir) and Path(rec_model_dir).exists()
        if not have_custom and not allow_stock_fallback:
            raise RuntimeError(_MODEL_DOWNLOAD_FAILED)

        try:
            from paddleocr import PaddleOCR
        except Exception as error:  # paddleocr / paddlepaddle missing
            raise RuntimeError(_BACKEND_MISSING) from error
        silence_paddle_logs()

        self.det_model_dir = str(det_model_dir) if det_model_dir else None
        self.block_pad = block_pad
        self.rec_score_thresh = rec_score_thresh

        # PaddleOCR.predict() runs text DETECTION (finds the lines) then
        # RECOGNITION on each line crop, so the (line-level) rec model always
        # receives single-line images — this is where "lines are fed to paddle".
        kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
        )
        if have_custom:
            # the custom Armenian rec carries its own dictionary
            self.rec_model_dir = str(rec_model_dir)
            self.using_stock = False
            kwargs["text_recognition_model_dir"] = self.rec_model_dir
            # PaddleX asserts the model NAME matches inference.yml's
            # Global.model_name; auto-detect it (overridable via kwarg/env).
            name = (
                rec_model_name
                or os.environ.get(ENV_REC_NAME)
                or _rec_model_name_from_dir(self.rec_model_dir)
            )
            if name:
                kwargs["text_recognition_model_name"] = name
        else:
            # plumbing test before the Armenian model exists: a stock rec
            # selected by name (auto-downloaded). Its output on Armenian
            # glyphs is meaningless — this only exercises layout→det→rec.
            self.rec_model_dir = None
            self.using_stock = True
            kwargs["text_recognition_model_name"] = stock_rec_model_name
        if self.det_model_dir:
            kwargs["text_detection_model_dir"] = self.det_model_dir
        # Tighter `det` so per-line boxes hug the text instead of merging
        # (see DEFAULT_DET_* above). limit_type="max" keeps the wide region
        # crop un-downscaled up to limit_side_len.
        if det_unclip_ratio is not None:
            kwargs["text_det_unclip_ratio"] = det_unclip_ratio
        if det_limit_side_len is not None:
            kwargs["text_det_limit_side_len"] = det_limit_side_len
            kwargs["text_det_limit_type"] = "max"
        if det_box_thresh is not None:
            kwargs["text_det_box_thresh"] = det_box_thresh
        # NOTE: PaddleOCR ignores (and warns about) `lang`/`ocr_version` once
        # explicit model dirs/names are given, so we do not pass `lang` — the
        # rec model carries its own dictionary. `lang` is kept in the
        # signature only for API symmetry.

        try:
            self._ocr = PaddleOCR(**kwargs)
        except Exception as error:  # e.g. paddlepaddle backend not installed
            raise RuntimeError(_BACKEND_MISSING) from error

    # -- image helpers (mirrors TesseractRecognizer) ----------------------

    def _crop(self, image, box: Box, pad: int):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(width, x2 + pad), min(height, y2 + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, 0, 0
        return image[cy1:cy2, cx1:cx2], cx1, cy1

    # -- PaddleOCR result parsing (normaliser in _paddle_common) ----------

    def _lines_from_result(
        self, data: dict, ox: int, oy: int
    ) -> List[Line]:
        # PaddleOCR returns these as numpy arrays, so pick with an explicit
        # length check — `a or b` raises on a (possibly empty) ndarray.
        texts = _first_nonempty(data.get("rec_texts"))
        scores = _first_nonempty(data.get("rec_scores"))
        polys = _first_nonempty(
            data.get("rec_polys"),
            data.get("dt_polys"),
            data.get("rec_boxes"),
        )

        lines: List[Line] = []
        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                continue
            score = float(scores[i]) if i < len(scores) else None
            confidence = (
                round(score * 100, 2) if score is not None else None
            )
            if score is not None and score < self.rec_score_thresh:
                continue
            poly = None
            if i < len(polys) and len(np.asarray(polys[i]).reshape(-1)) >= 4:
                box = _poly_to_box(polys[i], ox, oy)
                # keep the tight `det` quad (rec_polys/dt_polys) so slanted
                # lines are drawn/exported as polygons, not fat rectangles.
                poly = _poly_to_points(polys[i], ox, oy)
            else:
                box = (ox, oy, ox, oy)
            # PaddleOCR gives one text box per detected line; expose it as a
            # single-word Line so downstream (ALTO/JSON) still has word boxes.
            word = Word(box=box, text=text, confidence=confidence, poly=poly)
            lines.append(
                Line(
                    box=box,
                    words=[word],
                    text=text,
                    confidence=confidence,
                    poly=poly,
                )
            )
        # top-to-bottom within the region
        lines.sort(key=lambda line: (line.box[1], line.box[0]))
        return lines

    # -- Recognizer protocol ----------------------------------------------

    def recognize(
        self, image: np.ndarray, regions: Sequence[Region]
    ) -> List[Paragraph]:
        paragraphs: List[Paragraph] = []
        for region in regions:
            crop, ox, oy = self._crop(image, region.box, self.block_pad)
            lines: List[Line] = []
            if crop is not None:
                try:
                    raw = self._ocr.predict(crop)
                except Exception:
                    raw = None
                for data in iter_paddle_results(raw):
                    lines.extend(self._lines_from_result(data, ox, oy))
                lines.sort(key=lambda line: (line.box[1], line.box[0]))

            box = (
                _union([line.box for line in lines]) if lines else region.box
            )
            paragraphs.append(
                Paragraph(
                    box=box,
                    lines=lines,
                    label=region.label,
                    poly=region.poly,
                )
            )
        return paragraphs
