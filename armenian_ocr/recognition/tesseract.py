"""Tesseract recognizer using the hye-calfa-n Armenian model.

Two granularities, both driven by the `Region` layout:

- Region without pre-segmented lines (DocLayout-YOLO): the whole block is
  OCR'd with `--psm 4` (or `--psm 3` for a full-page fallback) and
  Tesseract does its own line/word segmentation. This matches the way the
  line-level fine-tuned model was trained and avoids fragile external
  word cropping.
- Region with pre-segmented lines (word-box grouping): each line box is
  OCR'd with `--psm 13` (raw line).

Word boxes and confidences come from `image_to_data` and are mapped back
to page coordinates. Tesseract runs one subprocess per OCR call, so a
thread pool over all calls gives real parallelism.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pytesseract
from PIL import Image

from armenian_ocr.types import Box, Line, Paragraph, Region, Word

# page-segmentation mode for the full-page fallback region
_FULL_PAGE_PSM = 3


def _union(boxes: Sequence[Box]) -> Box:
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return (min(xs1), min(ys1), max(xs2), max(ys2))


class TesseractRecognizer:
    def __init__(
        self,
        lang: str = "hye-calfa-n",
        tessdata_dir: Optional[Union[str, Path]] = None,
        *,
        dpi: int = 300,
        region_psm: int = 4,
        line_psm: int = 13,
        pad_ratio: float = 0.15,
        block_pad: int = 8,
        max_workers: Optional[int] = None,
        tesseract_cmd: Optional[str] = None,
        timeout: int = 120,
    ):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        if tessdata_dir is None:
            from armenian_ocr import models

            tessdata_dir = models.get_tessdata_dir()
        self.tessdata_dir = Path(tessdata_dir)

        traineddata = self.tessdata_dir / f"{lang}.traineddata"
        if not traineddata.exists():
            raise FileNotFoundError(
                f"Tesseract model not found: {traineddata}. Pass tessdata_dir "
                f"pointing to a directory containing {lang}.traineddata, or "
                f"let armenian_ocr.models download it from the Hub."
            )

        self.lang = lang
        self.dpi = dpi
        self.region_psm = region_psm
        self.line_psm = line_psm
        self.pad_ratio = pad_ratio
        self.block_pad = block_pad
        self.timeout = timeout
        self.max_workers = max_workers or min(8, os.cpu_count() or 2)

        # keep each tesseract subprocess single-threaded; the pool already
        # saturates the cores
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")

    def _config(self, psm: int) -> str:
        return (
            f'--tessdata-dir "{self.tessdata_dir}" '
            f"--psm {psm} --dpi {self.dpi}"
        )

    def _data(self, crop: np.ndarray, psm: int):
        image = Image.fromarray(crop).convert("L")
        try:
            return pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self._config(psm),
                output_type=pytesseract.Output.DICT,
                timeout=self.timeout,
            )
        except RuntimeError:  # pytesseract raises RuntimeError on timeout
            return None

    def recognize(
        self, image: np.ndarray, regions: Sequence[Region]
    ) -> List[Paragraph]:
        # one flat task list over all OCR calls, so line-mode and
        # block-mode regions parallelize together
        tasks: List[Tuple[int, str, Box, int]] = []
        for index, region in enumerate(regions):
            if region.lines is None:
                psm = (
                    _FULL_PAGE_PSM
                    if region.label == "page"
                    else self.region_psm
                )
                tasks.append((index, "block", region.box, psm))
            else:
                for line_box in region.lines:
                    tasks.append((index, "line", line_box, self.line_psm))

        if tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(
                    executor.map(lambda t: self._run(image, t), tasks)
                )
        else:
            results = []

        # assemble one Paragraph per region, preserving task order
        per_region: List[List[Line]] = [[] for _ in regions]
        for (index, _kind, _box, _psm), lines in zip(tasks, results):
            per_region[index].extend(lines)

        paragraphs = []
        for region, lines in zip(regions, per_region):
            lines = [line for line in lines if line.text]
            box = _union([line.box for line in lines]) if lines else region.box
            paragraphs.append(
                Paragraph(
                    box=box,
                    lines=lines,
                    label=region.label,
                    poly=region.poly,
                )
            )
        return paragraphs

    def _run(self, image, task: Tuple[int, str, Box, int]) -> List[Line]:
        _index, kind, box, psm = task
        if kind == "line":
            line = self._ocr_line(image, box, psm)
            return [line] if line.text else []
        return self._ocr_block(image, box, psm)

    def _crop(self, image, box: Box, pad: int):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(width, x2 + pad), min(height, y2 + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, 0, 0
        return image[cy1:cy2, cx1:cx2], cx1, cy1

    def _word(self, data, i, ox, oy) -> Optional[Word]:
        text = data["text"][i].strip()
        if not text:
            return None
        confidence = float(data["conf"][i])
        return Word(
            box=(
                ox + int(data["left"][i]),
                oy + int(data["top"][i]),
                ox + int(data["left"][i]) + int(data["width"][i]),
                oy + int(data["top"][i]) + int(data["height"][i]),
            ),
            text=text,
            confidence=confidence if confidence >= 0 else None,
        )

    def _ocr_line(self, image, box: Box, psm: int) -> Line:
        pad = max(2, round(self.pad_ratio * (box[3] - box[1])))
        crop, ox, oy = self._crop(image, box, pad)
        if crop is None:
            return Line(box=box, words=[], text="", confidence=None)
        data = self._data(crop, psm)
        if data is None:
            return Line(box=box, words=[], text="", confidence=None)

        words = [
            word
            for i in range(len(data["level"]))
            if int(data["level"][i]) == 5
            and (word := self._word(data, i, ox, oy)) is not None
        ]
        return self._make_line(words, fallback_box=box)

    def _ocr_block(self, image, box: Box, psm: int) -> List[Line]:
        crop, ox, oy = self._crop(image, box, self.block_pad)
        if crop is None:
            return []
        data = self._data(crop, psm)
        if data is None:
            return []

        # group words by (block, paragraph, line); insertion order is
        # Tesseract's own top-to-bottom reading order
        grouped: "OrderedDict[Tuple[int, int, int], List[Word]]" = (
            OrderedDict()
        )
        for i in range(len(data["level"])):
            if int(data["level"][i]) != 5:
                continue
            word = self._word(data, i, ox, oy)
            if word is None:
                continue
            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            grouped.setdefault(key, []).append(word)

        return [self._make_line(words) for words in grouped.values()]

    @staticmethod
    def _make_line(
        words: List[Word], fallback_box: Optional[Box] = None
    ) -> Line:
        if not words:
            return Line(
                box=fallback_box or (0, 0, 0, 0),
                words=[],
                text="",
                confidence=None,
            )
        confidences = [w.confidence for w in words if w.confidence is not None]
        return Line(
            box=_union([w.box for w in words]),
            words=words,
            text=" ".join(w.text for w in words),
            confidence=(
                sum(confidences) / len(confidences) if confidences else None
            ),
        )
