"""Searchable PDF export: page image + invisible text layer.

The text layer is written word by word with an embedded DejaVu Sans
font — the PDF base-14 fonts cannot encode Armenian, so an embedded
Unicode font is required even for invisible text, otherwise search and
copy/paste break. DejaVu Sans covers Armenian (with its U+FB13-FB17
ligatures), Cyrillic and Latin; the font is subset before saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import pymupdf

from armenian_ocr.types import Page

_FONT_FILE = str(Path(__file__).parent / "fonts" / "DejaVuSans.ttf")


def write_searchable_pdf(
    pages: List[Page],
    images: List[np.ndarray],
    output_path: Union[str, Path],
    jpeg_quality: int = 85,
) -> None:
    """Build a searchable PDF from OCR pages and their RGB page images."""
    if len(pages) != len(images):
        raise ValueError("pages and images must have the same length")

    font = pymupdf.Font(fontfile=_FONT_FILE)
    doc = pymupdf.open()

    for page, image in zip(pages, images):
        dpi = page.dpi or 300
        scale = 72 / dpi
        pdf_page = doc.new_page(
            width=page.width * scale, height=page.height * scale
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode page image")
        pdf_page.insert_image(pdf_page.rect, stream=encoded.tobytes())

        writer = pymupdf.TextWriter(pdf_page.rect)
        for line in page.lines:
            y1_line, y2_line = line.box[1], line.box[3]
            line_height = max(1, y2_line - y1_line)
            for word in line.words:
                if not word.text:
                    continue
                x1, _, x2, y2 = word.box
                width_1pt = font.text_length(word.text, fontsize=1)
                if width_1pt <= 0:
                    continue
                # fit the word into its box width, capped by line height
                fontsize = min(
                    (x2 - x1) * scale / width_1pt, line_height * scale
                )
                if fontsize <= 0:
                    continue
                baseline = y2 * scale + font.descender * fontsize
                writer.append(
                    (x1 * scale, baseline),
                    word.text,
                    font=font,
                    fontsize=fontsize,
                )
        writer.write_text(pdf_page, render_mode=3)  # invisible

        # Visible provenance line in the bottom margin, over the page image.
        # Size is proportional to the page height (~1%) so it stays small on
        # any document, and is shrunk further if it would overflow the width.
        page_h = pdf_page.rect.height
        page_w = pdf_page.rect.width
        footer_text = "OCRized by Calfa open OCR model"
        footer_size = max(2.0, min(4.5, page_h * 0.005))
        footer_width = font.text_length(footer_text, fontsize=footer_size)
        if footer_width > page_w * 0.9:  # keep within 90% of the width
            footer_size *= (page_w * 0.9) / footer_width
            footer_width = font.text_length(footer_text, fontsize=footer_size)
        footer = pymupdf.TextWriter(pdf_page.rect, color=(0.65, 0.65, 0.65))
        footer.append(
            (
                (page_w - footer_width) / 2,
                page_h - footer_size * 0.6,
            ),
            footer_text,
            font=font,
            fontsize=footer_size,
        )
        footer.write_text(pdf_page)  # visible (default render mode)

    try:
        doc.subset_fonts()  # needs fonttools; shrinks the embedded font
    except Exception:
        pass
    doc.save(str(output_path), garbage=3, deflate=True)
    doc.close()
