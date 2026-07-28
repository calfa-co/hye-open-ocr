"""Export of OCR results to txt / json / alto / pdf."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

from armenian_ocr.export.alto import pages_to_alto  # noqa: F401
from armenian_ocr.export.json_out import (  # noqa: F401
    pages_to_dict,
    pages_to_json,
)
from armenian_ocr.export.text import pages_to_text  # noqa: F401
from armenian_ocr.types import Page

FORMATS = ("txt", "json", "alto", "pdf")
EXTENSIONS = {"txt": ".txt", "json": ".json", "alto": ".xml", "pdf": ".pdf"}


def export_pages(
    pages: List[Page],
    formats: Iterable[str],
    output_dir: Union[str, Path],
    stem: str,
    *,
    images: Optional[List[np.ndarray]] = None,
    source_name: Optional[str] = None,
) -> List[Path]:
    """Write the requested formats to {output_dir}/{stem}{ext}.

    `images` (the RGB page images) is only required for `pdf`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for format_name in formats:
        if format_name not in FORMATS:
            raise ValueError(
                f"Unknown format '{format_name}'. Available: {FORMATS}"
            )
        path = output_dir / (stem + EXTENSIONS[format_name])

        if format_name == "txt":
            path.write_text(pages_to_text(pages), encoding="utf-8")
        elif format_name == "json":
            path.write_text(pages_to_json(pages), encoding="utf-8")
        elif format_name == "alto":
            path.write_text(
                pages_to_alto(pages, source_name=source_name),
                encoding="utf-8",
            )
        elif format_name == "pdf":
            if images is None:
                raise ValueError("pdf export requires the page images")
            from armenian_ocr.export.pdf import write_searchable_pdf

            write_searchable_pdf(pages, images, path)

        written.append(path)
    return written
