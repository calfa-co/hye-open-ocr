"""Document loading: images and PDFs to RGB numpy pages.

PDFs are rendered page by page with PyMuPDF, so only one page bitmap is
in memory at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image file as RGB uint8, blending any alpha over white."""
    with Image.open(path) as pil_image:
        return pil_to_rgb_array(pil_image)


def pil_to_rgb_array(pil_image: Image.Image) -> np.ndarray:
    if pil_image.mode in ("RGBA", "LA", "PA"):
        pil_image = pil_image.convert("RGBA")
        blended = Image.new("RGB", pil_image.size, (255, 255, 255))
        blended.paste(pil_image, mask=pil_image.split()[3])
        return np.array(blended)
    return np.array(pil_image.convert("RGB"))


def iter_pdf_pages(
    path: Union[str, Path], dpi: int = 300
) -> Iterator[np.ndarray]:
    """Render PDF pages as RGB uint8 arrays, one at a time."""
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        for page in doc:
            pixmap = page.get_pixmap(
                dpi=dpi, colorspace=pymupdf.csRGB, alpha=False
            )
            yield np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, 3
            ).copy()


def count_pdf_pages(path: Union[str, Path]) -> int:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return doc.page_count


def iter_pages(
    path: Union[str, Path], dpi: int = 300
) -> Iterator[Tuple[np.ndarray, Optional[int]]]:
    """Yield (rgb_image, dpi) for each page of an image or PDF file.

    For plain images the true DPI is unknown, so the given value is
    passed through as a hint.
    """
    path = Path(path)
    extension = path.suffix.lower()

    if extension in PDF_EXTENSIONS:
        for image in iter_pdf_pages(path, dpi=dpi):
            yield image, dpi
    elif extension in IMAGE_EXTENSIONS:
        yield load_image(path), dpi
    else:
        raise ValueError(
            f"Unsupported document type '{extension}'. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
