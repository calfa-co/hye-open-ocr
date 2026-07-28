"""Command-line interface: `armenian-ocr INPUT [options]`."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from time import time


def _progress(iterable, *, total=None, desc="pages"):
    """Wrap an iterable in a tqdm progress bar when tqdm is available.

    tqdm ships with the doclayout-yolo dependency; if it is somehow missing
    we degrade gracefully to the bare iterable.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="page")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyocr",
        description=(
            "Open OCR for Armenian documents (Classical / Western / Eastern). "
            "Pick a generic layout detector (--layout yolo|paddle) and an open "
            "recognizer (--recognizer tesseract|paddle); or use --compare to "
            "render layout + reading-order overlays without running OCR. "
            "This is a free, open release — for handwritten and complex "
            "documents, and structure extraction, see Calfa "
            "(https://ocr.calfa.fr)."
        ),
    )
    parser.add_argument("input", help="Input image (png/jpg/tiff) or PDF.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Output directory (default: current directory).",
    )
    parser.add_argument(
        "-f",
        "--formats",
        default="txt,json",
        help="Comma-separated output formats: txt,json,alto,pdf "
        "(default: txt,json).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PDF rendering resolution and Tesseract DPI hint (default: 300).",
    )
    parser.add_argument(
        "--lang",
        default="hye-calfa-n",
        help="Tesseract language/model name (default: hye-calfa-n).",
    )
    parser.add_argument(
        "--layout",
        choices=("yolo", "paddle"),
        default=None,
        help="Layout (region) detector — REQUIRED for OCR, no default. "
        "Third-party generic models: 'yolo' (DocLayout-YOLO, fast) or 'paddle' "
        "(PP-DocLayoutV3, tight polygons on degraded scans, needs the paddle "
        "extra). Not needed with --compare.",
    )
    parser.add_argument(
        "--reading-order",
        choices=("xycut", "native"),
        default="xycut",
        help="Reading-order strategy: 'xycut' (recursive X-Y cut, default) "
        "or 'native' (PP-DocLayoutV3's own order; --layout paddle only).",
    )
    parser.add_argument(
        "--recognizer",
        choices=("tesseract", "paddle"),
        default="tesseract",
        help="Open recognition model trained by Calfa: 'tesseract' "
        "(hye-calfa-n, default) or 'paddle' (paddle-calfa-tiny, PP-OCRv6-tiny). "
        "Both auto-download from github.com/calfa-co (cached); override paddle "
        "with ARMENIAN_OCR_PADDLE_REC_DIR, or set ARMENIAN_OCR_PADDLE_ALLOW_STOCK"
        "=1 to test with a stock rec.",
    )
    parser.add_argument(
        "--region-psm",
        type=int,
        default=4,
        help="Tesseract page-segmentation mode for regions (default: 4).",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.2,
        help="DocLayout-YOLO confidence threshold (default: 0.2).",
    )
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Local DocLayout-YOLO weights .pt (skips Hub download).",
    )
    parser.add_argument(
        "--paddle-layout-dir",
        default=None,
        help="Local PP-DocLayoutV3 model directory (skips auto-download; "
        "or set ARMENIAN_OCR_PADDLE_LAYOUT_DIR).",
    )
    parser.add_argument(
        "--paddle-rec-dir",
        default=None,
        help="Local paddle-calfa-tiny inference directory (skips "
        "auto-download; or set ARMENIAN_OCR_PADDLE_REC_DIR).",
    )
    parser.add_argument(
        "--tessdata-dir",
        default=None,
        help="Local tessdata directory containing {lang}.traineddata "
        "(skips Hub download).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for the layout model (default: cpu).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Tesseract worker threads (default: min(8, cpu count)).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-stage timings.",
    )

    compare = parser.add_argument_group(
        "layout comparison (no OCR)",
        "render region + reading-order overlays to compare detectors/orders",
    )
    compare.add_argument(
        "--compare",
        action="store_true",
        help="Compare layout engines / reading orders: write overlay PNGs "
        "and run NO recognition.",
    )
    compare.add_argument(
        "--detectors",
        default="yolo,paddle",
        help="Comma-separated detectors to compare (default: yolo,paddle).",
    )
    compare.add_argument(
        "--orders",
        default="xycut,native",
        help="Comma-separated reading orders to compare "
        "(default: xycut,native; native applies to paddle only).",
    )
    return parser


def _run_compare(args, input_path: Path) -> int:
    """OCR-free layout comparison: write one overlay PNG per candidate."""
    import cv2

    from armenian_ocr.compare import compare_layouts
    from armenian_ocr.documents import count_pdf_pages, iter_pages

    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multipage = input_path.suffix.lower() == ".pdf"
    total_pages = count_pdf_pages(input_path) if multipage else 1

    start = time()
    written = []
    page_iter = iter_pages(input_path, dpi=args.dpi)
    if not args.verbose:
        page_iter = _progress(page_iter, total=total_pages)
    for index, (image, _dpi) in enumerate(page_iter):
        page_tag = f".p{index}" if multipage else ""
        candidates = compare_layouts(
            image,
            detectors=detectors,
            orders=orders,
            device=args.device,
            overlay=True,
        )
        for candidate in candidates:
            if candidate.error:
                print(
                    f"  [{candidate.detector}] unavailable: "
                    f"{candidate.error}",
                    file=sys.stderr,
                )
                continue
            path = (
                output_dir
                / f"{input_path.stem}{page_tag}.{candidate.name}.png"
            )
            cv2.imwrite(
                str(path), cv2.cvtColor(candidate.overlay, cv2.COLOR_RGB2BGR)
            )
            written.append(path)
            if args.verbose:
                print(f"  {candidate.name}: {len(candidate.regions)} regions")

    print(f"layout comparison done in {time() - start:.1f}s")
    for path in written:
        print(f"  wrote {path}")
    if not written:
        print(
            "error: no overlays written (no detector available?)",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # PaddlePaddle prints a UserWarning about ccache being absent the first
    # time it is imported; it is harmless here (we ship prebuilt wheels, no
    # source recompilation), so silence just that one message.
    warnings.filterwarnings(
        "ignore", message="No ccache found.*", category=UserWarning
    )

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("error: --device cuda but CUDA is not available",
                  file=sys.stderr)
            return 1

    # Layout comparison is a separate, OCR-free flow.
    if args.compare:
        return _run_compare(args, input_path)

    # --- OCR path ---
    if args.layout is None:
        print(
            "error: --layout {yolo,paddle} is required for OCR "
            "(or use --compare)",
            file=sys.stderr,
        )
        return 1
    if args.reading_order == "native" and args.layout != "paddle":
        print(
            "error: --reading-order native is only valid with "
            "--layout paddle",
            file=sys.stderr,
        )
        return 1

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    from armenian_ocr import models
    from armenian_ocr.export import FORMATS, export_pages
    from armenian_ocr.pipeline import OcrPipeline

    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        print(
            f"error: unknown format(s) {unknown}, available: {FORMATS}",
            file=sys.stderr,
        )
        return 1

    # Build the recognizer first so a misconfigured paddle engine fails fast
    # (before downloading the layout weights).
    if args.recognizer == "paddle":
        from armenian_ocr.recognition.paddle import PaddleRecognizer

        if args.paddle_rec_dir is None and not models.paddle_rec_cached():
            print(
                "downloading paddle model paddle-calfa-tiny "
                "(github.com/calfa-co/hye-paddle)… (one-time, cached)"
            )
        try:
            recognizer = PaddleRecognizer(rec_model_dir=args.paddle_rec_dir)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    else:
        from armenian_ocr.recognition.tesseract import TesseractRecognizer

        tessdata_dir = args.tessdata_dir
        if tessdata_dir is None:
            if not models.tessdata_cached():
                print(
                    "downloading tesseract model hye-calfa-n "
                    "(github.com/calfa-co/hye-tesseract)… (one-time, cached)"
                )
            tessdata_dir = models.get_tessdata_dir()
        recognizer = TesseractRecognizer(
            lang=args.lang,
            tessdata_dir=tessdata_dir,
            dpi=args.dpi,
            region_psm=args.region_psm,
            max_workers=args.workers,
        )

    # Build the selected layout engine (also fail fast on a missing backend).
    if args.layout == "paddle":
        from armenian_ocr.layout_paddle import PaddleDocLayoutEngine

        try:
            engine = PaddleDocLayoutEngine(
                model_dir=args.paddle_layout_dir,
                device=args.device,
                reading_order=args.reading_order,
            )
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    else:
        from armenian_ocr.layout_yolo import YoloLayoutEngine

        if args.yolo_weights is None and not models.yolo_weights_cached():
            print("downloading DocLayout-YOLO weights… (one-time, cached)")
        engine = YoloLayoutEngine(
            weights=args.yolo_weights,
            device=args.device,
            conf=args.yolo_conf,
            reading_order=args.reading_order,
        )

    print("models ready.")
    pipeline = OcrPipeline(layout_engine=engine, recognizer=recognizer)

    from armenian_ocr.documents import count_pdf_pages, iter_pages

    total_pages = (
        count_pdf_pages(input_path)
        if input_path.suffix.lower() == ".pdf"
        else 1
    )

    start = time()
    keep_images = "pdf" in formats
    pages, images = [], ([] if keep_images else None)
    page_iter = iter_pages(input_path, dpi=args.dpi)
    # A bar only makes sense when we are not already printing per-page lines.
    if not args.verbose:
        page_iter = _progress(page_iter, total=total_pages)
    for index, (image, page_dpi) in enumerate(page_iter):
        if args.verbose:
            print(f"page {index + 1}…")
        pages.append(pipeline.process_image(image, dpi=page_dpi))
        if keep_images:
            images.append(image)

    written = export_pages(
        pages,
        formats,
        args.output_dir,
        stem=input_path.stem,
        images=images,
        source_name=input_path.name,
    )

    print(f"OCR done in {time() - start:.1f}s")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
