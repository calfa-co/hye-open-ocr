# Armenian OCR

An **open** OCR pipeline for **printed Armenian documents** (Classical, Western
and Eastern), maintained by [Calfa](https://calfa.fr). This open release is
designed for a wide variety of documents. For handwritten and complex documents,
and structure extraction, see [Calfa](https://ocr.calfa.fr). It is built from two
complementary stages:

- **Layout detection & reading order** — a third-party, generic document-layout
  detector ([DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) by
  [opendatalab](https://github.com/opendatalab), or PP-DocLayoutV3) finds text
  regions, which are then ordered with a recursive X-Y-cut reading-order
  heuristic drawn from Portmind's armenian-ocr and updated here.
- **Recognition** — open models trained by [Calfa](https://calfa.fr):
  [`hye-calfa-n`](https://github.com/calfa-co/hye-tesseract) (Tesseract, the
  default) and [`paddle-calfa-tiny`](https://github.com/calfa-co/hye-paddle)
  (PaddleOCR), covering Classical, Western and Eastern Armenian.

It ships as the **`hyocr` command-line tool** (the main way to use it) plus an
optional web app to illustrate and test it. Outputs: **plain text** (reading
order), **searchable PDF**, **ALTO XML v4** and **structured JSON** (paragraphs →
lines → words, with boxes and confidences).

## Architecture

The pipeline is modular: both stages are protocols
(`armenian_ocr/types.py`), so either can be swapped without touching the
rest.

```
image ──► LayoutEngine.analyze(image)            ──► regions (reading order)
      ──► Recognizer.recognize(image, regions)   ──► paragraphs → lines → words
      ──► export: txt / json / alto / pdf
```

| Stage | Options |
|---|---|
| Layout | `yolo` (DocLayout-YOLO, fast) or `paddle` (PP-DocLayoutV3, tight polygons on skewed/degraded scans); reading order `xycut` (default) or `native` (paddle only) |
| Recognition | `tesseract` (`hye-calfa-n`, default — faster) or `paddle` (`paddle-calfa-tiny` — higher accuracy) |

The layout stage only finds regions; line and word segmentation is left to the
recognizer.

## Models

All models are downloaded on first use and cached locally:

| Model | Default source | Local override |
|---|---|---|
| DocLayout-YOLO weights (~40 MB) | public repo [`juliozhao/DocLayout-YOLO-DocStructBench`](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) | `ARMENIAN_OCR_YOLO_WEIGHTS` (path to a `.pt` file) |
| PP-DocLayoutV3 | public repo [`PaddlePaddle/PP-DocLayoutV3_safetensors`](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) | `ARMENIAN_OCR_PADDLE_LAYOUT_DIR` (a model directory) |
| `hye-calfa-n` (Tesseract) | public repo [`calfa-co/hye-tesseract`](https://github.com/calfa-co/hye-tesseract) | `ARMENIAN_OCR_TESSDATA_DIR` (a tessdata directory) |
| `paddle-calfa-tiny` (PaddleOCR) | public repo [`calfa-co/hye-paddle`](https://github.com/calfa-co/hye-paddle) | `ARMENIAN_OCR_PADDLE_REC_DIR` (an inference directory) |

DocLayout-YOLO and PP-DocLayoutV3 are third-party, generic layout models. The
Tesseract and Paddle recognition models are trained by Calfa; either can be
replaced by a fine-tuned one via its override path without any code change.

## Installation

Requires Python ≥ 3.10 and the [Tesseract](https://tesseract-ocr.github.io/)
binary (`brew install tesseract` / `apt-get install tesseract-ocr`).

```shell
python3 -m venv ocr
source ocr/bin/activate
git clone https://github.com/calfa-co/hye-open-ocr.git
cd hye-open-ocr

# full install (recommended): Tesseract + Paddle models + web app
pip install ".[paddle,app]"
```

The base install (`pip install .`) covers the default pipeline —
DocLayout-YOLO layout + the `hye-calfa-n` Tesseract recognizer. Add the extras
you need:

- **`paddle`** — required for `--recognizer paddle` (`paddle-calfa-tiny`) and
  `--layout paddle` (PP-DocLayoutV3). Installs `paddleocr` + `paddlepaddle`
  (CPU wheel; GPU users install `paddlepaddle-gpu` instead).
- **`app`** — the optional web app (FastAPI + uvicorn).

## Command-line interface

`hyocr` is the primary way to run the pipeline — ideal for batch processing and
scripting.

```shell
hyocr document.pdf --layout yolo --recognizer paddle -f pdf -o output/
```

Useful options:

- `--layout {yolo,paddle}` — layout detector (required for OCR).
- `--recognizer {tesseract,paddle}` — recognition model (default: `tesseract`).
- `--reading-order {xycut,native}` — reading-order strategy (`native` is
  PP-DocLayoutV3's own order and applies to `--layout paddle` only).
- `-f, --formats txt,json,alto,pdf` — output formats (default: `txt,json`).
- `-o, --output-dir` — output directory (default: current directory).
- `--yolo-conf 0.2` — DocLayout-YOLO confidence threshold. Lower it to recover
  text on hard pages where regions are missed.
- `--region-psm 4` — Tesseract page-segmentation mode for regions.
- `--dpi 300` — PDF rendering resolution and Tesseract DPI hint.
- `--tessdata-dir` / `--yolo-weights` / `--paddle-layout-dir` — use local model
  files (skip the download).

Run `hyocr --help` for the full list. (`armenian-ocr` also works as an alias.)

## Python library

The same pipeline is available as a library:

```python
import numpy as np
from PIL import Image
from armenian_ocr import OcrPipeline
from armenian_ocr.export import export_pages

pipeline = OcrPipeline()  # downloads models on first use
image = np.array(Image.open("page.png").convert("RGB"))
page = pipeline.process_image(image)

print(page.text)  # reading order
export_pages([page], ["txt", "alto", "pdf"], "output/", stem="page",
             images=[image], source_name="page.png")
```

## Web app (optional)

A small web app is included to illustrate and test the pipeline — a convenience,
not the primary interface. The `app/` directory is a FastAPI + vanilla JS
single-page app: drag & drop an image or PDF, choose the layout and recognition
models, per-page progress and text preview, a region/line overlay you can zoom, a
**confidence preview** to see which regions a lower threshold recovers, a
**compare** view for the layout detectors, and the four download formats.

Run it locally:

```shell
pip install ".[app]"
uvicorn app.main:app --port 7860   # then open http://localhost:7860
```

## Tests

```shell
pytest                 # unit tests (no models needed)
```

## Credits & links

- **Reading order** — recursive X-Y-cut heuristic drawn from Portmind's
  armenian-ocr and updated here.
- **Layout detection** — [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
  by [opendatalab](https://github.com/opendatalab); PP-DocLayoutV3 by PaddlePaddle.
- **Recognition** — [`hye-calfa-n`](https://github.com/calfa-co/hye-tesseract)
  and [`paddle-calfa-tiny`](https://github.com/calfa-co/hye-paddle), open models
  trained by [Calfa](https://calfa.fr).
- **This project** — <https://github.com/calfa-co/hye-open-ocr>.
- **Handwritten & complex documents, structure extraction** — Calfa:
  <https://ocr.calfa.fr>.

## License

CC BY-NC 4.0 — see [LICENSE](LICENSE-CC-BY-NC-4.0.md). The recognition models
`hye-calfa-n` and `paddle-calfa-tiny` are © [Calfa](https://calfa.fr).
Third-party components keep their own licenses: **PyMuPDF** and
**DocLayout-YOLO** are AGPL-3.0; **PaddleOCR** is Apache-2.0.
