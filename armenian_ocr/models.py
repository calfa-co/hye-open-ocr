"""Model weight resolution: download with local caching.

Three model artifacts back the pipeline:

- the DocLayout-YOLO region detector, fetched from the authors' public
  ``juliozhao/DocLayout-YOLO-DocStructBench`` HuggingFace repo (override with
  ``ARMENIAN_OCR_YOLO_WEIGHTS``) — a third-party, generic layout model;
- the Tesseract ``hye-calfa-n.traineddata`` recognition model and the Paddle
  ``paddle-calfa-tiny`` inference model, both trained by Calfa and published as
  public GitHub repos (``calfa-co/hye-tesseract`` and ``calfa-co/hye-paddle``).
  They are fetched from ``raw.githubusercontent.com`` and cached locally.

Each model can be overridden with a local path (``ARMENIAN_OCR_TESSDATA_DIR`` /
``ARMENIAN_OCR_PADDLE_REC_DIR``), e.g. to test a freshly trained one without
touching the code.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

ENV_TESSDATA_DIR = "ARMENIAN_OCR_TESSDATA_DIR"
ENV_PADDLE_REC_DIR = "ARMENIAN_OCR_PADDLE_REC_DIR"
ENV_YOLO_WEIGHTS = "ARMENIAN_OCR_YOLO_WEIGHTS"
ENV_TESSERACT_VERSION = "ARMENIAN_OCR_TESSERACT_VERSION"
ENV_PADDLE_VERSION = "ARMENIAN_OCR_PADDLE_VERSION"

# Calfa-trained models, published as public GitHub repos under calfa-co. Files
# are pulled from raw.githubusercontent.com at a pinned version tag and cached
# locally (side by side per version). Bump the *_VERSION constant to roll the
# default to a new release; override at runtime with the matching env var.
GITHUB_ORG = "calfa-co"

TESSERACT_REPO = "hye-tesseract"
TESSERACT_VERSION = "v1.0.0"
TESSDATA_FILE = "hye-calfa-n.traineddata"

PADDLE_REPO = "hye-paddle"
PADDLE_VERSION = "v1.0.0"
PADDLE_INFERENCE_FILES = (
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)

# DocLayout-YOLO region detector (default layout engine). Hosted on the
# authors' public HuggingFace repo — a third-party generic model.
DEFAULT_YOLO_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
YOLO_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"


def _cache_root() -> Path:
    """Local cache directory for downloaded model files."""
    try:
        import platformdirs

        base = Path(platformdirs.user_cache_dir("armenian-ocr"))
    except Exception:
        base = Path.home() / ".cache" / "armenian-ocr"
    return base


def _github_download(repo: str, ref: str, path_in_repo: str) -> Path:
    """Download ``path_in_repo`` from ``calfa-co/{repo}@{ref}`` (cached).

    Returns the cached file path, fetching from raw.githubusercontent.com only
    when it is not already present. Raises ``RuntimeError`` on any network /
    HTTP error so callers can surface a normal engine error.
    """
    target = _cache_root() / repo / ref / path_in_repo
    if target.exists():
        return target

    url = (
        f"https://raw.githubusercontent.com/{GITHUB_ORG}/{repo}/{ref}/"
        f"{path_in_repo}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 (trusted host)
            if getattr(response, "status", 200) not in (200, None):
                raise RuntimeError(f"HTTP {response.status} for {url}")
            fd, tmp = tempfile.mkstemp(dir=str(target.parent))
            try:
                with os.fdopen(fd, "wb") as out:
                    shutil.copyfileobj(response, out)
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    except RuntimeError:
        raise
    except Exception as error:  # URLError, HTTPError, timeouts…
        raise RuntimeError(
            f"could not download {path_in_repo} from "
            f"{GITHUB_ORG}/{repo}@{ref}: {error}"
        ) from error
    return target


def tessdata_cached(revision: Optional[str] = None) -> bool:
    """True if the tesseract model is already available locally.

    Either an override directory is set (ARMENIAN_OCR_TESSDATA_DIR) or the
    pinned file is present in the cache — so no download will happen.
    """
    if os.environ.get(ENV_TESSDATA_DIR):
        return True
    ref = revision or os.environ.get(ENV_TESSERACT_VERSION) or TESSERACT_VERSION
    return (_cache_root() / TESSERACT_REPO / ref / TESSDATA_FILE).exists()


def paddle_rec_cached(revision: Optional[str] = None) -> bool:
    """True if the paddle recognizer model is already available locally."""
    if os.environ.get(ENV_PADDLE_REC_DIR):
        return True
    ref = revision or os.environ.get(ENV_PADDLE_VERSION) or PADDLE_VERSION
    root = _cache_root() / PADDLE_REPO / ref / "inference"
    return all((root / name).exists() for name in PADDLE_INFERENCE_FILES)


def yolo_weights_cached() -> bool:
    """True if the DocLayout-YOLO weights are already available locally.

    An override path counts; otherwise check the HuggingFace hub cache without
    triggering a download.
    """
    if os.environ.get(ENV_YOLO_WEIGHTS):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    hit = try_to_load_from_cache(DEFAULT_YOLO_REPO, YOLO_FILE)
    return isinstance(hit, str)


def get_yolo_weights(
    repo_id: Optional[str] = None, revision: Optional[str] = None
) -> Path:
    """Path to the DocLayout-YOLO weights (.pt).

    A local file set through ARMENIAN_OCR_YOLO_WEIGHTS takes precedence;
    otherwise the DocStructBench weights are downloaded from HuggingFace.
    """
    local = os.environ.get(ENV_YOLO_WEIGHTS)
    if local:
        return Path(local)

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id or DEFAULT_YOLO_REPO, YOLO_FILE, revision=revision
        )
    )


def get_tessdata_dir(revision: Optional[str] = None) -> Path:
    """Tessdata directory containing hye-calfa-n.traineddata.

    A local directory set through ARMENIAN_OCR_TESSDATA_DIR takes precedence;
    otherwise ``hye-calfa-n.traineddata`` is downloaded from the public
    ``calfa-co/hye-tesseract`` GitHub repo at the pinned ``TESSERACT_VERSION``
    (override with ``revision`` or ARMENIAN_OCR_TESSERACT_VERSION) and cached.
    Tesseract resolves languages as ``{tessdata_dir}/{lang}.traineddata``, so the
    returned path is the *directory* holding the file.
    """
    local = os.environ.get(ENV_TESSDATA_DIR)
    if local:
        return Path(local)

    ref = revision or os.environ.get(ENV_TESSERACT_VERSION) or TESSERACT_VERSION
    path = _github_download(TESSERACT_REPO, ref, TESSDATA_FILE)
    return path.parent


def get_paddle_rec_dir(revision: Optional[str] = None) -> Path:
    """Directory holding the paddle-calfa-tiny inference model.

    A local directory set through ARMENIAN_OCR_PADDLE_REC_DIR takes precedence;
    otherwise the ``inference/*`` files are downloaded from the public
    ``calfa-co/hye-paddle`` GitHub repo at the pinned ``PADDLE_VERSION``
    (override with ``revision`` or ARMENIAN_OCR_PADDLE_VERSION) and cached into
    one directory, which is what ``PaddleRecognizer`` expects (it reads
    ``inference.yml`` there).
    """
    local = os.environ.get(ENV_PADDLE_REC_DIR)
    if local:
        return Path(local)

    ref = revision or os.environ.get(ENV_PADDLE_VERSION) or PADDLE_VERSION
    directory = None
    for name in PADDLE_INFERENCE_FILES:
        path = _github_download(PADDLE_REPO, ref, f"inference/{name}")
        directory = path.parent
    assert directory is not None  # PADDLE_INFERENCE_FILES is non-empty
    return directory
