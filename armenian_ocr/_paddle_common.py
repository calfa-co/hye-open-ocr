"""Small helpers shared by the PaddleOCR-backed engines.

Both the layout detector (`layout_paddle`) and the recognizer
(`recognition.paddle`) talk to PaddleOCR / PaddleX, so the three things they
have in common live here: normalising the various result shapes `predict`
returns, translating our device string to Paddle's, and muting PaddleOCR's
very chatty loggers. Nothing here imports paddle at module load — the engines
import paddle lazily so importing this module never requires it.
"""

from __future__ import annotations

import os
from typing import List

# Quiet PaddleX's startup before paddleocr is ever imported: skip the
# model-source connectivity probe and the dev-mode banner. setdefault keeps
# both overridable from the environment.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_DISABLE_DEV_MODE", "True")


def to_paddle_device(device: str) -> str:
    """Map our device string to PaddleOCR's.

    We use torch's ``cuda``; PaddleOCR expects ``gpu``. Anything else (incl.
    ``None``) falls back to ``cpu`` — the default, since running on CPU is the
    priority here.
    """
    return "gpu" if device in ("cuda", "gpu") else "cpu"


def silence_paddle_logs() -> None:
    """Quiet PaddleOCR/PaddleX INFO spam (one call per engine construction).

    Only touches the paddle loggers — never the root logger — so the app/CLI
    logging config is left alone. Safe to call before or after the paddle
    import; it just sets levels on named loggers.
    """
    import logging

    for name in ("paddleocr", "ppocr", "paddlex", "paddle"):
        logging.getLogger(name).setLevel(logging.ERROR)


def iter_paddle_results(raw) -> List[dict]:
    """Normalise the various shapes ``predict`` / ``ocr`` return to dicts.

    PaddleOCR 3.x returns a list of result objects that behave like dicts
    (recognition: keys ``rec_texts`` / ``rec_scores`` / polygon boxes; layout:
    a ``boxes`` list). Depending on version the payload is the item itself, is
    reachable via ``item.json`` (a PaddleX result object, possibly nested under
    ``res``), or via ``item["res"]``. Yields one plain dict per result.
    """
    results: List[dict] = []
    if raw is None:
        return results
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    for item in items:
        data = None
        if isinstance(item, dict):
            data = item
        elif hasattr(item, "json"):  # PaddleX result object
            blob = item.json
            data = blob.get("res", blob) if isinstance(blob, dict) else None
        if data is None and hasattr(item, "__getitem__"):
            try:
                data = item["res"]  # some versions nest under "res"
            except Exception:
                data = None
        if isinstance(data, dict):
            results.append(data)
    return results
