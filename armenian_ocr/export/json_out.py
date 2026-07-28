"""Structured JSON export: pages > paragraphs > lines > words."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from armenian_ocr.types import Page


def pages_to_dict(pages: List[Page]) -> Dict[str, Any]:
    return {
        "pages": [
            {"index": index + 1, **page.to_dict()}
            for index, page in enumerate(pages)
        ]
    }


def pages_to_json(pages: List[Page], indent: int = 2) -> str:
    return json.dumps(pages_to_dict(pages), ensure_ascii=False, indent=indent)
