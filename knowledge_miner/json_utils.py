"""JSON parsing with repair and file I/O utilities."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Code-fence stripping and repair passes
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```$", "", text)
    return text.strip()


def _repair_pass1(text: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _repair_pass2(text: str) -> str:
    """Pass 1 + naive single-quote → double-quote replacement."""
    text = _repair_pass1(text)
    text = re.sub(r"'([^'\\]*)'", r'"\1"', text)
    return text


def _repair_pass3(text: str) -> str:
    """Pass 2 + truncate at last valid closing delimiter."""
    text = _repair_pass2(text)
    last = max(text.rfind("}"), text.rfind("]"))
    if last >= 0:
        text = text[: last + 1]
    return text


def parse_json_with_repair(text: str) -> Any:
    """Try to parse JSON, applying progressively more aggressive repairs on failure."""
    text = _strip_code_fences(text)
    for repair_fn in (None, _repair_pass1, _repair_pass2, _repair_pass3):
        candidate = repair_fn(text) if repair_fn else text
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise JSONParseError(
        f"Failed to parse JSON after all repair attempts. Preview: {text[:200]!r}"
    )


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.debug("Saved JSON → %s", path)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
