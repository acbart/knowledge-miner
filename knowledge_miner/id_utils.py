"""ID generation helpers for learning objectives and knowledge components."""

from __future__ import annotations

import re
from typing import List

from .models import LearningObjective


def make_lo_id(index: int) -> str:
    return f"lo-{index:03d}"


def make_kc_id(title: str, existing_ids: set) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    candidate = f"kc-{slug}"
    if candidate not in existing_ids:
        return candidate
    i = 2
    while f"{candidate}-{i}" in existing_ids:
        i += 1
    return f"{candidate}-{i}"


def normalize_lo_ids(objectives: List[LearningObjective]) -> List[LearningObjective]:
    return [
        LearningObjective(id=make_lo_id(i), title=lo.title, description=lo.description)
        for i, lo in enumerate(objectives, start=1)
    ]
