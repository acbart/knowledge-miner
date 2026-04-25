"""Duplicate KC detection using title similarity and optional embedding similarity."""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Tuple

from .models import KnowledgeComponent


def compute_title_similarity(t1: str, t2: str) -> float:
    return difflib.SequenceMatcher(None, t1.lower(), t2.lower()).ratio()


def find_duplicate_kcs(
    kcs: List[KnowledgeComponent], threshold: float = 0.85
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(kcs):
        for b in kcs[i + 1 :]:
            if compute_title_similarity(a.title, b.title) >= threshold:
                pairs.append((a.id, b.id))
    return pairs


class EmbeddingSimilarity:
    """Optional numpy-based bag-of-words cosine similarity."""

    def __init__(self) -> None:
        try:
            import numpy as np  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def compute(self, texts: List[str]) -> Any:
        if not self._available:
            raise RuntimeError("numpy is not available")
        import numpy as np

        vocab: Dict[str, int] = {}
        for text in texts:
            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        mat = np.zeros((len(texts), len(vocab)), dtype=float)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                mat[i, vocab[word]] += 1
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
