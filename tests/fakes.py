from __future__ import annotations

import hashlib

import numpy as np

from rag.indexing.embeddings import Embedder


class FakeEmbedder(Embedder):
    def __init__(self, dimensions: int = 32) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self._dimensions)
        return (vec / np.linalg.norm(vec)).tolist()


class FailAfterNCallsEmbedder(Embedder):
    def __init__(self, fail_after: int, dimensions: int = 32) -> None:
        self._fail_after = fail_after
        self._calls = 0
        self._inner = FakeEmbedder(dimensions=dimensions)

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._calls += 1
        if self._calls > self._fail_after:
            raise RuntimeError("simulated quota exhaustion")
        return self._inner.embed(texts)
