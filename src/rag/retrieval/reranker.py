from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol
from rag.schemas import ScoredChunk

class _PredictsRelevance(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        pass


class CrossEncoderReranker(Reranker):
    def __init__(self, model: _PredictsRelevance) -> None:
        self._model = model

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, sc.chunk.content) for sc in candidates]
        scores = self._model.predict(pairs)
        rescored = [
            ScoredChunk(chunk=sc.chunk, score=float(score))
            for sc, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:top_n]

def load_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> _PredictsRelevance:

    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)
