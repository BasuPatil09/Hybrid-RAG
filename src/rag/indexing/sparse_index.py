from __future__ import annotations
import re
from typing import Callable
from rank_bm25 import BM25Okapi
from rag.schemas import Chunk, ScoredChunk

_TOKEN_PATTERN = re.compile(r"\w+")

def default_tokenizer(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())

class SparseIndex:
    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None) -> None:
        self._tokenizer = tokenizer or default_tokenizer
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def rebuild(self, chunks: list[Chunk]) -> None:

        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        tokenized = [self._tokenizer(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def query(self, query_text: str, top_k: int) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenizer(query_text))
        ranked = sorted(
            zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            ScoredChunk(chunk=c, score=float(s)) for c, s in ranked[:top_k] if s > 0
        ]
