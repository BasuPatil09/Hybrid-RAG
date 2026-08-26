from __future__ import annotations

from rag.indexing.sparse_index import SparseIndex
from rag.schemas import ScoredChunk

def sparse_retrieve(
    query: str, sparse_index: SparseIndex, top_k: int
) -> list[ScoredChunk]:
    return sparse_index.query(query, top_k=top_k)
