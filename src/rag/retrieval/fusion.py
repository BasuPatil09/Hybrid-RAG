from __future__ import annotations

from rag.schemas import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
    *,
    k: int,
    dense_weight: float,
    sparse_weight: float,
) -> list[ScoredChunk]:

    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for rank, scored in enumerate(dense_results, start=1):
        chunk_by_id[scored.chunk.id] = scored.chunk
        rrf_scores[scored.chunk.id] = rrf_scores.get(
            scored.chunk.id, 0.0
        ) + dense_weight / (k + rank)

    for rank, scored in enumerate(sparse_results, start=1):
        chunk_by_id[scored.chunk.id] = scored.chunk
        rrf_scores[scored.chunk.id] = rrf_scores.get(
            scored.chunk.id, 0.0
        ) + sparse_weight / (k + rank)

    fused = [
        ScoredChunk(chunk=chunk_by_id[cid], score=score)
        for cid, score in rrf_scores.items()
    ]
    fused.sort(key=lambda sc: sc.score, reverse=True)
    return fused
