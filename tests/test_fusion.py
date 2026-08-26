from __future__ import annotations

import pytest

from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.schemas import Chunk, ChunkMetadata, ScoredChunk


def _scored(cid: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(
        id=cid,
        content=cid,
        metadata=ChunkMetadata(source="f.md", format="md", chunk_index=0, char_count=1),
    )
    return ScoredChunk(chunk=chunk, score=score)


class TestReciprocalRankFusion:
    def test_matches_known_rrf_formula(self) -> None:
        dense = [_scored("a"), _scored("b")]
        sparse = [_scored("b"), _scored("a")]

        fused = reciprocal_rank_fusion(
            dense, sparse, k=60, dense_weight=0.7, sparse_weight=0.3
        )

        scores = {sc.chunk.id: sc.score for sc in fused}

        assert scores["a"] == pytest.approx(0.7 / 61 + 0.3 / 62)

        assert scores["b"] == pytest.approx(0.7 / 62 + 0.3 / 61)

    def test_chunk_in_both_lists_outranks_chunk_in_one(self) -> None:
        dense = [_scored("both"), _scored("dense_only")]
        sparse = [_scored("both")]

        fused = reciprocal_rank_fusion(
            dense, sparse, k=60, dense_weight=0.7, sparse_weight=0.3
        )

        assert fused[0].chunk.id == "both"

    def test_sorted_descending_by_fused_score(self) -> None:
        dense = [_scored("first"), _scored("second"), _scored("third")]
        fused = reciprocal_rank_fusion(
            dense, [], k=60, dense_weight=0.7, sparse_weight=0.3
        )
        assert [sc.chunk.id for sc in fused] == ["first", "second", "third"]

    def test_weights_shift_relative_ranking(self) -> None:
        dense = [_scored("dense_top")]
        sparse = [_scored("sparse_top")]

        dense_favored = reciprocal_rank_fusion(
            dense, sparse, k=60, dense_weight=0.99, sparse_weight=0.01
        )
        assert dense_favored[0].chunk.id == "dense_top"

        sparse_favored = reciprocal_rank_fusion(
            dense, sparse, k=60, dense_weight=0.01, sparse_weight=0.99
        )
        assert sparse_favored[0].chunk.id == "sparse_top"

    def test_both_empty_returns_empty(self) -> None:
        assert (
            reciprocal_rank_fusion([], [], k=60, dense_weight=0.7, sparse_weight=0.3)
            == []
        )

    def test_one_empty_returns_only_the_other(self) -> None:
        dense = [_scored("a"), _scored("b")]
        fused = reciprocal_rank_fusion(
            dense, [], k=60, dense_weight=0.7, sparse_weight=0.3
        )
        assert {sc.chunk.id for sc in fused} == {"a", "b"}

    def test_does_not_truncate_results(self) -> None:

        dense = [_scored(str(i)) for i in range(30)]
        fused = reciprocal_rank_fusion(
            dense, [], k=60, dense_weight=0.7, sparse_weight=0.3
        )
        assert len(fused) == 30
