from __future__ import annotations

from rag.retrieval.reranker import CrossEncoderReranker
from rag.schemas import Chunk, ChunkMetadata, ScoredChunk


def _scored(cid: str, content: str, score: float = 0.0) -> ScoredChunk:
    chunk = Chunk(
        id=cid,
        content=content,
        metadata=ChunkMetadata(
            source="f.md", format="md", chunk_index=0, char_count=len(content)
        ),
    )
    return ScoredChunk(chunk=chunk, score=score)


class _FakeCrossEncoderModel:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        scores = []
        for query, text in pairs:
            overlap = set(query.lower().split()) & set(text.lower().split())
            scores.append(float(len(overlap)))
        return scores


class TestCrossEncoderReranker:
    def test_empty_candidates_returns_empty_without_calling_model(self) -> None:
        model = _FakeCrossEncoderModel()
        reranker = CrossEncoderReranker(model)
        assert reranker.rerank("query", [], top_n=5) == []
        assert model.calls == []

    def test_reorders_by_model_score_not_input_order(self) -> None:
        model = _FakeCrossEncoderModel()
        reranker = CrossEncoderReranker(model)
        candidates = [
            _scored("weak", "completely unrelated content about gardening"),
            _scored("strong", "gradient descent optimizes model parameters"),
        ]
        results = reranker.rerank("gradient descent parameters", candidates, top_n=2)
        assert results[0].chunk.id == "strong"

    def test_truncates_to_top_n(self) -> None:
        model = _FakeCrossEncoderModel()
        reranker = CrossEncoderReranker(model)
        candidates = [_scored(str(i), f"content about topic {i}") for i in range(20)]
        results = reranker.rerank("topic", candidates, top_n=5)
        assert len(results) == 5

    def test_passes_query_paired_with_each_candidate_content_to_model(self) -> None:
        model = _FakeCrossEncoderModel()
        reranker = CrossEncoderReranker(model)
        candidates = [
            _scored("a", "first chunk text"),
            _scored("b", "second chunk text"),
        ]
        reranker.rerank("my query", candidates, top_n=2)
        assert model.calls == [
            [("my query", "first chunk text"), ("my query", "second chunk text")]
        ]

    def test_score_is_replaced_not_combined_with_fusion_score(self) -> None:

        model = _FakeCrossEncoderModel()
        reranker = CrossEncoderReranker(model)
        candidates = [_scored("a", "no overlap here", score=999.0)]
        results = reranker.rerank("zzz", candidates, top_n=1)
        assert results[0].score == 0.0

    def test_top_n_larger_than_candidates_returns_all(self) -> None:
        model = _FakeCrossEncoderModel()
        reranker = CrossEncoderReranker(model)
        candidates = [_scored("a", "solo candidate")]
        assert len(reranker.rerank("query", candidates, top_n=20)) == 1
