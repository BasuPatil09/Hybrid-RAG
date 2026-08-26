from __future__ import annotations

from rag.indexing.sparse_index import SparseIndex, default_tokenizer
from rag.schemas import Chunk, ChunkMetadata


def _chunk(cid: str, content: str) -> Chunk:
    return Chunk(
        id=cid,
        content=content,
        metadata=ChunkMetadata(
            source="f.md", format="md", chunk_index=0, char_count=len(content)
        ),
    )


class TestDefaultTokenizer:
    def test_lowercases_and_splits_on_word_boundaries(self) -> None:
        assert default_tokenizer("Hello, World! BM25.") == ["hello", "world", "bm25"]


class TestSparseIndex:
    def test_query_before_rebuild_returns_empty(self) -> None:
        assert SparseIndex().query("anything", top_k=5) == []

    def test_rebuild_with_empty_chunks_leaves_index_empty(self) -> None:
        index = SparseIndex()
        index.rebuild([])
        assert index.query("anything", top_k=5) == []

    def test_exact_keyword_match_ranks_above_unrelated_chunk(self) -> None:

        index = SparseIndex()
        index.rebuild(
            [
                _chunk(
                    "a",
                    "PyTorch tensors support automatic differentiation via autograd",
                ),
                _chunk("b", "The weather today is sunny with a chance of rain"),
                _chunk("c", "Baking bread requires flour, water, yeast, and time"),
                _chunk("d", "The stock market closed higher today on strong earnings"),
                _chunk("e", "Migratory birds travel thousands of miles each winter"),
            ]
        )
        results = index.query("autograd differentiation", top_k=2)
        assert results[0].chunk.id == "a"

    def test_respects_top_k(self) -> None:
        index = SparseIndex()
        index.rebuild(
            [_chunk(str(i), f"machine learning topic number {i}") for i in range(10)]
        )
        results = index.query("machine learning", top_k=3)
        assert len(results) == 3

    def test_excludes_zero_score_results(self) -> None:
        index = SparseIndex()
        index.rebuild([_chunk("a", "completely unrelated content about gardening")])
        assert index.query("quantum physics equations", top_k=5) == []

    def test_rebuild_replaces_previous_index_entirely(self) -> None:
        index = SparseIndex()
        index.rebuild([_chunk("old", "old content about cooking")])
        index.rebuild([_chunk("new", "new content about coding")])
        results = index.query("cooking", top_k=5)
        assert results == []
