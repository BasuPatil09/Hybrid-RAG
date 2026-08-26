from __future__ import annotations

import uuid

import chromadb
import pytest

from rag.indexing.vector_store import ChromaVectorStore
from rag.schemas import Chunk, ChunkMetadata


def _store() -> ChromaVectorStore:
    return ChromaVectorStore(
        chromadb.EphemeralClient(), collection_name=f"test_{uuid.uuid4().hex[:12]}"
    )


def _chunk(cid: str, content: str, **meta_overrides) -> Chunk:
    meta = {
        "source": "f.md",
        "format": "md",
        "chunk_index": 0,
        "char_count": len(content),
        **meta_overrides,
    }
    return Chunk(id=cid, content=content, metadata=ChunkMetadata(**meta))


class TestUpsertAndQuery:
    def test_empty_store_query_returns_empty(self) -> None:
        assert _store().query([1.0, 0.0, 0.0], top_k=5) == []

    def test_upsert_then_query_round_trips_content_and_metadata(self) -> None:
        store = _store()
        chunk = _chunk("a", "hello world", section="Intro", page=3)
        store.upsert([chunk], [[1.0, 0.0, 0.0]])

        results = store.query([1.0, 0.0, 0.0], top_k=1)

        assert results[0].chunk.id == "a"
        assert results[0].chunk.content == "hello world"
        assert results[0].chunk.metadata.section == "Intro"
        assert results[0].chunk.metadata.page == 3

    def test_query_orders_by_similarity(self) -> None:
        store = _store()
        store.upsert(
            [_chunk("close", "x"), _chunk("far", "y"), _chunk("exact", "z")],
            [[0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        )
        results = store.query([1.0, 0.0, 0.0], top_k=3)
        assert [r.chunk.id for r in results] == ["exact", "close", "far"]

    def test_score_is_cosine_similarity_not_raw_distance(self) -> None:
        store = _store()
        store.upsert([_chunk("a", "x")], [[1.0, 0.0]])
        results = store.query([1.0, 0.0], top_k=1)
        assert results[0].score == pytest.approx(1.0, abs=1e-6)

    def test_top_k_caps_at_collection_size(self) -> None:
        store = _store()
        store.upsert([_chunk("a", "x")], [[1.0, 0.0]])
        assert len(store.query([1.0, 0.0], top_k=50)) == 1

    def test_upsert_is_idempotent_on_same_id(self) -> None:
        store = _store()
        store.upsert([_chunk("a", "version one")], [[1.0, 0.0]])
        store.upsert([_chunk("a", "version two")], [[1.0, 0.0]])
        assert store.count() == 1
        assert store.query([1.0, 0.0], top_k=1)[0].chunk.content == "version two"

    def test_mismatched_lengths_raises(self) -> None:
        store = _store()
        with pytest.raises(ValueError):
            store.upsert([_chunk("a", "x"), _chunk("b", "y")], [[1.0, 0.0]])

    def test_upsert_batches_when_exceeding_max_batch_size(self) -> None:

        client = chromadb.EphemeralClient()
        client.get_max_batch_size = lambda: 2
        store = ChromaVectorStore(client, f"batch_test_{uuid.uuid4().hex[:12]}")

        chunks = [_chunk(str(i), f"content {i}") for i in range(5)]
        store.upsert(chunks, [[float(i), 0.0] for i in range(5)])

        assert store.count() == 5

    def test_none_metadata_fields_reconstruct_as_none(self) -> None:

        store = _store()
        store.upsert([_chunk("a", "x")], [[1.0, 0.0]])
        result = store.query([1.0, 0.0], top_k=1)[0]
        assert result.chunk.metadata.page is None
        assert result.chunk.metadata.section is None


class TestGetAll:
    def test_get_all_ids_empty_store(self) -> None:
        assert _store().get_all_ids() == set()

    def test_get_all_ids_returns_everything_indexed(self) -> None:
        store = _store()
        store.upsert([_chunk("a", "x"), _chunk("b", "y")], [[1.0, 0.0], [0.0, 1.0]])
        assert store.get_all_ids() == {"a", "b"}

    def test_get_all_chunks_empty_store(self) -> None:
        assert _store().get_all_chunks() == []

    def test_get_all_chunks_returns_everything_indexed(self) -> None:
        store = _store()
        store.upsert([_chunk("a", "x"), _chunk("b", "y")], [[1.0, 0.0], [0.0, 1.0]])
        ids = {c.id for c in store.get_all_chunks()}
        assert ids == {"a", "b"}

    def test_get_all_embeddings_empty_store(self) -> None:
        assert _store().get_all_embeddings() == []

    def test_get_all_embeddings_returns_plain_lists(self) -> None:
        store = _store()
        store.upsert([_chunk("a", "x")], [[1.0, 0.0]])
        embeddings = store.get_all_embeddings()
        assert embeddings == [[1.0, 0.0]]
        assert isinstance(embeddings, list) and isinstance(embeddings[0], list)
