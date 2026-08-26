from __future__ import annotations

import uuid

import chromadb

from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.retrieval.dense import dense_retrieve
from rag.retrieval.sparse import sparse_retrieve
from rag.schemas import Chunk, ChunkMetadata
from tests.fakes import FakeEmbedder


def _chunk(cid: str, content: str) -> Chunk:
    return Chunk(
        id=cid,
        content=content,
        metadata=ChunkMetadata(
            source="f.md", format="md", chunk_index=0, char_count=len(content)
        ),
    )


def _fresh_store() -> ChromaVectorStore:
    return ChromaVectorStore(
        chromadb.EphemeralClient(), f"test_{uuid.uuid4().hex[:12]}"
    )


class TestDenseRetrieve:
    def test_query_text_retrieves_the_chunk_with_identical_content(self) -> None:

        embedder = FakeEmbedder()
        store = _fresh_store()
        target_text = "the exact text we will query for"
        chunks = [
            _chunk("target", target_text),
            _chunk("other", "something unrelated entirely"),
        ]
        store.upsert(chunks, embedder.embed([c.content for c in chunks]))

        results = dense_retrieve(target_text, embedder, store, top_k=2)

        assert results[0].chunk.id == "target"

    def test_respects_top_k(self) -> None:
        embedder = FakeEmbedder()
        store = _fresh_store()
        chunks = [_chunk(str(i), f"content number {i}") for i in range(5)]
        store.upsert(chunks, embedder.embed([c.content for c in chunks]))

        assert len(dense_retrieve("content", embedder, store, top_k=2)) == 2


class TestSparseRetrieve:
    def test_delegates_to_sparse_index(self) -> None:

        index = SparseIndex()
        index.rebuild(
            [
                _chunk("a", "gradient descent optimizes model parameters"),
                _chunk("b", "The weather today is sunny with a chance of rain"),
                _chunk("c", "Baking bread requires flour, water, yeast, and time"),
                _chunk("d", "The stock market closed higher today on strong earnings"),
                _chunk("e", "Migratory birds travel thousands of miles each winter"),
            ]
        )
        results = sparse_retrieve("gradient descent", index, top_k=5)
        assert results[0].chunk.id == "a"
