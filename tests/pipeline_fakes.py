from __future__ import annotations

import uuid

import chromadb

from rag.generation.generator import GeneratedAnswer
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.retrieval.reranker import Reranker
from rag.schemas import Chunk, ChunkMetadata, ScoredChunk
from tests.fakes import FakeEmbedder

CORPUS = {
    ("guide.pdf", 1): "gradient descent updates parameters along the negative gradient",
    ("guide.pdf", 2): "the learning rate controls the step size of every update",
    ("other.pdf", 1): "baking bread requires flour water yeast and patience",
    ("other.pdf", 2): "the stock market closed higher on strong quarterly earnings",
    ("other.pdf", 3): "migratory birds travel thousands of miles each winter",
}


def make_chunk(source: str, page: int, text: str) -> Chunk:
    return Chunk(
        id=f"{source}::p{page}::c0",
        content=text,
        metadata=ChunkMetadata(
            source=source, format="pdf", page=page, chunk_index=0, char_count=len(text)
        ),
    )


class FakeReranker(Reranker):
    def rerank(self, query, candidates, top_n):
        return [ScoredChunk(chunk=c.chunk, score=1.0) for c in candidates][:top_n]


class FakeGenerator:
    def generate(self, query, chunks):
        if "unanswerable" in query.lower():
            return GeneratedAnswer(
                answer="INSUFFICIENT_CONTEXT: not covered.",
                is_insufficient=True,
                cited_chunk_ids=[],
            )
        return GeneratedAnswer(
            answer="A real answer [1].",
            is_insufficient=False,
            cited_chunk_ids=[chunks[0].id] if chunks else [],
        )


class FakeVerifier:
    def verify_claim(self, claim_text, source_text):
        return True, "fake supported"


class FakeCompletenessJudge:
    def judge(self, query, answer_text):
        return 0.9, "fake complete"


def make_components() -> dict:
    embedder = FakeEmbedder()
    store = ChromaVectorStore(chromadb.EphemeralClient(), f"pf_{uuid.uuid4().hex[:10]}")
    chunks = [make_chunk(s, p, t) for (s, p), t in CORPUS.items()]
    store.upsert(chunks, embedder.embed([c.content for c in chunks]))
    sparse = SparseIndex()
    sparse.rebuild(chunks)
    return {
        "embedder": embedder,
        "vector_store": store,
        "sparse_index": sparse,
        "reranker": FakeReranker(),
        "generator": FakeGenerator(),
        "verifier": FakeVerifier(),
        "completeness_judge": FakeCompletenessJudge(),
    }


RETRIEVAL_KWARGS = {
    "dense_top_k": 10,
    "sparse_top_k": 10,
    "rrf_k": 60,
    "rrf_dense_weight": 0.7,
    "rrf_sparse_weight": 0.3,
}
