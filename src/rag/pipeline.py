from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from rag.generation.confidence import CompletenessJudge, ConfidenceScore, score_answer
from rag.generation.generator import GeneratedAnswer, Generator
from rag.generation.verification import CitationVerifier, VerifiedAnswer, verify_answer
from rag.indexing.dedup import find_duplicate_indices
from rag.indexing.embeddings import Embedder
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.ingestion.chunking import BaseChunker
from rag.ingestion.loaders import DocumentLoaderRegistry
from rag.retrieval.dense import dense_retrieve
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.reranker import Reranker
from rag.retrieval.sparse import sparse_retrieve
from rag.schemas import ScoredChunk

class IngestionReport(BaseModel):
    documents_loaded: int
    chunks_created: int
    chunks_skipped_already_indexed: int
    duplicates_skipped: int
    chunks_indexed: int
    stopped_early: str | None = None

def run_ingestion(
    corpus_dir: Path,
    registry: DocumentLoaderRegistry,
    chunker: BaseChunker,
    embedder: Embedder,
    vector_store: ChromaVectorStore,
    sparse_index: SparseIndex,
    dedup_threshold: float,
    checkpoint_every: int = 100,
) -> IngestionReport:

    documents = registry.load_corpus(corpus_dir)
    chunks = chunker.chunk(documents)

    if not chunks:
        sparse_index.rebuild(vector_store.get_all_chunks())
        return IngestionReport(
            documents_loaded=len(documents),
            chunks_created=0,
            chunks_skipped_already_indexed=0,
            duplicates_skipped=0,
            chunks_indexed=vector_store.count(),
        )

    already_indexed_ids = vector_store.get_all_ids()
    new_chunks = [c for c in chunks if c.id not in already_indexed_ids]
    already_indexed_count = len(chunks) - len(new_chunks)

    duplicates_skipped = 0
    stopped_early: str | None = None
    for start in range(0, len(new_chunks), checkpoint_every):
        group = new_chunks[start : start + checkpoint_every]
        try:
            group_embeddings = embedder.embed([c.content for c in group])
        except Exception as exc:
            stopped_early = f"{type(exc).__name__}: {exc}"
            break

        existing_embeddings = vector_store.get_all_embeddings()
        duplicate_indices = find_duplicate_indices(
            group_embeddings, existing_embeddings, threshold=dedup_threshold
        )
        kept_chunks = [c for i, c in enumerate(group) if i not in duplicate_indices]
        kept_embeddings = [
            e for i, e in enumerate(group_embeddings) if i not in duplicate_indices
        ]

        vector_store.upsert(kept_chunks, kept_embeddings)
        duplicates_skipped += len(duplicate_indices)

    sparse_index.rebuild(vector_store.get_all_chunks())

    return IngestionReport(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        chunks_skipped_already_indexed=already_indexed_count,
        duplicates_skipped=duplicates_skipped,
        chunks_indexed=vector_store.count(),
        stopped_early=stopped_early,
    )


def retrieve(
    query: str,
    embedder: Embedder,
    vector_store: ChromaVectorStore,
    sparse_index: SparseIndex,
    *,
    dense_top_k: int,
    sparse_top_k: int,
    rrf_k: int,
    rrf_dense_weight: float,
    rrf_sparse_weight: float,
) -> list[ScoredChunk]:

    dense_results = dense_retrieve(query, embedder, vector_store, top_k=dense_top_k)
    sparse_results = sparse_retrieve(query, sparse_index, top_k=sparse_top_k)
    return reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        k=rrf_k,
        dense_weight=rrf_dense_weight,
        sparse_weight=rrf_sparse_weight,
    )


class AnsweredQuestion(BaseModel):
    query: str
    answer: GeneratedAnswer
    verified: VerifiedAnswer
    confidence: ConfidenceScore
    reranked: list[ScoredChunk]

    model_config = {"arbitrary_types_allowed": True}


def answer_question(
    query: str,
    embedder: Embedder,
    vector_store: ChromaVectorStore,
    sparse_index: SparseIndex,
    reranker: Reranker,
    generator: Generator,
    verifier: CitationVerifier,
    completeness_judge: CompletenessJudge,
    *,
    dense_top_k: int,
    sparse_top_k: int,
    rrf_k: int,
    rrf_dense_weight: float,
    rrf_sparse_weight: float,
    rerank_candidate_pool: int = 20,
    top_n: int = 5,
) -> AnsweredQuestion:

    fused = retrieve(
        query,
        embedder,
        vector_store,
        sparse_index,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        rrf_k=rrf_k,
        rrf_dense_weight=rrf_dense_weight,
        rrf_sparse_weight=rrf_sparse_weight,
    )
    reranked = reranker.rerank(query, fused[:rerank_candidate_pool], top_n=top_n)
    chunks = [r.chunk for r in reranked]

    answer = generator.generate(query, chunks)
    verified = verify_answer(answer, chunks, verifier)
    confidence = score_answer(query, answer, verified, reranked, completeness_judge)

    return AnsweredQuestion(
        query=query,
        answer=answer,
        verified=verified,
        confidence=confidence,
        reranked=reranked,
    )
