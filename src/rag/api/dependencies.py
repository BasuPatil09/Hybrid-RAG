from __future__ import annotations

import hmac
from dataclasses import dataclass

import chromadb
from fastapi import Depends, Header, HTTPException, Request

from rag.config import Settings
from rag.generation.confidence import CompletenessJudge, get_completeness_judge
from rag.generation.generator import Generator, get_generator
from rag.generation.verification import CitationVerifier, get_citation_verifier
from rag.indexing.embeddings import Embedder, get_embedder
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.ingestion.chunking import BaseChunker, get_chunker
from rag.ingestion.loaders import DocumentLoaderRegistry
from rag.observability import ObservabilityLog
from rag.retrieval.reranker import CrossEncoderReranker, Reranker, load_cross_encoder


@dataclass
class AppComponents:
    settings: Settings
    embedder: Embedder
    vector_store: ChromaVectorStore
    sparse_index: SparseIndex
    reranker: Reranker
    generator: Generator
    verifier: CitationVerifier
    completeness_judge: CompletenessJudge
    registry: DocumentLoaderRegistry
    chunker: BaseChunker
    observability_log: ObservabilityLog


def build_components(settings: Settings) -> AppComponents:

    embedder = get_embedder(
        settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        device=settings.embedding_device,
    )
    vector_store = ChromaVectorStore(
        chromadb.PersistentClient(path=str(settings.chroma_persist_dir)),
        settings.chroma_collection_name,
    )
    sparse_index = SparseIndex()
    sparse_index.rebuild(vector_store.get_all_chunks())

    reranker = CrossEncoderReranker(load_cross_encoder(settings.reranker_model_name))
    generator = get_generator(
        settings.generation_provider,
        api_key=settings.groq_api_key,
        model=settings.generation_model,
    )
    verifier = get_citation_verifier(
        settings.generation_provider,
        api_key=settings.groq_api_key,
        model=settings.generation_model,
    )
    completeness_judge = get_completeness_judge(
        settings.generation_provider,
        api_key=settings.groq_api_key,
        model=settings.generation_model,
    )

    registry = DocumentLoaderRegistry(pdf_min_page_chars=settings.pdf_min_page_chars)
    chunker = get_chunker(
        settings.chunking_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        header_levels=settings.header_levels,
    )
    observability_log = ObservabilityLog(settings.observability_log_path)

    return AppComponents(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        sparse_index=sparse_index,
        reranker=reranker,
        generator=generator,
        verifier=verifier,
        completeness_judge=completeness_judge,
        registry=registry,
        chunker=chunker,
        observability_log=observability_log,
    )

def get_components(request: Request) -> AppComponents:
    return request.app.state.components

def require_api_key(
    components: AppComponents = Depends(get_components),
    x_api_key: str | None = Header(default=None),
) -> None:
    configured_key = components.settings.app_api_key
    if configured_key and not hmac.compare_digest(x_api_key or "", configured_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
