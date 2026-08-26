from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str = Field(
        default="", description="Required for generation (the only supported provider)"
    )
    app_api_key: str = Field(
        default="",
        description="Optional shared secret for /v1/*; empty disables API authentication",
    )
    environment: Literal["development", "production"] = "development"
    embedding_provider: Literal["local"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_device: str | None = None
    generation_provider: Literal["groq"] = "groq"
    generation_model: str = "llama-3.3-70b-versatile"

    chroma_collection_name: str = "rag_chunks"
    chroma_persist_dir: Path = Path("./data/chroma")
    corpus_dir: Path = Path("./data/corpus")
    pdf_min_page_chars: int = 20
    ingestion_checkpoint_every: int = 100
    chunking_strategy: Literal["structure_aware", "fixed_size"] = "structure_aware"
    chunk_size: int = 800
    chunk_overlap: int = 120
    header_levels: tuple[str, ...] = ("#", "##", "###")
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    dedup_similarity_threshold: float = 0.95
    rerank_top_n: int = 5
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ask_max_question_length: int = 2000
    ask_rate_limit: str = "30/minute"
    ingest_rate_limit: str = "6/hour"
    eval_rate_limit: str = "2/hour"
    golden_qa_path: Path = Path("./eval/golden_qa_draft.csv")
    observability_log_path: Path = Path("./data/observability/requests.jsonl")

@lru_cache
def get_settings() -> Settings:
    return Settings()
