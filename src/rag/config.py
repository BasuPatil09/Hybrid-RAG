from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str = Field(
        default="", description="Required for generation (the only supported provider)"
    )
    gemini_api_key: str = Field(
        default="",
        description="Required for the judge models when JUDGE_PROVIDER=gemini",
    )
    app_api_key: str = Field(
        default="",
        description="Optional shared secret for /v1/*; empty disables API authentication",
    )
    environment: Literal["development", "production"] = "development"
    console_enabled: bool = True
    console_allowed_hosts: str = "localhost,127.0.0.1,[::1]"
    embedding_provider: Literal["local"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_device: str | None = None
    generation_provider: Literal["groq"] = "groq"
    generation_model: str = "llama-3.3-70b-versatile"

    judge_provider: Literal["groq", "gemini"] = "gemini"
    judge_model: str = "gemini-3.5-flash-lite"
    judge_thinking_level: Literal["", "minimal", "low", "medium", "high"] = "low"
    judge_max_attempts: int = 4
    judge_timeout_seconds: float = 60.0
    judge_requests_per_minute: int = 0

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

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "hybrid-rag"
    langsmith_endpoint: str = ""
    trace_include_content: bool = True

    eval_runs_dir: Path = Path("./data/eval_runs")
    eval_labels_path: Path = Path("./eval/retrieval_labels.csv")
    eval_k_values: str = "1,3,5,10,20"
    ragas_metrics: str = "faithfulness,answer_relevancy,context_precision"
    ragas_judge_model: str = ""
    ragas_max_concurrency: int = 2

    observability_log_path: Path = Path("./data/observability/requests.jsonl")

    @model_validator(mode="after")
    def _judge_model_matches_provider(self) -> "Settings":
        if self.judge_provider == "groq" and self.judge_model.lower().startswith(
            "gemini"
        ):
            raise ValueError(
                "JUDGE_PROVIDER=groq needs a Groq JUDGE_MODEL "
                f"(got Gemini model id {self.judge_model!r})."
            )
        return self

    @property
    def eval_k_list(self) -> tuple[int, ...]:
        return tuple(
            sorted({int(k) for k in self.eval_k_values.split(",") if k.strip()})
        )

    @property
    def ragas_metric_list(self) -> tuple[str, ...]:
        return tuple(m.strip() for m in self.ragas_metrics.split(",") if m.strip())

    @property
    def tracing_active(self) -> bool:
        return self.langsmith_tracing and bool(self.langsmith_api_key)

    @property
    def console_host_allowlist(self) -> frozenset[str]:
        return frozenset(
            h.strip().lower().strip("[]")
            for h in self.console_allowed_hosts.split(",")
            if h.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
