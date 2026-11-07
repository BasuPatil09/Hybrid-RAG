from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from rag.config import Settings
from rag.eval.golden_qa import GoldenQuestion
from rag.eval.runner import _behavior_matches
from rag.generation.confidence import CompletenessJudge
from rag.generation.generator import Generator
from rag.generation.verification import CitationVerifier
from rag.indexing.embeddings import Embedder
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.pipeline import answer_question
from rag.retrieval.reranker import Reranker
from rag.tracing import Tracer

logger = logging.getLogger(__name__)


class ContextRecord(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    section: str | None = None
    score: float
    content: str


class RunRecord(BaseModel):
    run_id: str
    question_id: str
    category: str
    expected_behavior: str
    question: str
    reference: str = ""
    answer: str
    is_insufficient: bool
    cited_chunk_ids: list[str]
    contexts: list[ContextRecord]
    citation_accuracy: float
    retrieval_confidence: float
    answer_completeness: float
    composite_confidence: float
    latency_ms: float
    trace_id: str | None = None


def behavior_matched(record: RunRecord) -> bool:
    return _behavior_matches(record.expected_behavior, record.is_insufficient)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def config_snapshot(settings: Settings) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "git_commit": commit,
        "generator": f"{settings.generation_provider}/{settings.generation_model}",
        "judge": f"{settings.judge_provider}/{settings.judge_model}",
        "judge_thinking_level": settings.judge_thinking_level,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model_name,
        "chunking": f"{settings.chunking_strategy}:{settings.chunk_size}/{settings.chunk_overlap}",
        "dense_top_k": settings.dense_top_k,
        "sparse_top_k": settings.sparse_top_k,
        "rrf": [settings.rrf_k, settings.rrf_dense_weight, settings.rrf_sparse_weight],
        "rerank_top_n": settings.rerank_top_n,
    }


def iter_run_records(
    questions: Sequence[GoldenQuestion],
    *,
    run_id: str,
    embedder: Embedder,
    vector_store: ChromaVectorStore,
    sparse_index: SparseIndex,
    reranker: Reranker,
    generator: Generator,
    verifier: CitationVerifier,
    completeness_judge: CompletenessJudge,
    tracer: Tracer,
    snapshot: dict[str, object],
    dense_top_k: int,
    sparse_top_k: int,
    rrf_k: int,
    rrf_dense_weight: float,
    rrf_sparse_weight: float,
    rerank_top_n: int,
    on_error: Callable[[str, Exception], None] | None = None,
) -> Iterator[RunRecord]:
    for q in questions:
        start = time.perf_counter()
        try:
            with tracer.scope(
                tags=["eval", f"eval-run:{run_id}"],
                metadata={
                    "eval_run_id": run_id,
                    "question_id": q.id,
                    "category": q.category,
                    "expected_behavior": q.expected_behavior,
                    **snapshot,
                },
            ):
                answered = answer_question(
                    q.question,
                    embedder,
                    vector_store,
                    sparse_index,
                    reranker,
                    generator,
                    verifier,
                    completeness_judge,
                    dense_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                    rrf_k=rrf_k,
                    rrf_dense_weight=rrf_dense_weight,
                    rrf_sparse_weight=rrf_sparse_weight,
                    top_n=rerank_top_n,
                )
        except Exception as exc:  # noqa: BLE001 - one bad question must not kill the run
            logger.warning("Question %s failed: %s", q.id, exc)
            if on_error:
                on_error(q.id, exc)
            continue

        yield RunRecord(
            run_id=run_id,
            question_id=q.id,
            category=q.category,
            expected_behavior=q.expected_behavior,
            question=q.question,
            reference=q.reference,
            answer=answered.answer.answer,
            is_insufficient=answered.answer.is_insufficient,
            cited_chunk_ids=answered.answer.cited_chunk_ids,
            contexts=[
                ContextRecord(
                    chunk_id=sc.chunk.id,
                    source=sc.chunk.metadata.source,
                    page=sc.chunk.metadata.page,
                    section=sc.chunk.metadata.section,
                    score=sc.score,
                    content=sc.chunk.content,
                )
                for sc in answered.reranked
            ],
            citation_accuracy=answered.verified.citation_accuracy,
            retrieval_confidence=answered.confidence.retrieval_confidence,
            answer_completeness=answered.confidence.answer_completeness,
            composite_confidence=answered.confidence.composite,
            latency_ms=(time.perf_counter() - start) * 1000,
            trace_id=answered.trace_id,
        )


def append_record(path: Path, record: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def read_records(path: Path) -> list[RunRecord]:
    with open(path, encoding="utf-8") as f:
        return [RunRecord.model_validate_json(line) for line in f if line.strip()]


def latest_run_dir(runs_dir: Path) -> Path | None:
    candidates = sorted(
        (d for d in runs_dir.glob("*") if (d / "records.jsonl").exists()),
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None
