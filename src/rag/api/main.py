from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag.api.dependencies import (
    AppComponents,
    build_components,
    get_components,
    require_api_key,
)
from rag.config import get_settings
from rag.eval.golden_qa import load_golden_questions
from rag.eval.runner import EvalResult, EvalSummary, run_eval_suite, summarize
from rag.observability import (
    ObservabilityEvent,
    ObservabilitySummary,
    now_iso,
    redact_error,
)
from rag.observability import summarize as summarize_observability
from rag.pipeline import (
    AnsweredQuestion,
    IngestionReport,
    answer_question,
    run_ingestion,
)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.components = build_components(get_settings())
    yield

def _docs_urls(environment: str) -> tuple[str | None, str | None]:
    return (None, None) if environment == "production" else ("/docs", "/redoc")

settings = get_settings()
docs_url, redoc_url = _docs_urls(settings.environment)
app = FastAPI(
    title="RAG Pipeline",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=docs_url,
    redoc_url=redoc_url,
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/health")
def health_check() -> dict[str, str]:

    return {"status": "ok"}

class AskRequest(BaseModel):
    question: str = Field(max_length=settings.ask_max_question_length)


@app.post(
    "/v1/ask", response_model=AnsweredQuestion, dependencies=[Depends(require_api_key)]
)
@limiter.limit(lambda: get_settings().ask_rate_limit)
def ask(
    request: Request,
    ask_request: AskRequest,
    components: AppComponents = Depends(get_components),
) -> AnsweredQuestion:
    if not ask_request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")
    if not components.settings.groq_api_key:
        raise HTTPException(
            status_code=503, detail="GROQ_API_KEY is not configured on the server."
        )
    if components.vector_store.count() == 0:
        raise HTTPException(
            status_code=503, detail="Nothing indexed yet -- call POST /v1/ingest first."
        )

    settings = components.settings
    start = time.perf_counter()
    try:
        result = answer_question(
            ask_request.question,
            components.embedder,
            components.vector_store,
            components.sparse_index,
            components.reranker,
            components.generator,
            components.verifier,
            components.completeness_judge,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            rrf_k=settings.rrf_k,
            rrf_dense_weight=settings.rrf_dense_weight,
            rrf_sparse_weight=settings.rrf_sparse_weight,
            top_n=settings.rerank_top_n,
        )
    except Exception as exc:
        components.observability_log.record(
            ObservabilityEvent(
                timestamp=now_iso(),
                event_type="ask",
                latency_ms=(time.perf_counter() - start) * 1000,
                success=False,
                error=redact_error(exc, settings.groq_api_key, settings.app_api_key),
                question=ask_request.question,
            )
        )
        raise

    components.observability_log.record(
        ObservabilityEvent(
            timestamp=now_iso(),
            event_type="ask",
            latency_ms=(time.perf_counter() - start) * 1000,
            success=True,
            question=ask_request.question,
            is_insufficient=result.answer.is_insufficient,
            retrieval_confidence=result.confidence.retrieval_confidence,
            citation_coverage=result.confidence.citation_coverage,
            answer_completeness=result.confidence.answer_completeness,
            composite_confidence=result.confidence.composite,
            claims_verified=len(result.verified.claim_verdicts),
            claims_supported=sum(
                1 for v in result.verified.claim_verdicts if v.supported
            ),
        )
    )
    return result

@app.post(
    "/v1/ingest",
    response_model=IngestionReport,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(lambda: get_settings().ingest_rate_limit)
def ingest(
    request: Request, components: AppComponents = Depends(get_components)
) -> IngestionReport:
    settings = components.settings
    start = time.perf_counter()
    try:
        report = run_ingestion(
            settings.corpus_dir,
            components.registry,
            components.chunker,
            components.embedder,
            components.vector_store,
            components.sparse_index,
            dedup_threshold=settings.dedup_similarity_threshold,
            checkpoint_every=settings.ingestion_checkpoint_every,
        )
    except Exception as exc:
        components.observability_log.record(
            ObservabilityEvent(
                timestamp=now_iso(),
                event_type="ingest",
                latency_ms=(time.perf_counter() - start) * 1000,
                success=False,
                error=redact_error(exc, settings.groq_api_key, settings.app_api_key),
            )
        )
        raise

    components.sparse_index.rebuild(components.vector_store.get_all_chunks())
    components.observability_log.record(
        ObservabilityEvent(
            timestamp=now_iso(),
            event_type="ingest",
            latency_ms=(time.perf_counter() - start) * 1000,
            success=True,
            chunks_indexed=report.chunks_indexed,
        )
    )
    return report

class DocumentSummary(BaseModel):
    source: str
    chunk_count: int

class DocumentsResponse(BaseModel):
    total_chunks: int
    documents: list[DocumentSummary]

@app.get(
    "/v1/documents",
    response_model=DocumentsResponse,
    dependencies=[Depends(require_api_key)],
)
def list_documents(
    components: AppComponents = Depends(get_components),
) -> DocumentsResponse:
    counts: dict[str, int] = {}
    for chunk in components.vector_store.get_all_chunks():
        counts[chunk.metadata.source] = counts.get(chunk.metadata.source, 0) + 1
    documents = [
        DocumentSummary(source=source, chunk_count=n)
        for source, n in sorted(counts.items())
    ]
    return DocumentsResponse(total_chunks=sum(counts.values()), documents=documents)


class EvalResponse(BaseModel):
    summary: EvalSummary
    results: list[EvalResult]

@app.post(
    "/v1/eval", response_model=EvalResponse, dependencies=[Depends(require_api_key)]
)
@limiter.limit(lambda: get_settings().eval_rate_limit)
def run_eval(
    request: Request, components: AppComponents = Depends(get_components)
) -> EvalResponse:

    if not components.settings.groq_api_key:
        raise HTTPException(
            status_code=503, detail="GROQ_API_KEY is not configured on the server."
        )
    if components.vector_store.count() == 0:
        raise HTTPException(
            status_code=503, detail="Nothing indexed yet -- call POST /v1/ingest first."
        )
    questions = load_golden_questions(components.settings.golden_qa_path)
    if not questions:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No approved questions found in {components.settings.golden_qa_path}. "
                f"Mark at least one row's 'approved' column 'yes' first -- unreviewed "
                f"drafts don't count as ground truth."
            ),
        )
    settings = components.settings
    start = time.perf_counter()
    try:
        results = run_eval_suite(
            questions,
            components.embedder,
            components.vector_store,
            components.sparse_index,
            components.reranker,
            components.generator,
            components.verifier,
            components.completeness_judge,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            rrf_k=settings.rrf_k,
            rrf_dense_weight=settings.rrf_dense_weight,
            rrf_sparse_weight=settings.rrf_sparse_weight,
            rerank_top_n=settings.rerank_top_n,
        )
    except Exception as exc:
        components.observability_log.record(
            ObservabilityEvent(
                timestamp=now_iso(),
                event_type="eval",
                latency_ms=(time.perf_counter() - start) * 1000,
                success=False,
                error=redact_error(exc, settings.groq_api_key, settings.app_api_key),
            )
        )
        raise

    eval_summary = summarize(results)
    components.observability_log.record(
        ObservabilityEvent(
            timestamp=now_iso(),
            event_type="eval",
            latency_ms=(time.perf_counter() - start) * 1000,
            success=True,
            eval_questions=eval_summary.total_questions,
            eval_behavior_match_rate=eval_summary.behavior_match_rate,
        )
    )
    return EvalResponse(summary=eval_summary, results=results)


class ObservabilityResponse(BaseModel):
    summary: ObservabilitySummary
    recent_events: list[ObservabilityEvent]


@app.get(
    "/v1/observability",
    response_model=ObservabilityResponse,
    dependencies=[Depends(require_api_key)],
)
def get_observability(
    limit: int = 200, components: AppComponents = Depends(get_components)
) -> ObservabilityResponse:

    events = components.observability_log.read_recent(limit=limit)
    return ObservabilityResponse(
        summary=summarize_observability(events), recent_events=list(reversed(events))
    )

app.mount(
    "/", StaticFiles(directory="static", html=True, check_dir=False), name="static"
)
