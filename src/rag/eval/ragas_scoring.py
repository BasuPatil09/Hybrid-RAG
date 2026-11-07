from __future__ import annotations

import asyncio
import inspect
import math
import os
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel

from rag.config import Settings
from rag.eval.run_records import RunRecord
from rag.indexing.embeddings import Embedder

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

RAGAS_METRICS: dict[str, tuple[str, bool]] = {
    "faithfulness": ("Faithfulness", False),
    "answer_relevancy": ("AnswerRelevancy", True),
    "context_precision": ("ContextPrecisionWithoutReference", False),
    "context_relevance": ("ContextRelevance", False),
    "context_precision_ref": ("ContextPrecisionWithReference", False),
    "context_recall": ("ContextRecall", False),
    "factual_correctness": ("FactualCorrectness", False),
}


class RagasRow(BaseModel):
    run_id: str
    question_id: str
    category: str
    trace_id: str | None = None
    scores: dict[str, float | None]
    errors: dict[str, str] = {}


class AsyncRateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def wait(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            now = self._clock()
            start = max(now, self._next_slot)
            self._next_slot = start + self._interval
        if start > now:
            await self._sleep(start - now)

    async def hook(self, request: Any) -> None:
        await self.wait()


def make_local_embeddings(embedder: Embedder) -> Any:
    from ragas.embeddings.base import BaseRagasEmbedding

    class LocalEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embedder.embed([text])[0]

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return embedder.embed(list(texts))

        async def aembed_texts(
            self, texts: list[str], **kwargs: Any
        ) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts)

    return LocalEmbeddings()


def build_metrics(names: Sequence[str], llm: Any, embeddings: Any) -> dict[str, Any]:
    unknown = [n for n in names if n not in RAGAS_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown RAGAS metric(s) {unknown}; choose from {sorted(RAGAS_METRICS)}"
        )
    from ragas.metrics import collections as ragas_metrics

    built: dict[str, Any] = {}
    for name in names:
        class_name, needs_embeddings = RAGAS_METRICS[name]
        cls = getattr(ragas_metrics, class_name)
        built[name] = (
            cls(llm=llm, embeddings=embeddings) if needs_embeddings else cls(llm=llm)
        )
    return built


def _inputs_for(metric: Any, record: RunRecord) -> dict[str, Any] | None:
    available: dict[str, Any] = {
        "user_input": record.question,
        "response": record.answer,
        "retrieved_contexts": [c.content for c in record.contexts],
        "reference": record.reference,
    }
    kwargs: dict[str, Any] = {}
    for name, param in inspect.signature(metric.ascore).parameters.items():
        if name not in available:
            if param.default is inspect.Parameter.empty:
                raise ValueError(
                    f"{type(metric).__name__}.ascore needs unsupported field {name!r}"
                )
            continue
        if not available[name]:
            return None
        kwargs[name] = available[name]
    return kwargs


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


async def ascore_records(
    records: Sequence[RunRecord],
    metrics: dict[str, Any],
    *,
    max_concurrency: int = 2,
    on_row: Callable[[RagasRow], None] | None = None,
) -> list[RagasRow]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def score_one(
        name: str, metric: Any, record: RunRecord
    ) -> tuple[str, float | None, str | None]:
        kwargs = _inputs_for(metric, record)
        if kwargs is None:
            return name, None, None
        async with semaphore:
            try:
                result = await metric.ascore(**kwargs)
            except Exception as exc:  # noqa: BLE001 - one metric must not sink the run
                return name, None, f"{type(exc).__name__}: {exc}"[:300]
        return name, _to_float(getattr(result, "value", result)), None

    async def score_record(record: RunRecord) -> RagasRow:
        outcomes = await asyncio.gather(
            *(score_one(n, m, record) for n, m in metrics.items())
        )
        row = RagasRow(
            run_id=record.run_id,
            question_id=record.question_id,
            category=record.category,
            trace_id=record.trace_id,
            scores={n: v for n, v, _ in outcomes},
            errors={n: e for n, _, e in outcomes if e},
        )
        if on_row:
            on_row(row)
        return row

    answered = [r for r in records if not r.is_insufficient and r.contexts]
    return list(await asyncio.gather(*(score_record(r) for r in answered)))


async def run_ragas(
    records: Sequence[RunRecord],
    *,
    settings: Settings,
    embedder: Embedder,
    metric_names: Sequence[str],
    on_row: Callable[[RagasRow], None] | None = None,
) -> list[RagasRow]:
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    import httpx
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    limiter = AsyncRateLimiter(settings.judge_requests_per_minute)
    async with httpx.AsyncClient(
        event_hooks={"request": [limiter.hook]},
        timeout=settings.judge_timeout_seconds,
    ) as http_client:
        client = AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url=GEMINI_OPENAI_BASE_URL,
            http_client=http_client,
            max_retries=5,
        )
        llm_options: dict[str, Any] = {"max_tokens": 4096}
        if settings.judge_thinking_level:
            llm_options["reasoning_effort"] = settings.judge_thinking_level
        llm = llm_factory(
            settings.ragas_judge_model or settings.judge_model,
            client=client,
            **llm_options,
        )
        metrics = build_metrics(metric_names, llm, make_local_embeddings(embedder))
        return await ascore_records(
            records,
            metrics,
            max_concurrency=settings.ragas_max_concurrency,
            on_row=on_row,
        )


def summarize_ragas(rows: Sequence[RagasRow]) -> dict[str, Any]:
    def aggregate(subset: Sequence[RagasRow]) -> dict[str, dict[str, float | int]]:
        names = sorted({n for r in subset for n in r.scores})
        out: dict[str, dict[str, float | int]] = {}
        for name in names:
            values = [r.scores[name] for r in subset if r.scores.get(name) is not None]
            out[name] = {
                "mean": sum(values) / len(values) if values else float("nan"),
                "n_scored": len(values),
                "n_failed": sum(1 for r in subset if name in r.errors),
            }
        return out

    by_category: dict[str, list[RagasRow]] = defaultdict(list)
    for row in rows:
        by_category[row.category].append(row)
    return {
        "n_questions": len(rows),
        "overall": aggregate(rows),
        "by_category": {c: aggregate(r) for c, r in by_category.items()},
    }
