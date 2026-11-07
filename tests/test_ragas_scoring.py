from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from rag.eval.ragas_scoring import (
    RAGAS_METRICS,
    AsyncRateLimiter,
    RagasRow,
    _inputs_for,
    ascore_records,
    build_metrics,
    summarize_ragas,
)
from rag.eval.run_records import ContextRecord, RunRecord
from tests.fakes import FakeEmbedder


def _record(
    qid="q1", *, declined=False, reference="", contexts=2, category="direct_lookup"
):
    return RunRecord(
        run_id="r",
        question_id=qid,
        category=category,
        expected_behavior="answerable",
        question="What is X?",
        reference=reference,
        answer="INSUFFICIENT_CONTEXT: no" if declined else "X is Y [1].",
        is_insufficient=declined,
        cited_chunk_ids=[],
        contexts=[
            ContextRecord(chunk_id=f"c{i}", source="s", score=1.0, content=f"ctx {i}")
            for i in range(contexts)
        ],
        citation_accuracy=1.0,
        retrieval_confidence=0.5,
        answer_completeness=1.0,
        composite_confidence=0.8,
        latency_ms=1.0,
        trace_id="t-" + qid,
    )


class _Metric:
    def __init__(self, value=0.5, error: Exception | None = None, delay=0.0):
        self.value, self.error, self.delay = value, error, delay
        self.calls: list[dict] = []
        self.active = 0
        self.peak = 0

    async def ascore(self, user_input: str, response: str, retrieved_contexts: list):
        self.calls.append(
            {"user_input": user_input, "response": response, "ctx": retrieved_contexts}
        )
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        if self.error:
            raise self.error
        return SimpleNamespace(value=self.value)


class _RefMetric:
    async def ascore(self, user_input: str, reference: str, retrieved_contexts: list):
        return SimpleNamespace(value=1.0)


def _run(records, metrics, **kw):
    return asyncio.run(ascore_records(records, metrics, **kw))


class TestAscoreRecords:
    def test_scores_every_metric_for_each_answered_record(self) -> None:
        rows = _run(
            [_record("q1"), _record("q2")], {"f": _Metric(0.8), "g": _Metric(0.4)}
        )
        assert [r.question_id for r in rows] == ["q1", "q2"]
        assert rows[0].scores == {"f": 0.8, "g": 0.4} and rows[0].trace_id == "t-q1"

    def test_declined_answers_are_not_scored(self) -> None:
        metric = _Metric()
        rows = _run([_record("q1"), _record("q2", declined=True)], {"f": metric})
        assert [r.question_id for r in rows] == ["q1"] and len(metric.calls) == 1

    def test_records_without_contexts_are_not_scored(self) -> None:
        assert _run([_record(contexts=0)], {"f": _Metric()}) == []

    def test_metric_receives_exactly_what_the_generator_saw(self) -> None:
        metric = _Metric()
        _run([_record()], {"f": metric})
        assert metric.calls[0] == {
            "user_input": "What is X?",
            "response": "X is Y [1].",
            "ctx": ["ctx 0", "ctx 1"],
        }

    def test_one_failing_metric_is_recorded_and_does_not_sink_the_others(self) -> None:
        rows = _run(
            [_record()],
            {"bad": _Metric(error=RuntimeError("quota")), "good": _Metric(0.9)},
        )
        assert rows[0].scores == {"bad": None, "good": 0.9}
        assert "RuntimeError: quota" in rows[0].errors["bad"]

    def test_nan_scores_become_none(self) -> None:
        rows = _run([_record()], {"f": _Metric(float("nan"))})
        assert rows[0].scores["f"] is None

    def test_reference_based_metric_is_skipped_without_a_reference(self) -> None:
        rows = _run(
            [_record("q1"), _record("q2", reference="the gold answer")],
            {"recall": _RefMetric()},
        )
        assert rows[0].scores["recall"] is None and rows[1].scores["recall"] == 1.0
        assert rows[0].errors == {}

    def test_concurrency_is_bounded(self) -> None:
        metric = _Metric(delay=0.01)
        _run([_record(f"q{i}") for i in range(6)], {"f": metric}, max_concurrency=2)
        assert metric.peak <= 2

    def test_on_row_callback_fires_per_record(self) -> None:
        seen: list[str] = []
        _run(
            [_record("q1"), _record("q2")],
            {"f": _Metric()},
            on_row=lambda r: seen.append(r.question_id),
        )
        assert sorted(seen) == ["q1", "q2"]

    def test_metric_needing_an_unknown_field_fails_loudly(self) -> None:
        class Weird:
            async def ascore(self, mystery_field: str): ...

        with pytest.raises(ValueError, match="mystery_field"):
            _inputs_for(Weird(), _record())


class TestSummarize:
    def test_means_skip_failures_but_count_them(self) -> None:
        rows = [
            RagasRow(run_id="r", question_id="a", category="x", scores={"f": 1.0}),
            RagasRow(run_id="r", question_id="b", category="x", scores={"f": 0.5}),
            RagasRow(
                run_id="r",
                question_id="c",
                category="y",
                scores={"f": None},
                errors={"f": "boom"},
            ),
        ]
        s = summarize_ragas(rows)
        assert s["overall"]["f"] == {"mean": 0.75, "n_scored": 2, "n_failed": 1}
        assert s["by_category"]["x"]["f"]["mean"] == 0.75
        assert math.isnan(s["by_category"]["y"]["f"]["mean"])


class TestRateLimiter:
    def test_spaces_requests_by_the_configured_rate(self) -> None:
        now = [0.0]
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        limiter = AsyncRateLimiter(10, clock=lambda: now[0], sleep=sleep)

        async def go():
            for _ in range(3):
                await limiter.wait()

        asyncio.run(go())
        assert sleeps == [pytest.approx(6.0), pytest.approx(6.0)]

    def test_zero_disables_it(self) -> None:
        sleeps: list[float] = []

        async def sleep(s: float) -> None:
            sleeps.append(s)

        limiter = AsyncRateLimiter(0, sleep=sleep)
        asyncio.run(limiter.wait())
        assert sleeps == []


@pytest.fixture(scope="module")
def llm_and_embeddings():
    pytest.importorskip("ragas")
    pytest.importorskip("openai")
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    from rag.eval.ragas_scoring import GEMINI_OPENAI_BASE_URL, make_local_embeddings

    client = AsyncOpenAI(api_key="fake", base_url=GEMINI_OPENAI_BASE_URL)
    llm = llm_factory("gemini-3.5-flash-lite", client=client, max_tokens=64)
    return llm, make_local_embeddings(FakeEmbedder())


class TestAgainstRealRagas:
    def test_every_registered_metric_builds(self, llm_and_embeddings) -> None:
        llm, embeddings = llm_and_embeddings
        metrics = build_metrics(list(RAGAS_METRICS), llm, embeddings)
        assert set(metrics) == set(RAGAS_METRICS)

    def test_every_metric_can_be_fed_from_a_run_record(
        self, llm_and_embeddings
    ) -> None:
        llm, embeddings = llm_and_embeddings
        record = _record(reference="gold")
        for name, metric in build_metrics(list(RAGAS_METRICS), llm, embeddings).items():
            assert _inputs_for(metric, record) is not None, name

    def test_reference_metrics_are_not_applicable_without_a_reference(
        self, llm_and_embeddings
    ) -> None:
        llm, embeddings = llm_and_embeddings
        metrics = build_metrics(["context_recall", "faithfulness"], llm, embeddings)
        record = _record(reference="")
        assert _inputs_for(metrics["context_recall"], record) is None
        assert _inputs_for(metrics["faithfulness"], record) is not None

    def test_local_embeddings_adapter_embeds_sync_async_and_batch(
        self, llm_and_embeddings
    ) -> None:
        _, embeddings = llm_and_embeddings
        one = embeddings.embed_text("hello")
        assert len(one) == FakeEmbedder().dimensions
        assert asyncio.run(embeddings.aembed_text("hello")) == one
        batch = asyncio.run(embeddings.aembed_texts(["hello", "world"]))
        assert batch[0] == one and len(batch) == 2

    def test_unknown_metric_name_lists_valid_choices(self) -> None:
        with pytest.raises(ValueError, match="faithfulness"):
            build_metrics(["nonsense"], llm=None, embeddings=None)
