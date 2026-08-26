from __future__ import annotations

from pathlib import Path

import pytest

from rag.observability import (
    ObservabilityEvent,
    ObservabilityLog,
    ObservabilitySummary,
    summarize,
)


def _ask_event(
    *,
    success: bool = True,
    latency_ms: float = 100.0,
    is_insufficient: bool = False,
    composite_confidence: float | None = 0.8,
    error: str | None = None,
) -> ObservabilityEvent:
    return ObservabilityEvent(
        timestamp="2026-01-01T00:00:00+00:00",
        event_type="ask",
        latency_ms=latency_ms,
        success=success,
        error=error,
        question="a question",
        is_insufficient=is_insufficient,
        retrieval_confidence=0.9,
        citation_coverage=0.7,
        answer_completeness=0.85,
        composite_confidence=composite_confidence,
        claims_verified=3,
        claims_supported=2,
    )


class TestObservabilityLogInMemory:
    def test_records_and_reads_back_an_event(self) -> None:
        log = ObservabilityLog()
        log.record(_ask_event())
        events = log.read_recent()
        assert len(events) == 1
        assert events[0].event_type == "ask"

    def test_read_recent_respects_limit_and_keeps_most_recent(self) -> None:
        log = ObservabilityLog()
        for i in range(5):
            log.record(_ask_event(latency_ms=float(i)))
        recent = log.read_recent(limit=2)
        assert [e.latency_ms for e in recent] == [3.0, 4.0]

    def test_no_path_means_no_file_written(self, tmp_path: Path) -> None:

        log = ObservabilityLog(path=None)
        log.record(_ask_event())
        assert list(tmp_path.iterdir()) == []


class TestObservabilityLogPersistence:
    def test_events_survive_across_separate_instances_pointed_at_same_path(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "requests.jsonl"
        first = ObservabilityLog(path=path)
        first.record(_ask_event(latency_ms=42.0))

        second = ObservabilityLog(path=path)
        events = second.read_recent()

        assert len(events) == 1
        assert events[0].latency_ms == 42.0

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "requests.jsonl"
        ObservabilityLog(path=path)
        assert path.parent.exists()

    def test_appends_rather_than_overwrites(self, tmp_path: Path) -> None:
        path = tmp_path / "requests.jsonl"
        log = ObservabilityLog(path=path)
        log.record(_ask_event(latency_ms=1.0))
        log.record(_ask_event(latency_ms=2.0))

        reloaded = ObservabilityLog(path=path)
        assert [e.latency_ms for e in reloaded.read_recent()] == [1.0, 2.0]

    def test_empty_or_missing_file_reads_back_as_no_events(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "does_not_exist_yet.jsonl"
        log = ObservabilityLog(path=path)
        assert log.read_recent() == []


class TestSummarize:
    def test_empty_events_returns_zeroed_summary(self) -> None:
        summary = summarize([])
        assert summary == ObservabilitySummary(
            total_events=0,
            ask_count=0,
            ingest_count=0,
            eval_count=0,
            error_count=0,
            mean_latency_ms=0.0,
            mean_composite_confidence=0.0,
            insufficient_context_rate=0.0,
        )

    def test_counts_events_by_type(self) -> None:
        events = [
            _ask_event(),
            _ask_event(),
            ObservabilityEvent(
                timestamp="t",
                event_type="ingest",
                latency_ms=10.0,
                success=True,
                chunks_indexed=100,
            ),
            ObservabilityEvent(
                timestamp="t",
                event_type="eval",
                latency_ms=10.0,
                success=True,
                eval_questions=27,
            ),
        ]
        summary = summarize(events)
        assert summary.total_events == 4
        assert summary.ask_count == 2
        assert summary.ingest_count == 1
        assert summary.eval_count == 1

    def test_error_count_includes_any_event_type(self) -> None:
        events = [
            _ask_event(success=True),
            _ask_event(success=False, error="boom", composite_confidence=None),
            ObservabilityEvent(
                timestamp="t",
                event_type="ingest",
                latency_ms=1.0,
                success=False,
                error="also boom",
            ),
        ]
        assert summarize(events).error_count == 2

    def test_mean_latency_averages_across_all_events_regardless_of_type(self) -> None:
        events = [
            _ask_event(latency_ms=100.0),
            ObservabilityEvent(
                timestamp="t", event_type="ingest", latency_ms=300.0, success=True
            ),
        ]
        assert summarize(events).mean_latency_ms == 200.0

    def test_mean_composite_confidence_only_counts_successful_ask_events(self) -> None:
        events = [
            _ask_event(composite_confidence=0.8),
            _ask_event(composite_confidence=0.4),
            _ask_event(success=False, composite_confidence=None),
            ObservabilityEvent(
                timestamp="t", event_type="ingest", latency_ms=1.0, success=True
            ),
        ]
        assert summarize(events).mean_composite_confidence == pytest.approx(0.6)

    def test_mean_composite_confidence_is_zero_when_no_successful_asks(self) -> None:
        events = [
            ObservabilityEvent(
                timestamp="t", event_type="ingest", latency_ms=1.0, success=True
            )
        ]
        assert summarize(events).mean_composite_confidence == 0.0

    def test_insufficient_context_rate_only_counts_successful_ask_events(self) -> None:
        events = [
            _ask_event(is_insufficient=True),
            _ask_event(is_insufficient=True),
            _ask_event(is_insufficient=False),
            _ask_event(is_insufficient=False),
        ]
        assert summarize(events).insufficient_context_rate == 0.5

    def test_insufficient_context_rate_is_zero_when_no_successful_asks(self) -> None:
        events = [
            ObservabilityEvent(
                timestamp="t", event_type="ingest", latency_ms=1.0, success=True
            )
        ]
        assert summarize(events).insufficient_context_rate == 0.0
