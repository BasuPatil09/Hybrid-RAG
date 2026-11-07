from __future__ import annotations

from rag.eval.golden_qa import GoldenQuestion
from rag.eval.run_records import (
    append_record,
    iter_run_records,
    latest_run_dir,
    read_records,
)
from rag.tracing import LangSmithTracer, NullTracer
from tests.pipeline_fakes import CORPUS, RETRIEVAL_KWARGS, make_components
from tests.test_tracing import RecordingClient


def _q(qid, text, behavior="answerable", reference=""):
    return GoldenQuestion(
        id=qid,
        category="direct_lookup",
        question=text,
        expected_behavior=behavior,
        reference=reference,
    )


def _collect(questions, tracer=None, **overrides):
    components = {**make_components(), **overrides}
    errors: list[tuple[str, Exception]] = []
    records = list(
        iter_run_records(
            questions,
            run_id="run-1",
            tracer=tracer or NullTracer(),
            snapshot={"generator": "fake"},
            rerank_top_n=3,
            on_error=lambda qid, exc: errors.append((qid, exc)),
            **components,
            **RETRIEVAL_KWARGS,
        )
    )
    return records, errors


class TestIterRunRecords:
    def test_record_carries_everything_scoring_needs(self) -> None:
        text = CORPUS[("guide.pdf", 1)]
        (record,), _ = _collect([_q("q1", text, reference="ref answer")])

        assert record.run_id == "run-1" and record.question_id == "q1"
        assert record.question == text and record.reference == "ref answer"
        assert record.answer == "A real answer [1]."
        assert record.is_insufficient is False
        assert record.cited_chunk_ids == [record.contexts[0].chunk_id]
        assert len(record.contexts) == 3
        assert record.contexts[0].source and record.contexts[0].content
        assert record.latency_ms > 0
        assert record.trace_id is None

    def test_declined_question_is_recorded_as_insufficient(self) -> None:
        (record,), _ = _collect(
            [_q("q2", "an unanswerable question", "insufficient_context")]
        )
        assert record.is_insufficient is True and record.cited_chunk_ids == []

    def test_one_failing_question_is_reported_and_skipped_not_fatal(self) -> None:
        class FlakyGenerator:
            def generate(self, query, chunks):
                if "boom" in query:
                    raise RuntimeError("rate limited")
                return make_components()["generator"].generate(query, chunks)

        records, errors = _collect(
            [_q("q1", "boom"), _q("q2", "gradient descent")],
            generator=FlakyGenerator(),
        )
        assert [r.question_id for r in records] == ["q2"]
        assert errors[0][0] == "q1" and "rate limited" in str(errors[0][1])

    def test_each_question_gets_its_own_trace_tagged_with_run_and_config(self) -> None:
        client = RecordingClient()
        records, _ = _collect(
            [_q("q1", "gradient descent"), _q("q2", "learning rate")],
            tracer=LangSmithTracer(client, "proj"),
        )
        roots = [r for r in client.recorded_runs if r["name"] == "rag.answer_question"]
        assert len(roots) == 2
        assert {r["metadata"]["question_id"] for r in roots} == {"q1", "q2"}
        assert all("eval-run:run-1" in r["tags"] for r in roots)
        assert all(r["metadata"]["generator"] == "fake" for r in roots)
        assert [r.trace_id for r in records] == [r["id"] for r in roots]


class TestPersistence:
    def test_records_round_trip_through_jsonl(self, tmp_path) -> None:
        records, _ = _collect([_q("q1", "gradient descent"), _q("q2", "learning rate")])
        path = tmp_path / "run" / "records.jsonl"
        for r in records:
            append_record(path, r)
        assert read_records(path) == records

    def test_latest_run_dir_ignores_dirs_without_records(self, tmp_path) -> None:
        for name, has_records in [
            ("20260101-000000", True),
            ("20260202-000000", True),
            ("20260303-000000", False),
        ]:
            d = tmp_path / name
            d.mkdir()
            if has_records:
                (d / "records.jsonl").write_text("")
        assert latest_run_dir(tmp_path).name == "20260202-000000"

    def test_latest_run_dir_is_none_when_nothing_ran(self, tmp_path) -> None:
        assert latest_run_dir(tmp_path) is None
