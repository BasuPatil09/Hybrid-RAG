from __future__ import annotations

import csv
import json

import pytest

from rag.api.dependencies import AppComponents
from rag.config import Settings
from rag.eval import ragas_scoring
from rag.eval.cli import main
from rag.eval.ragas_scoring import RagasRow
from rag.eval.run_records import read_records
from rag.ingestion.chunking import get_chunker
from rag.ingestion.loaders import DocumentLoaderRegistry
from rag.observability import ObservabilityLog
from rag.tracing import LangSmithTracer, NullTracer
from tests.pipeline_fakes import CORPUS, make_components
from tests.test_tracing import RecordingClient

_GOLDEN_HEADER = (
    "id,category,question,expected_behavior,source_hint,your_correction,approved\n"
)
_LABEL_HEADER = "question_id,source,page,llm_relevant,human_relevant,snippet\n"
_Q1 = CORPUS[("guide.pdf", 1)]
_Q3 = CORPUS[("guide.pdf", 2)]


class _JudgeStub:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return "RELEVANT: 1"


@pytest.fixture
def env(tmp_path):
    golden = tmp_path / "golden.csv"
    golden.write_text(
        _GOLDEN_HEADER
        + f"q1,direct_lookup,{_Q1},answerable,h,,yes\n"
        + "q2,unanswerable,an unanswerable question,insufficient_context,h,,yes\n"
        + f"q3,multi_hop,{_Q3},answerable,h,,yes\n"
    )
    settings = Settings(
        _env_file=None,
        golden_qa_path=golden,
        eval_labels_path=tmp_path / "labels.csv",
        eval_runs_dir=tmp_path / "runs",
        groq_api_key="g",
        gemini_api_key="k",
        rerank_top_n=3,
        eval_k_values="1,3,5",
    )
    recording = RecordingClient()
    components = AppComponents(
        settings=settings,
        registry=DocumentLoaderRegistry(),
        chunker=get_chunker("structure_aware", chunk_size=200, chunk_overlap=20),
        observability_log=ObservabilityLog(),
        tracer=NullTracer(),
        **make_components(),
    )

    class Env:
        pass

    e = Env()
    e.settings, e.components, e.recording, e.tmp = (
        settings,
        components,
        recording,
        tmp_path,
    )
    e.lines: list[str] = []
    e.judge = _JudgeStub()

    def run(*argv: str) -> int:
        return main(
            list(argv),
            build=lambda s: components,
            settings_factory=lambda: settings,
            judge_factory=lambda s: e.judge,
            out=e.lines.append,
        )

    e.run = run
    e.text = lambda: "\n".join(e.lines)
    e.label_sheet = lambda body: settings.eval_labels_path.write_text(
        _LABEL_HEADER + body
    )
    e.runs = lambda: sorted(settings.eval_runs_dir.glob("*"))
    return e


class TestLabelDraft:
    def test_drafts_only_answerable_questions_one_call_each(self, env) -> None:
        assert env.run("label-draft") == 0
        with open(env.settings.eval_labels_path, newline="", encoding="utf-8-sig") as f:
            ids = {row["question_id"] for row in csv.DictReader(f)}
        assert ids == {"q1", "q3"}
        assert env.judge.calls == 2

    def test_draft_never_counts_as_ground_truth_until_a_human_reviews_it(
        self, env
    ) -> None:
        env.run("label-draft")
        assert env.run("retrieval") == 0
        assert "No labeled questions" in env.text()

    def test_rerun_does_not_redraft_or_touch_reviewed_rows(self, env) -> None:
        env.run("label-draft")
        before = env.settings.eval_labels_path.read_text()
        calls = env.judge.calls
        env.run("label-draft")
        assert env.settings.eval_labels_path.read_text() == before
        assert env.judge.calls == calls
        assert "Nothing to draft" in env.text()

    def test_new_questions_are_appended_without_losing_existing_rows(self, env) -> None:
        env.label_sheet("q1,guide.pdf,1,yes,yes,kept\n")
        assert env.run("label-draft") == 0
        sheet = env.settings.eval_labels_path.read_text()
        assert "kept" in sheet and sheet.count("question_id") == 1
        assert "q3" in sheet and env.judge.calls == 1

    def test_force_starts_over(self, env) -> None:
        env.label_sheet("q1,guide.pdf,1,yes,yes,kept\n")
        env.run("label-draft", "--force")
        assert "kept" not in env.settings.eval_labels_path.read_text()
        assert env.judge.calls == 2

    def test_requires_a_gemini_key(self, env) -> None:
        env.settings.gemini_api_key = ""
        assert env.run("label-draft") == 2
        assert "GEMINI_API_KEY" in env.text()


class TestRetrieval:
    def test_scores_reviewed_labels_and_saves_a_report(self, env) -> None:
        env.label_sheet(
            "q1,guide.pdf,1,yes,yes,s\n"
            "q3,guide.pdf,2,yes,yes,s\n"
            "q3,other.pdf,3,yes,yes,s\n"
        )
        assert env.run("retrieval") == 0
        text = env.text()
        for stage in ("dense", "sparse", "fused", "reranked"):
            assert stage in text
        (run_dir,) = env.runs()
        report = json.loads((run_dir / "retrieval_report.json").read_text())
        assert report["n_evaluated"] == 2
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["config"]["generator"] and "retrieval" in summary

    def test_missing_label_sheet_gives_actionable_error(self, env) -> None:
        assert env.run("retrieval") == 2
        assert "label-draft" in env.text()

    def test_empty_index_is_rejected(self, env) -> None:
        env.components.vector_store._collection.delete(
            ids=list(env.components.vector_store.get_all_ids())
        )
        assert env.run("retrieval") == 2
        assert "Nothing indexed" in env.text()


class TestRun:
    def test_writes_records_summary_and_retrieval_report(self, env) -> None:
        env.label_sheet("q1,guide.pdf,1,yes,yes,s\n")
        assert env.run("run") == 0
        (run_dir,) = env.runs()
        records = read_records(run_dir / "records.jsonl")
        assert [r.question_id for r in records] == ["q1", "q2", "q3"]
        assert (run_dir / "retrieval_report.json").exists()
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["run"]["n_completed"] == 3
        assert summary["run"]["behavior_match_rate"] == 1.0
        assert "behavior match 100%" in env.text()

    def test_limit_restricts_questions(self, env) -> None:
        env.run("run", "--limit", "1")
        (run_dir,) = env.runs()
        assert len(read_records(run_dir / "records.jsonl")) == 1

    def test_without_tracing_it_says_so_and_still_works(self, env) -> None:
        env.run("run", "--limit", "1")
        assert "tracing is off" in env.text()

    def test_scores_are_attached_to_each_questions_trace(self, env) -> None:
        env.label_sheet("q1,guide.pdf,1,yes,yes,s\n")
        env.components.tracer = LangSmithTracer(env.recording, "proj")
        assert env.run("run") == 0

        (run_dir,) = env.runs()
        records = {r.question_id: r for r in read_records(run_dir / "records.jsonl")}
        by_trace: dict[str, dict[str, float]] = {}
        for fb in env.recording.recorded_feedback:
            by_trace.setdefault(fb["run_id"], {})[fb["key"]] = fb["score"]

        q1 = by_trace[records["q1"].trace_id]
        assert q1["behavior_match"] == 1.0
        assert q1["retrieval.recall@3"] == 1.0 and q1["retrieval.mrr"] == 1.0
        assert "citation_accuracy" in q1
        q2 = by_trace[records["q2"].trace_id]
        assert "citation_accuracy" not in q2 and "retrieval.mrr" not in q2


class TestRagas:
    @pytest.fixture
    def scored_calls(self, monkeypatch):
        calls: list[list[str]] = []

        async def fake_run_ragas(records, *, settings, embedder, metric_names, on_row):
            calls.append([r.question_id for r in records])
            rows = [
                RagasRow(
                    run_id=r.run_id,
                    question_id=r.question_id,
                    category=r.category,
                    trace_id=r.trace_id,
                    scores={m: 0.5 for m in metric_names},
                )
                for r in records
            ]
            for row in rows:
                on_row(row)
            return rows

        monkeypatch.setattr(ragas_scoring, "run_ragas", fake_run_ragas)
        return calls

    def test_scores_only_answered_questions_and_writes_a_summary(
        self, env, scored_calls
    ) -> None:
        env.run("run")
        assert env.run("ragas") == 0
        assert scored_calls == [["q1", "q3"]]
        (run_dir,) = env.runs()
        summary = json.loads((run_dir / "ragas_summary.json").read_text())
        assert summary["n_questions"] == 2
        assert summary["overall"]["faithfulness"]["mean"] == 0.5
        assert "estimated judge requests" in env.text()

    def test_second_invocation_does_not_pay_to_rescore(self, env, scored_calls) -> None:
        env.run("run")
        env.run("ragas")
        env.run("ragas")
        assert len(scored_calls) == 1
        assert "Nothing to score" in env.text()

    def test_failed_rows_are_retried_and_good_rows_kept(
        self, env, scored_calls
    ) -> None:
        env.run("run")
        env.run("ragas")
        (run_dir,) = env.runs()
        path = run_dir / "ragas_scores.jsonl"
        rows = [
            RagasRow.model_validate_json(line) for line in path.read_text().splitlines()
        ]
        rows[1] = rows[1].model_copy(update={"errors": {"faithfulness": "quota"}})
        path.write_text("".join(r.model_dump_json() + "\n" for r in rows))

        env.run("ragas")
        assert scored_calls[-1] == ["q3"]
        final = [
            RagasRow.model_validate_json(line) for line in path.read_text().splitlines()
        ]
        assert sorted(r.question_id for r in final) == ["q1", "q3"]
        assert all(not r.errors for r in final)

    def test_rescore_flag_ignores_saved_scores(self, env, scored_calls) -> None:
        env.run("run")
        env.run("ragas")
        env.run("ragas", "--rescore")
        assert scored_calls[-1] == ["q1", "q3"]

    def test_metrics_flag_overrides_defaults(self, env, scored_calls) -> None:
        env.run("run")
        env.run("ragas", "--metrics", "faithfulness")
        (run_dir,) = env.runs()
        summary = json.loads((run_dir / "ragas_summary.json").read_text())
        assert list(summary["overall"]) == ["faithfulness"]

    def test_ragas_scores_are_attached_to_traces(self, env, scored_calls) -> None:
        env.components.tracer = LangSmithTracer(env.recording, "proj")
        env.run("run")
        env.recording.recorded_feedback.clear()
        env.run("ragas", "--metrics", "faithfulness")
        keys = {fb["key"] for fb in env.recording.recorded_feedback}
        assert keys == {"ragas.faithfulness"}

    def test_no_records_gives_actionable_error(self, env, scored_calls) -> None:
        assert env.run("ragas") == 2
        assert "eval_cli.py run" in env.text()

    def test_missing_ragas_install_points_at_the_extra(self, env, monkeypatch) -> None:
        env.run("run")

        async def not_installed(*a, **k):
            raise ImportError("No module named 'ragas'")

        monkeypatch.setattr(ragas_scoring, "run_ragas", not_installed)
        assert env.run("ragas") == 2
        assert ".[eval]" in env.text()
