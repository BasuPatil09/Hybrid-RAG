from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag.api.dependencies import AppComponents, build_components
from rag.config import Settings, get_settings
from rag.eval.golden_qa import GoldenQuestion, load_golden_questions
from rag.eval.label_drafting import draft_labels_for_question, pool_candidates
from rag.eval.relevance import (
    existing_question_ids,
    load_relevance_labels,
    write_draft_labels,
)
from rag.eval.retrieval_eval import (
    RetrievalReport,
    evaluate_retrieval,
    format_retrieval_report,
)
from rag.eval.run_records import (
    RunRecord,
    append_record,
    behavior_matched,
    config_snapshot,
    iter_run_records,
    latest_run_dir,
    new_run_id,
    read_records,
)
from rag.generation.gemini_client import TextClient
from rag.generation.judges import build_gemini_client
from rag.pipeline import retrieve_stages

Out = Callable[[str], None]

_CALLS_PER_SAMPLE = {
    "faithfulness": 2,
    "answer_relevancy": 1,
    "context_precision": 5,
    "context_relevance": 1,
    "context_precision_ref": 5,
    "context_recall": 1,
    "factual_correctness": 2,
}


class CliError(Exception):
    """A problem the user can fix; printed without a traceback."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _update_summary(run_dir: Path, **sections: Any) -> None:
    path = run_dir / "summary.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(sections)
    _write_json(path, current)


def _questions(settings: Settings, *, answerable_only: bool = False, limit=None):
    questions = load_golden_questions(settings.golden_qa_path)
    if answerable_only:
        questions = [q for q in questions if q.expected_behavior == "answerable"]
    if not questions:
        raise CliError(f"No approved questions in {settings.golden_qa_path}.")
    return questions[:limit] if limit else questions


def _require_labels(settings: Settings):
    if not settings.eval_labels_path.exists():
        raise CliError(
            f"{settings.eval_labels_path} not found. Create the review sheet with "
            "`label-draft`, then mark `human_relevant` = yes on the pages that "
            "really answer each question."
        )
    return load_relevance_labels(settings.eval_labels_path)


def _retrieval(
    settings: Settings, c: AppComponents, questions: list[GoldenQuestion], labels
) -> RetrievalReport:
    return evaluate_retrieval(
        questions,
        labels,
        c.embedder,
        c.vector_store,
        c.sparse_index,
        c.reranker,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        rrf_dense_weight=settings.rrf_dense_weight,
        rrf_sparse_weight=settings.rrf_sparse_weight,
        rerank_top_n=settings.rerank_top_n,
        k_values=settings.eval_k_list,
    )


def _misses_text(report: RetrievalReport) -> list[str]:
    lines = []
    for row in report.per_question:
        missed = row.stages["reranked"].missed_units
        if missed:
            lines.append(f"  {row.question_id}: not retrieved -> {missed}")
    return ["\nRelevant pages missing from the reranked pool:", *lines] if lines else []


def cmd_label_draft(
    args, settings: Settings, c: AppComponents, out: Out, judge: TextClient
) -> None:
    path = settings.eval_labels_path
    if args.force and path.exists():
        path.unlink()
    done = existing_question_ids(path)
    todo = [q for q in _questions(settings, answerable_only=True) if q.id not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        out(f"Nothing to draft: every answerable question already has rows in {path}.")
        return
    if not settings.gemini_api_key:
        raise CliError("GEMINI_API_KEY is required to draft labels.")

    out(f"Drafting labels for {len(todo)} questions (1 judge call each)...")
    rows: list[dict[str, object]] = []
    for q in todo:
        stages = retrieve_stages(
            q.question,
            c.embedder,
            c.vector_store,
            c.sparse_index,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            rrf_k=settings.rrf_k,
            rrf_dense_weight=settings.rrf_dense_weight,
            rrf_sparse_weight=settings.rrf_sparse_weight,
        )
        candidates = pool_candidates(stages, pool_size=20, max_candidates=15)
        drafted = draft_labels_for_question(q, candidates, judge)
        out(f"  {q.id}: {sum(r['llm_relevant'] == 'yes' for r in drafted)} suggested")
        rows.extend(drafted)

    n = write_draft_labels(path, rows, append=bool(done))
    out(
        f"\nWrote {n} rows to {path}.\nNEXT: open it, set `human_relevant` to yes/no "
        "per row (skim the 'no' rows too -- the LLM can miss), save, then run "
        "`retrieval`. Rows you leave blank do not count as ground truth."
    )


def cmd_retrieval(args, settings: Settings, c: AppComponents, out: Out) -> None:
    labels = _require_labels(settings)
    questions = _questions(settings, limit=args.limit)
    report = _retrieval(settings, c, questions, labels)

    run_dir = settings.eval_runs_dir / new_run_id()
    _write_json(run_dir / "retrieval_report.json", report.model_dump())
    _update_summary(
        run_dir,
        config=config_snapshot(settings),
        retrieval={"n_evaluated": report.n_evaluated, "summary": report.summary},
    )
    out(format_retrieval_report(report))
    for line in _misses_text(report):
        out(line)
    out(f"\nSaved to {run_dir}")


def cmd_run(args, settings: Settings, c: AppComponents, out: Out) -> None:
    questions = _questions(settings, limit=args.limit)
    run_id = new_run_id()
    run_dir = settings.eval_runs_dir / run_id
    snapshot = config_snapshot(settings)
    if not c.tracer.enabled:
        out(
            "(LangSmith tracing is off; set LANGSMITH_TRACING=true + LANGSMITH_API_KEY)"
        )

    failures: list[str] = []
    records: list[RunRecord] = []
    out(f"Run {run_id}: {len(questions)} questions")
    for record in iter_run_records(
        questions,
        run_id=run_id,
        embedder=c.embedder,
        vector_store=c.vector_store,
        sparse_index=c.sparse_index,
        reranker=c.reranker,
        generator=c.generator,
        verifier=c.verifier,
        completeness_judge=c.completeness_judge,
        tracer=c.tracer,
        snapshot=snapshot,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        rrf_dense_weight=settings.rrf_dense_weight,
        rrf_sparse_weight=settings.rrf_sparse_weight,
        rerank_top_n=settings.rerank_top_n,
        on_error=lambda qid, exc: failures.append(f"{qid}: {exc}"),
    ):
        append_record(run_dir / "records.jsonl", record)
        records.append(record)
        mark = "ok " if behavior_matched(record) else "MISS"
        out(
            f"  [{mark}] {record.question_id} {record.category:<13} "
            f"cit={record.citation_accuracy:.0%} comp={record.composite_confidence:.0%}"
        )

    matched = sum(behavior_matched(r) for r in records)
    summary: dict[str, Any] = {
        "n_questions": len(questions),
        "n_completed": len(records),
        "failed": failures,
        "behavior_match_rate": matched / len(records) if records else 0.0,
    }

    report = None
    if settings.eval_labels_path.exists():
        labels = load_relevance_labels(settings.eval_labels_path)
        report = _retrieval(settings, c, questions, labels)
        _write_json(run_dir / "retrieval_report.json", report.model_dump())
        summary["retrieval"] = report.summary
        out("\n" + format_retrieval_report(report))
    _update_summary(run_dir, config=snapshot, run=summary)

    if c.tracer.enabled:
        _log_run_feedback(c, records, report, settings.rerank_top_n)
        c.tracer.flush()
    for failure in failures:
        out(f"FAILED {failure}")
    out(f"\nbehavior match {summary['behavior_match_rate']:.0%} | saved to {run_dir}")
    out(f"Next: python scripts/eval_cli.py ragas --run {run_dir}")


def _log_run_feedback(
    c: AppComponents,
    records: list[RunRecord],
    report: RetrievalReport | None,
    final_k: int,
) -> None:
    """Attach every score to its trace, so LangSmith can filter/sort/chart them."""
    by_question = {r.question_id: r for r in (report.per_question if report else [])}
    for rec in records:
        if not rec.trace_id:
            continue
        c.tracer.log_feedback(
            rec.trace_id, "behavior_match", float(behavior_matched(rec))
        )
        c.tracer.log_feedback(
            rec.trace_id, "composite_confidence", rec.composite_confidence
        )
        if not rec.is_insufficient:
            c.tracer.log_feedback(
                rec.trace_id, "citation_accuracy", rec.citation_accuracy
            )
        row = by_question.get(rec.question_id)
        if row:
            metrics = row.stages["reranked"].metrics
            for key in (
                f"recall@{final_k}",
                f"precision@{final_k}",
                f"hit@{final_k}",
                "mrr",
            ):
                if key in metrics:
                    c.tracer.log_feedback(
                        rec.trace_id, f"retrieval.{key}", metrics[key]
                    )


def cmd_ragas(args, settings: Settings, c: AppComponents, out: Out) -> None:
    from rag.eval.ragas_scoring import RagasRow, run_ragas, summarize_ragas

    run_dir = Path(args.run) if args.run else latest_run_dir(settings.eval_runs_dir)
    if run_dir is None or not (run_dir / "records.jsonl").exists():
        raise CliError("No run records found. Run `eval_cli.py run` first.")
    if not settings.gemini_api_key:
        raise CliError("GEMINI_API_KEY is required (the RAGAS judge is Gemini).")

    metric_names = (
        tuple(m.strip() for m in args.metrics.split(",") if m.strip())
        if args.metrics
        else settings.ragas_metric_list
    )
    records = read_records(run_dir / "records.jsonl")
    answered = [r for r in records if not r.is_insufficient and r.contexts]

    scores_path = run_dir / "ragas_scores.jsonl"
    kept: list[RagasRow] = []
    if scores_path.exists() and not args.rescore:
        kept = [
            r
            for r in (
                RagasRow.model_validate_json(line)
                for line in scores_path.read_text().splitlines()
                if line.strip()
            )
            if not r.errors and set(metric_names) <= set(r.scores)
        ]
    done = {r.question_id for r in kept}
    todo = [r for r in answered if r.question_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    scores_path.write_text("".join(r.model_dump_json() + "\n" for r in kept))

    requests = sum(_CALLS_PER_SAMPLE.get(m, 2) for m in metric_names) * len(todo)
    rpm = settings.judge_requests_per_minute
    eta = f", ~{requests / rpm:.0f} min at {rpm} RPM" if rpm else ""
    out(
        f"RAGAS on {run_dir.name}: {len(todo)} to score ({len(done)} already done), "
        f"{len(records) - len(answered)} declined skipped.\n"
        f"metrics={list(metric_names)} judge={settings.ragas_judge_model or settings.judge_model}"
        f"\nestimated judge requests: ~{requests}{eta}"
    )
    if not todo:
        out("Nothing to score.")
        return

    def persist(row: RagasRow) -> None:
        with open(scores_path, "a", encoding="utf-8") as f:
            f.write(row.model_dump_json() + "\n")
        flag = " (errors)" if row.errors else ""
        out(f"  scored {row.question_id}{flag}")

    new_rows = asyncio.run(
        run_ragas(
            todo,
            settings=settings,
            embedder=c.embedder,
            metric_names=metric_names,
            on_row=persist,
        )
    )
    rows = kept + new_rows
    summary = summarize_ragas(rows)
    _write_json(run_dir / "ragas_summary.json", summary)
    _update_summary(run_dir, ragas=summary)

    out(f"\nRAGAS means over {summary['n_questions']} answered questions:")
    for name, agg in summary["overall"].items():
        out(
            f"  {name:<24}{agg['mean']:.3f}   (scored {agg['n_scored']}, failed {agg['n_failed']})"
        )
    for row in new_rows:
        for metric, score in row.scores.items():
            if row.trace_id and score is not None:
                c.tracer.log_feedback(row.trace_id, f"ragas.{metric}", score)
    c.tracer.flush()
    out(f"\nSaved to {run_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_cli", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "label-draft", help="LLM-draft retrieval labels for human review"
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true", help="discard the existing sheet")

    p = sub.add_parser(
        "retrieval", help="Recall/Precision/MRR/nDCG per retrieval stage"
    )
    p.add_argument("--limit", type=int)

    p = sub.add_parser("run", help="run the full pipeline over the golden set")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("ragas", help="score a run's records with RAGAS")
    p.add_argument("--run", help="run directory (default: latest)")
    p.add_argument("--metrics", help="comma-separated; default from RAGAS_METRICS")
    p.add_argument("--limit", type=int)
    p.add_argument("--rescore", action="store_true", help="ignore saved scores")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    build: Callable[[Settings], AppComponents] = build_components,
    settings_factory: Callable[[], Settings] = get_settings,
    judge_factory: Callable[[Settings], TextClient] = build_gemini_client,
    out: Out = print,
) -> int:
    args = _parser().parse_args(argv)
    settings = settings_factory()
    try:
        components = build(settings)
        if components.vector_store.count() == 0:
            raise CliError("Nothing indexed yet -- run ingestion first.")
        if args.command == "label-draft":
            cmd_label_draft(args, settings, components, out, judge_factory(settings))
        elif args.command == "retrieval":
            cmd_retrieval(args, settings, components, out)
        elif args.command == "run":
            cmd_run(args, settings, components, out)
        else:
            cmd_ragas(args, settings, components, out)
    except CliError as exc:
        out(f"error: {exc}")
        return 2
    except ImportError as exc:
        out(f'error: {exc}. Install the eval extras: pip install -e ".[eval]"')
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
