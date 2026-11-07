from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from rag.eval.golden_qa import GoldenQuestion
from rag.eval.relevance import RelevanceLabels
from rag.eval.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.indexing.embeddings import Embedder
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.pipeline import retrieve_stages
from rag.retrieval.reranker import Reranker
from rag.schemas import ScoredChunk

STAGE_NAMES = ("dense", "sparse", "fused", "reranked")


class StageResult(BaseModel):
    ranked_chunk_ids: list[str]
    metrics: dict[str, float]
    missed_units: list[str]


class QuestionRetrieval(BaseModel):
    question_id: str
    category: str
    question: str
    n_relevant_units: int
    n_relevant_chunks: int
    stages: dict[str, StageResult]


class RetrievalReport(BaseModel):
    n_evaluated: int
    skipped_no_labels: list[str]
    unmatched_labels: list[str]
    k_values: list[int]
    final_context_k: int
    summary: dict[str, dict[str, float]]
    summary_by_category: dict[str, dict[str, dict[str, float]]]
    per_question: list[QuestionRetrieval]


def _score_ranking(
    ranking: Sequence[ScoredChunk],
    question_id: str,
    labels: RelevanceLabels,
    n_relevant_chunks: int,
    ks: Sequence[int],
) -> StageResult:
    matches = [labels.matches(question_id, sc.chunk) for sc in ranking]
    relevant_units = set(labels.units_for(question_id))
    metrics: dict[str, float] = {"mrr": reciprocal_rank(matches)}
    for k in ks:
        metrics[f"hit@{k}"] = hit_at_k(matches, k)
        metrics[f"precision@{k}"] = precision_at_k(matches, k)
        metrics[f"recall@{k}"] = recall_at_k(matches, relevant_units, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(matches, k, n_relevant_chunks)
    covered = set().union(*matches) if matches else set()
    return StageResult(
        ranked_chunk_ids=[sc.chunk.id for sc in ranking],
        metrics=metrics,
        missed_units=sorted(u.label() for u in relevant_units - covered),
    )


def _mean_by_stage(
    rows: Sequence[QuestionRetrieval],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for stage in STAGE_NAMES:
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for metric, value in row.stages[stage].metrics.items():
                values[metric].append(value)
        out[stage] = {m: sum(v) / len(v) for m, v in values.items()}
    return out


def evaluate_retrieval(
    questions: Sequence[GoldenQuestion],
    labels: RelevanceLabels,
    embedder: Embedder,
    vector_store: ChromaVectorStore,
    sparse_index: SparseIndex,
    reranker: Reranker,
    *,
    dense_top_k: int,
    sparse_top_k: int,
    rrf_k: int,
    rrf_dense_weight: float,
    rrf_sparse_weight: float,
    rerank_candidate_pool: int = 20,
    rerank_top_n: int = 5,
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
) -> RetrievalReport:
    all_chunks = vector_store.get_all_chunks()
    depth = {
        "dense": dense_top_k,
        "sparse": sparse_top_k,
        "fused": rerank_candidate_pool,
        "reranked": rerank_candidate_pool,
    }

    rows: list[QuestionRetrieval] = []
    skipped: list[str] = []
    unmatched: list[str] = []
    categories: dict[str, str] = {}

    for q in questions:
        units = labels.units_for(q.id)
        if not units:
            if q.expected_behavior == "answerable":
                skipped.append(q.id)
            continue

        for unit in units:
            if not any(unit.matches(c) for c in all_chunks):
                unmatched.append(f"{q.id}: {unit.label()}")
        n_relevant_chunks = sum(1 for c in all_chunks if labels.matches(q.id, c))

        stages = retrieve_stages(
            q.question,
            embedder,
            vector_store,
            sparse_index,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            rrf_k=rrf_k,
            rrf_dense_weight=rrf_dense_weight,
            rrf_sparse_weight=rrf_sparse_weight,
        )
        pool = stages.fused[:rerank_candidate_pool]
        reranked = reranker.rerank(q.question, pool, top_n=len(pool)) if pool else []
        rankings = {
            "dense": stages.dense,
            "sparse": stages.sparse,
            "fused": pool,
            "reranked": reranked,
        }

        stage_results: dict[str, StageResult] = {}
        for stage, ranking in rankings.items():
            ks = sorted(
                {k for k in k_values if k <= depth[stage]}
                | ({rerank_top_n} if stage in ("fused", "reranked") else set())
            )
            stage_results[stage] = _score_ranking(
                ranking, q.id, labels, n_relevant_chunks, ks
            )
        categories[q.id] = q.category
        rows.append(
            QuestionRetrieval(
                question_id=q.id,
                category=q.category,
                question=q.question,
                n_relevant_units=len(units),
                n_relevant_chunks=n_relevant_chunks,
                stages=stage_results,
            )
        )

    by_category: dict[str, list[QuestionRetrieval]] = defaultdict(list)
    for row in rows:
        by_category[row.category].append(row)

    return RetrievalReport(
        n_evaluated=len(rows),
        skipped_no_labels=skipped,
        unmatched_labels=unmatched,
        k_values=sorted(set(k_values)),
        final_context_k=rerank_top_n,
        summary=_mean_by_stage(rows) if rows else {},
        summary_by_category={c: _mean_by_stage(r) for c, r in by_category.items()},
        per_question=rows,
    )


def format_retrieval_report(report: RetrievalReport) -> str:
    if report.n_evaluated == 0:
        return "No labeled questions to evaluate (see `eval_cli.py label-draft`)."
    n = report.final_context_k
    cols = [f"hit@{n}", f"precision@{n}", f"recall@{n}", f"ndcg@{n}", "mrr"]
    lines = [
        f"Retrieval metrics over {report.n_evaluated} labeled questions "
        f"(final context = top {n})",
        f"{'stage':<10}" + "".join(f"{c:>13}" for c in cols),
    ]
    for stage in STAGE_NAMES:
        m = report.summary[stage]
        lines.append(
            f"{stage:<10}"
            + "".join(f"{m[c]:>13.2f}" if c in m else f"{'n/a':>13}" for c in cols)
        )
    pool_key = f"recall@{max(report.k_values)}"
    fused = report.summary["fused"]
    if pool_key in fused:
        lines.append(
            f"\nreranker ceiling ({pool_key} of fused pool): {fused[pool_key]:.2f}"
            f" | reranker lift @{n} (recall): "
            f"{report.summary['reranked'][f'recall@{n}'] - fused[f'recall@{n}']:+.2f}"
        )
    if report.skipped_no_labels:
        lines.append(f"unlabeled answerable questions: {report.skipped_no_labels}")
    if report.unmatched_labels:
        lines.append(
            f"LABELS MATCHING NO CHUNK (typo / page mismatch?): "
            f"{report.unmatched_labels}"
        )
    return "\n".join(lines)
