from __future__ import annotations

from pydantic import BaseModel

from rag.eval.golden_qa import GoldenQuestion
from rag.generation.confidence import CompletenessJudge
from rag.generation.generator import Generator
from rag.generation.verification import CitationVerifier
from rag.indexing.embeddings import Embedder
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.pipeline import answer_question
from rag.retrieval.reranker import Reranker


class EvalResult(BaseModel):
    question_id: str
    category: str
    question: str
    expected_behavior: str
    actual_is_insufficient: bool
    behavior_matched: bool
    answer: str
    citation_accuracy: float
    retrieval_confidence: float
    composite_confidence: float

class EvalSummary(BaseModel):
    total_questions: int
    behavior_match_rate: float
    behavior_match_rate_by_category: dict[str, float]
    mean_citation_accuracy: float
    mean_retrieval_confidence: float
    mean_composite_confidence: float

def _behavior_matches(expected: str, is_insufficient: bool) -> bool:

    if expected == "answerable":
        return not is_insufficient
    if expected == "insufficient_context":
        return is_insufficient
    return True

def run_eval_suite(
    questions: list[GoldenQuestion],
    embedder: Embedder,
    vector_store: ChromaVectorStore,
    sparse_index: SparseIndex,
    reranker: Reranker,
    generator: Generator,
    verifier: CitationVerifier,
    completeness_judge: CompletenessJudge,
    *,
    dense_top_k: int,
    sparse_top_k: int,
    rrf_k: int,
    rrf_dense_weight: float,
    rrf_sparse_weight: float,
    rerank_top_n: int,
) -> list[EvalResult]:
    results = []
    for q in questions:
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
        results.append(
            EvalResult(
                question_id=q.id,
                category=q.category,
                question=q.question,
                expected_behavior=q.expected_behavior,
                actual_is_insufficient=answered.answer.is_insufficient,
                behavior_matched=_behavior_matches(
                    q.expected_behavior, answered.answer.is_insufficient
                ),
                answer=answered.answer.answer,
                citation_accuracy=answered.verified.citation_accuracy,
                retrieval_confidence=answered.confidence.retrieval_confidence,
                composite_confidence=answered.confidence.composite,
            )
        )
    return results

def summarize(results: list[EvalResult]) -> EvalSummary:
    if not results:
        return EvalSummary(
            total_questions=0,
            behavior_match_rate=0.0,
            behavior_match_rate_by_category={},
            mean_citation_accuracy=0.0,
            mean_retrieval_confidence=0.0,
            mean_composite_confidence=0.0,
        )
    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    answered_results = [r for r in results if not r.actual_is_insufficient]

    return EvalSummary(
        total_questions=len(results),
        behavior_match_rate=sum(1 for r in results if r.behavior_matched)
        / len(results),
        behavior_match_rate_by_category={
            cat: sum(1 for r in rs if r.behavior_matched) / len(rs)
            for cat, rs in by_category.items()
        },
        mean_citation_accuracy=(
            sum(r.citation_accuracy for r in answered_results) / len(answered_results)
            if answered_results
            else 0.0
        ),
        mean_retrieval_confidence=sum(r.retrieval_confidence for r in results)
        / len(results),
        mean_composite_confidence=sum(r.composite_confidence for r in results)
        / len(results),
    )
