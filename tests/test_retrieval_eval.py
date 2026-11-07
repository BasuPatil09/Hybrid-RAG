from __future__ import annotations

import uuid

import chromadb

from rag.eval.golden_qa import GoldenQuestion
from rag.eval.relevance import RelevanceLabels, RelevanceUnit
from rag.eval.retrieval_eval import (
    STAGE_NAMES,
    evaluate_retrieval,
    format_retrieval_report,
)
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.retrieval.reranker import Reranker
from rag.schemas import Chunk, ChunkMetadata, ScoredChunk
from tests.fakes import FakeEmbedder

_TEXT = {
    ("guide.pdf", 1): "gradient descent updates parameters along the negative gradient",
    ("guide.pdf", 2): "the learning rate controls the step size of every update",
    ("other.pdf", 1): "baking bread requires flour water yeast and patience",
    ("other.pdf", 2): "the stock market closed higher on strong quarterly earnings",
    ("other.pdf", 3): "migratory birds travel thousands of miles each winter",
    ("other.pdf", 4): "simmer the tomato sauce for twenty minutes before serving",
}


class _OverlapReranker(Reranker):
    def rerank(self, query, candidates, top_n):
        q = set(query.lower().split())
        scored = [
            ScoredChunk(
                chunk=c.chunk, score=float(len(q & set(c.chunk.content.split())))
            )
            for c in candidates
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_n]


def _components():
    embedder = FakeEmbedder()
    store = ChromaVectorStore(chromadb.EphemeralClient(), f"re_{uuid.uuid4().hex[:10]}")
    chunks = [
        Chunk(
            id=f"{src}::p{page}::c0",
            content=text,
            metadata=ChunkMetadata(
                source=src, format="pdf", page=page, chunk_index=0, char_count=len(text)
            ),
        )
        for (src, page), text in _TEXT.items()
    ]
    store.upsert(chunks, embedder.embed([c.content for c in chunks]))
    sparse = SparseIndex()
    sparse.rebuild(chunks)
    return embedder, store, sparse


def _q(qid: str, text: str, behavior: str = "answerable", category="direct_lookup"):
    return GoldenQuestion(
        id=qid, category=category, question=text, expected_behavior=behavior
    )


def _run(questions, labels):
    embedder, store, sparse = _components()
    return evaluate_retrieval(
        questions,
        labels,
        embedder,
        store,
        sparse,
        _OverlapReranker(),
        dense_top_k=10,
        sparse_top_k=10,
        rrf_k=60,
        rrf_dense_weight=0.7,
        rrf_sparse_weight=0.3,
        rerank_top_n=3,
        k_values=(1, 3, 5, 10),
    )


def _labels(*units: tuple[str, str, int]) -> RelevanceLabels:
    return RelevanceLabels(
        RelevanceUnit(question_id=q, source=s, page=p) for q, s, p in units
    )


class TestEvaluateRetrieval:
    def test_exact_text_query_is_perfect_at_every_stage(self) -> None:
        q = _q("q1", _TEXT[("guide.pdf", 1)])
        report = _run([q], _labels(("q1", "guide.pdf", 1)))

        assert report.n_evaluated == 1
        for stage in STAGE_NAMES:
            m = report.summary[stage]
            assert m["hit@1"] == 1.0 and m["precision@1"] == 1.0, stage
            assert m["recall@1"] == 1.0 and m["mrr"] == 1.0, stage

    def test_multi_unit_question_reports_partial_recall_at_small_k(self) -> None:
        q = _q("q2", _TEXT[("guide.pdf", 2)], category="multi_hop")
        report = _run([q], _labels(("q2", "guide.pdf", 2), ("q2", "other.pdf", 3)))

        dense = report.per_question[0].stages["dense"]
        assert dense.metrics["recall@1"] == 0.5
        assert dense.metrics["recall@10"] == 1.0
        assert dense.missed_units == []

    def test_unretrieved_relevant_unit_is_listed_as_missed(self) -> None:
        q = _q("q3", _TEXT[("guide.pdf", 1)])
        embedder, store, sparse = _components()
        report = evaluate_retrieval(
            [q],
            _labels(("q3", "guide.pdf", 1), ("q3", "other.pdf", 4)),
            embedder,
            store,
            sparse,
            _OverlapReranker(),
            dense_top_k=1,
            sparse_top_k=1,
            rrf_k=60,
            rrf_dense_weight=0.7,
            rrf_sparse_weight=0.3,
            rerank_top_n=1,
            k_values=(1,),
        )
        assert report.per_question[0].stages["dense"].missed_units == ["other.pdf p4"]

    def test_k_values_beyond_a_stage_depth_are_not_reported_for_it(self) -> None:
        q = _q("q1", _TEXT[("guide.pdf", 1)])
        embedder, store, sparse = _components()
        report = evaluate_retrieval(
            [q],
            _labels(("q1", "guide.pdf", 1)),
            embedder,
            store,
            sparse,
            _OverlapReranker(),
            dense_top_k=3,
            sparse_top_k=3,
            rrf_k=60,
            rrf_dense_weight=0.7,
            rrf_sparse_weight=0.3,
            rerank_top_n=3,
            k_values=(1, 3, 10),
        )
        assert "recall@10" not in report.summary["dense"]
        assert "recall@10" in report.summary["fused"]

    def test_unlabeled_answerable_question_is_skipped_and_named(self) -> None:
        questions = [
            _q("q1", _TEXT[("guide.pdf", 1)]),
            _q("q9", "something unlabeled"),
            _q("q8", "off topic", behavior="insufficient_context"),
        ]
        report = _run(questions, _labels(("q1", "guide.pdf", 1)))
        assert report.n_evaluated == 1
        assert report.skipped_no_labels == ["q9"]

    def test_label_matching_no_chunk_is_flagged_for_review(self) -> None:
        q = _q("q1", _TEXT[("guide.pdf", 1)])
        report = _run([q], _labels(("q1", "guide.pdf", 1), ("q1", "typo.pdf", 7)))
        assert report.unmatched_labels == ["q1: typo.pdf p7"]

    def test_counts_relevant_chunks_in_the_whole_index(self) -> None:
        q = _q("q1", _TEXT[("guide.pdf", 1)])
        report = _run([q], _labels(("q1", "other.pdf", 1)))
        assert report.per_question[0].n_relevant_chunks == 1

    def test_summary_is_broken_down_by_category(self) -> None:
        questions = [
            _q("q1", _TEXT[("guide.pdf", 1)], category="direct_lookup"),
            _q("q2", _TEXT[("guide.pdf", 2)], category="multi_hop"),
        ]
        report = _run(
            questions, _labels(("q1", "guide.pdf", 1), ("q2", "guide.pdf", 2))
        )
        assert set(report.summary_by_category) == {"direct_lookup", "multi_hop"}

    def test_no_labels_yields_empty_report_and_helpful_message(self) -> None:
        report = _run([_q("q1", "x")], RelevanceLabels([]))
        assert report.n_evaluated == 0 and report.summary == {}
        assert "label-draft" in format_retrieval_report(report)

    def test_formatted_table_names_every_stage(self) -> None:
        q = _q("q1", _TEXT[("guide.pdf", 1)])
        text = format_retrieval_report(_run([q], _labels(("q1", "guide.pdf", 1))))
        for stage in STAGE_NAMES:
            assert stage in text
        assert "reranker ceiling" in text
