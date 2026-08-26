from __future__ import annotations

import chromadb

from rag.eval.golden_qa import GoldenQuestion
from rag.eval.runner import EvalResult, _behavior_matches, run_eval_suite, summarize
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.retrieval.reranker import Reranker
from rag.schemas import Chunk, ChunkMetadata, ScoredChunk
from tests.fakes import FakeEmbedder


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        content="some source content",
        metadata=ChunkMetadata(source="f.md", format="md", chunk_index=0, char_count=5),
    )


class _FakeReranker(Reranker):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        return candidates[:top_n] or [ScoredChunk(chunk=_chunk("fallback"), score=1.0)]


class _FakeGenerator:
    def generate(self, query: str, chunks):
        from rag.generation.generator import GeneratedAnswer

        if "unanswerable" in query.lower():
            return GeneratedAnswer(
                answer="INSUFFICIENT_CONTEXT: not covered.",
                is_insufficient=True,
                cited_chunk_ids=[],
            )
        return GeneratedAnswer(
            answer="A real answer [1].",
            is_insufficient=False,
            cited_chunk_ids=[chunks[0].id] if chunks else [],
        )


class _FakeVerifier:
    def verify_claim(self, claim_text: str, source_text: str) -> tuple[bool, str]:
        return True, "fake supported"


class _FakeCompletenessJudge:
    def judge(self, query: str, answer_text: str) -> tuple[float, str]:
        return 0.9, "fake mostly complete"


def _new_components(collection_name: str):
    import uuid

    return {
        "embedder": FakeEmbedder(),
        "vector_store": ChromaVectorStore(
            chromadb.EphemeralClient(), f"{collection_name}_{uuid.uuid4().hex[:8]}"
        ),
        "sparse_index": SparseIndex(),
        "reranker": _FakeReranker(),
        "generator": _FakeGenerator(),
        "verifier": _FakeVerifier(),
        "completeness_judge": _FakeCompletenessJudge(),
    }


def _seed_store(
    vector_store: ChromaVectorStore, embedder: FakeEmbedder, sparse_index: SparseIndex
) -> None:
    chunks = [_chunk("a")]
    vector_store.upsert(chunks, embedder.embed([c.content for c in chunks]))
    sparse_index.rebuild(chunks)


class TestBehaviorMatches:
    def test_answerable_expects_a_real_answer(self) -> None:
        assert _behavior_matches("answerable", is_insufficient=False) is True
        assert _behavior_matches("answerable", is_insufficient=True) is False

    def test_insufficient_context_expects_a_decline(self) -> None:
        assert _behavior_matches("insufficient_context", is_insufficient=True) is True
        assert _behavior_matches("insufficient_context", is_insufficient=False) is False

    def test_ambiguous_always_matches(self) -> None:
        assert _behavior_matches("ambiguous", is_insufficient=True) is True
        assert _behavior_matches("ambiguous", is_insufficient=False) is True


class TestRunEvalSuite:
    def test_runs_every_question_and_produces_a_result(self) -> None:
        components = _new_components("eval_run_test")
        _seed_store(
            components["vector_store"],
            components["embedder"],
            components["sparse_index"],
        )
        questions = [
            GoldenQuestion(
                id="q1",
                category="direct_lookup",
                question="A real question?",
                expected_behavior="answerable",
            ),
            GoldenQuestion(
                id="q2",
                category="unanswerable",
                question="An unanswerable question?",
                expected_behavior="insufficient_context",
            ),
        ]

        results = run_eval_suite(
            questions,
            **components,
            dense_top_k=10,
            sparse_top_k=10,
            rrf_k=60,
            rrf_dense_weight=0.7,
            rrf_sparse_weight=0.3,
            rerank_top_n=5,
        )

        assert len(results) == 2
        assert results[0].behavior_matched is True
        assert results[1].behavior_matched is True

    def test_result_carries_question_metadata(self) -> None:
        components = _new_components("eval_metadata_test")
        _seed_store(
            components["vector_store"],
            components["embedder"],
            components["sparse_index"],
        )
        questions = [
            GoldenQuestion(
                id="q1",
                category="multi_hop",
                question="A question?",
                expected_behavior="answerable",
            )
        ]

        results = run_eval_suite(
            questions,
            **components,
            dense_top_k=10,
            sparse_top_k=10,
            rrf_k=60,
            rrf_dense_weight=0.7,
            rrf_sparse_weight=0.3,
            rerank_top_n=5,
        )

        assert results[0].question_id == "q1"
        assert results[0].category == "multi_hop"


def _result(
    category: str,
    behavior_matched: bool,
    is_insufficient: bool,
    citation_accuracy: float = 1.0,
) -> EvalResult:
    return EvalResult(
        question_id="q",
        category=category,
        question="q",
        expected_behavior="answerable",
        actual_is_insufficient=is_insufficient,
        behavior_matched=behavior_matched,
        answer="a",
        citation_accuracy=citation_accuracy,
        retrieval_confidence=0.8,
        composite_confidence=0.8,
    )


class TestSummarize:
    def test_empty_results_returns_zeroed_summary(self) -> None:
        summary = summarize([])
        assert summary.total_questions == 0
        assert summary.behavior_match_rate == 0.0

    def test_behavior_match_rate_across_results(self) -> None:
        results = [
            _result("direct_lookup", True, False),
            _result("direct_lookup", False, False),
        ]
        assert summarize(results).behavior_match_rate == 0.5

    def test_behavior_match_rate_by_category(self) -> None:
        results = [
            _result("direct_lookup", True, False),
            _result("direct_lookup", True, False),
            _result("unanswerable", False, False),
        ]
        summary = summarize(results)
        assert summary.behavior_match_rate_by_category["direct_lookup"] == 1.0
        assert summary.behavior_match_rate_by_category["unanswerable"] == 0.0

    def test_citation_accuracy_excludes_insufficient_context_results(self) -> None:

        results = [
            _result(
                "direct_lookup", True, is_insufficient=False, citation_accuracy=0.5
            ),
            _result("unanswerable", True, is_insufficient=True, citation_accuracy=1.0),
        ]
        summary = summarize(results)
        assert summary.mean_citation_accuracy == 0.5

    def test_mean_citation_accuracy_zero_when_all_insufficient(self) -> None:
        results = [
            _result("unanswerable", True, is_insufficient=True, citation_accuracy=1.0)
        ]
        assert summarize(results).mean_citation_accuracy == 0.0
