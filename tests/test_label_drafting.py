from __future__ import annotations

from rag.eval.golden_qa import GoldenQuestion
from rag.eval.label_drafting import (
    draft_labels_for_question,
    parse_relevant,
    pool_candidates,
)
from rag.pipeline import RetrievalStages
from rag.schemas import ScoredChunk
from tests.pipeline_fakes import make_chunk


def _sc(source: str, page: int, text: str = "t", score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(source, page, text), score=score)


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.reply


class TestParseRelevant:
    def test_comma_separated_numbers(self) -> None:
        assert parse_relevant("RELEVANT: 1, 4,7", 8) == {1, 4, 7}

    def test_none_means_empty(self) -> None:
        assert parse_relevant("RELEVANT: NONE", 5) == set()

    def test_out_of_range_numbers_are_dropped(self) -> None:
        assert parse_relevant("RELEVANT: 0, 2, 99", 3) == {2}

    def test_unparseable_reply_fails_closed(self) -> None:
        assert parse_relevant("I think the first one is good.", 3) == set()

    def test_tolerates_markdown_bold(self) -> None:
        assert parse_relevant("**RELEVANT:** 2", 3) == {2}


class TestPoolCandidates:
    def test_fused_first_then_dense_and_sparse_extras_deduplicated(self) -> None:
        a, b, c, d = (_sc("f.pdf", i) for i in (1, 2, 3, 4))
        stages = RetrievalStages(dense=[b, d], sparse=[c, a], fused=[a, b])
        pooled = pool_candidates(stages, pool_size=20, max_candidates=10)
        assert [p.chunk.id for p in pooled] == [x.chunk.id for x in (a, b, d, c)]

    def test_capped_at_max_candidates(self) -> None:
        many = [_sc("f.pdf", i) for i in range(1, 9)]
        stages = RetrievalStages(dense=many, sparse=[], fused=many)
        assert len(pool_candidates(stages, pool_size=20, max_candidates=5)) == 5


class TestDraftLabels:
    def _q(self) -> GoldenQuestion:
        return GoldenQuestion(
            id="q1",
            category="direct_lookup",
            question="What is X?",
            expected_behavior="answerable",
        )

    def test_rows_group_by_page_and_flagged_pages_come_first(self) -> None:
        cands = [
            _sc("a.pdf", 1, "intro"),
            _sc("a.pdf", 2, "X is a thing"),
            _sc("b.pdf", 7, "noise"),
        ]
        rows = draft_labels_for_question(self._q(), cands, _FakeClient("RELEVANT: 2"))
        assert [(r["source"], r["page"], r["llm_relevant"]) for r in rows] == [
            ("a.pdf", 2, "yes"),
            ("a.pdf", 1, "no"),
            ("b.pdf", 7, "no"),
        ]
        assert rows[0]["snippet"] == "X is a thing"

    def test_page_is_flagged_if_any_of_its_chunks_is(self) -> None:
        cands = [_sc("a.pdf", 2, "first chunk"), _sc("a.pdf", 2, "second chunk")]
        rows = draft_labels_for_question(self._q(), cands, _FakeClient("RELEVANT: 2"))
        assert len(rows) == 1 and rows[0]["llm_relevant"] == "yes"
        assert rows[0]["snippet"] == "second chunk"

    def test_one_listwise_call_per_question_with_numbered_excerpts(self) -> None:
        client = _FakeClient("RELEVANT: NONE")
        draft_labels_for_question(
            self._q(), [_sc("a.pdf", 1, "alpha"), _sc("a.pdf", 2, "beta")], client
        )
        assert len(client.calls) == 1
        assert "[1] alpha" in client.calls[0][1] and "[2] beta" in client.calls[0][1]
        assert "QUESTION: What is X?" in client.calls[0][1]

    def test_human_column_is_never_prefilled(self) -> None:
        rows = draft_labels_for_question(
            self._q(), [_sc("a.pdf", 1)], _FakeClient("RELEVANT: 1")
        )
        assert "human_relevant" not in rows[0]

    def test_no_candidates_makes_no_llm_call(self) -> None:
        client = _FakeClient("RELEVANT: 1")
        assert draft_labels_for_question(self._q(), [], client) == []
        assert client.calls == []
