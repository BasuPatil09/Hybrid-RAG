from __future__ import annotations

import math

import pytest

from rag.eval.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

NONE = frozenset()
A, B, C = "A", "B", "C"
RANKED = [NONE, frozenset({A}), NONE, frozenset({B}), frozenset({A})]
RELEVANT = {A, B, C}


class TestHit:
    def test_zero_until_first_relevant_rank(self) -> None:
        assert hit_at_k(RANKED, 1) == 0.0
        assert hit_at_k(RANKED, 2) == 1.0


class TestPrecision:
    def test_counts_relevant_chunks_over_k(self) -> None:
        assert precision_at_k(RANKED, 3) == pytest.approx(1 / 3)
        assert precision_at_k(RANKED, 5) == pytest.approx(3 / 5)

    def test_short_result_list_is_penalized_not_rewarded(self) -> None:
        assert precision_at_k([frozenset({A})], 5) == pytest.approx(0.2)


class TestRecall:
    def test_counts_distinct_relevant_units_covered(self) -> None:
        assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)
        assert recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(2 / 3)

    def test_repeat_hits_on_the_same_unit_do_not_inflate_recall(self) -> None:
        assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)

    def test_unit_never_retrieved_caps_recall_below_one(self) -> None:
        assert recall_at_k(RANKED, RELEVANT, 100) < 1.0

    def test_no_relevant_units_is_zero_not_a_crash(self) -> None:
        assert recall_at_k(RANKED, set(), 5) == 0.0


class TestMRR:
    def test_reciprocal_of_first_relevant_rank(self) -> None:
        assert reciprocal_rank(RANKED) == pytest.approx(0.5)

    def test_zero_when_nothing_relevant(self) -> None:
        assert reciprocal_rank([NONE, NONE]) == 0.0


class TestNDCG:
    def test_matches_hand_computed_value(self) -> None:
        dcg = 1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(6)
        idcg = sum(1 / math.log2(r + 1) for r in range(1, 5))
        assert ndcg_at_k(RANKED, 5, n_relevant_total=4) == pytest.approx(dcg / idcg)

    def test_perfect_ranking_scores_one(self) -> None:
        perfect = [frozenset({A}), frozenset({B}), NONE]
        assert ndcg_at_k(perfect, 3, n_relevant_total=2) == pytest.approx(1.0)

    def test_ideal_is_capped_by_k(self) -> None:
        assert ndcg_at_k([frozenset({A})], 1, n_relevant_total=50) == pytest.approx(1.0)

    def test_no_relevant_chunks_in_index_is_zero(self) -> None:
        assert ndcg_at_k(RANKED, 5, n_relevant_total=0) == 0.0


@pytest.mark.parametrize("fn", [hit_at_k, precision_at_k])
def test_k_below_one_is_rejected(fn) -> None:
    with pytest.raises(ValueError):
        fn(RANKED, 0)
