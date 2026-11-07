from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Sequence

Matches = Sequence[frozenset[Hashable]]


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def hit_at_k(matches: Matches, k: int) -> float:
    _check_k(k)
    return 1.0 if any(matches[:k]) else 0.0


def precision_at_k(matches: Matches, k: int) -> float:
    _check_k(k)
    return sum(1 for m in matches[:k] if m) / k


def recall_at_k(
    matches: Matches, relevant_units: Collection[Hashable], k: int
) -> float:
    _check_k(k)
    if not relevant_units:
        return 0.0
    covered: set[Hashable] = set()
    for m in matches[:k]:
        covered |= m
    return len(covered & set(relevant_units)) / len(set(relevant_units))


def reciprocal_rank(matches: Matches) -> float:
    for rank, m in enumerate(matches, start=1):
        if m:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(matches: Matches, k: int, n_relevant_total: int) -> float:
    _check_k(k)
    ideal_hits = min(k, n_relevant_total)
    if ideal_hits == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, m in enumerate(matches[:k], 1) if m)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
