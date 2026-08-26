from __future__ import annotations

from rag.indexing.dedup import find_duplicate_indices


class TestFindDuplicateIndices:
    def test_identical_vectors_within_batch_are_flagged(self) -> None:
        vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        dupes = find_duplicate_indices(vecs, existing_embeddings=[], threshold=0.95)
        assert dupes == {1}

    def test_orthogonal_vectors_not_flagged(self) -> None:
        vecs = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        assert (
            find_duplicate_indices(vecs, existing_embeddings=[], threshold=0.95)
            == set()
        )

    def test_near_but_not_over_threshold_not_flagged(self) -> None:

        vecs = [[1.0, 0.0], [0.7, 0.7]]
        assert (
            find_duplicate_indices(vecs, existing_embeddings=[], threshold=0.95)
            == set()
        )

    def test_flags_against_existing_embeddings(self) -> None:
        new = [[1.0, 0.0]]
        existing = [[1.0, 0.0001]]
        assert find_duplicate_indices(new, existing, threshold=0.95) == {0}

    def test_existing_embeddings_never_flagged_themselves(self) -> None:

        new = [[1.0, 0.0], [0.0, 1.0]]
        existing = [[1.0, 0.0]]
        dupes = find_duplicate_indices(new, existing, threshold=0.95)
        assert dupes == {0}
        assert all(0 <= i < len(new) for i in dupes)

    def test_empty_new_embeddings_returns_empty_set(self) -> None:
        assert (
            find_duplicate_indices([], existing_embeddings=[[1.0, 0.0]], threshold=0.95)
            == set()
        )

    def test_empty_existing_embeddings_handled(self) -> None:
        assert (
            find_duplicate_indices([[1.0, 0.0]], existing_embeddings=[], threshold=0.95)
            == set()
        )

    def test_threshold_is_configurable(self) -> None:
        vecs = [[1.0, 0.0], [0.9, 0.1]]
        assert find_duplicate_indices(vecs, [], threshold=0.999) == set()
        assert find_duplicate_indices(vecs, [], threshold=0.9) == {1}
