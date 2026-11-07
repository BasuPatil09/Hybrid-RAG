from __future__ import annotations

import numpy as np


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def find_duplicate_indices(
    new_embeddings: list[list[float]],
    existing_embeddings: list[list[float]],
    threshold: float,
) -> set[int]:

    if not new_embeddings:
        return set()

    new_arr = np.array(new_embeddings)
    new_vs_new = _cosine_similarity_matrix(new_arr, new_arr)
    np.fill_diagonal(new_vs_new, -1.0)

    new_vs_existing = None
    if existing_embeddings:
        new_vs_existing = _cosine_similarity_matrix(
            new_arr, np.array(existing_embeddings)
        )

    duplicate_indices: set[int] = set()
    kept: list[int] = []
    for i in range(len(new_embeddings)):
        if new_vs_existing is not None and new_vs_existing[i].max() > threshold:
            duplicate_indices.add(i)
            continue
        if any(new_vs_new[i, k] > threshold for k in kept):
            duplicate_indices.add(i)
            continue
        kept.append(i)
    return duplicate_indices
