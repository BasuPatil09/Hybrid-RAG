from __future__ import annotations

from rag.indexing.sparse_index import SparseIndex
from rag.schemas import ScoredChunk
from rag.tracing import query_only_inputs, scored_chunks_outputs, traced


@traced(
    "sparse_retrieve",
    run_type="retriever",
    process_inputs=query_only_inputs,
    process_outputs=scored_chunks_outputs,
)
def sparse_retrieve(
    query: str, sparse_index: SparseIndex, top_k: int
) -> list[ScoredChunk]:

    return sparse_index.query(query, top_k=top_k)
