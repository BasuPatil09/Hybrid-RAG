from __future__ import annotations

from rag.indexing.embeddings import Embedder
from rag.indexing.vector_store import ChromaVectorStore
from rag.schemas import ScoredChunk

def dense_retrieve(
    query: str, embedder: Embedder, vector_store: ChromaVectorStore, top_k: int
) -> list[ScoredChunk]:

    query_embedding = embedder.embed([query])[0]
    return vector_store.query(query_embedding, top_k=top_k)
