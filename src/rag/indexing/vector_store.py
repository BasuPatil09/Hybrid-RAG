from __future__ import annotations
from typing import Any
import chromadb
from rag.schemas import Chunk, ChunkMetadata, ScoredChunk

def _metadata_to_chroma(metadata: ChunkMetadata) -> dict[str, Any]:
    return {k: v for k, v in metadata.model_dump().items() if v is not None}

class ChromaVectorStore:
    def __init__(self, client: chromadb.ClientAPI, collection_name: str) -> None:
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        if not chunks:
            return

        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            self._collection.upsert(
                ids=[c.id for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c.content for c in batch_chunks],
                metadatas=[_metadata_to_chroma(c.metadata) for c in batch_chunks],
            )

    def count(self) -> int:
        return self._collection.count()

    def get_all_ids(self) -> set[str]:

        if self.count() == 0:
            return set()
        result = self._collection.get(include=[])
        return set(result["ids"])

    def get_all_embeddings(self) -> list[list[float]]:

        if self.count() == 0:
            return []
        result = self._collection.get(include=["embeddings"])
        return result["embeddings"].tolist()

    def get_all_chunks(self) -> list[Chunk]:
        if self.count() == 0:
            return []
        result = self._collection.get(include=["metadatas", "documents"])
        return [
            Chunk(id=cid, content=doc, metadata=ChunkMetadata(**meta))
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def query(self, query_embedding: list[float], top_k: int) -> list[ScoredChunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["metadatas", "documents", "distances"],
        )
        scored: list[ScoredChunk] = []
        for cid, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = Chunk(id=cid, content=document, metadata=ChunkMetadata(**metadata))
            scored.append(ScoredChunk(chunk=chunk, score=1.0 - distance))
        return scored
