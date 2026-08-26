from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

class DocumentMetadata(BaseModel):
    source: str
    format: Literal["md", "txt", "html", "pdf"]
    page: int | None = None

class Document(BaseModel):
    content: str
    metadata: DocumentMetadata

class ChunkMetadata(DocumentMetadata):
    section: str | None = None
    chunk_index: int
    char_count: int

class Chunk(BaseModel):
    id: str
    content: str
    metadata: ChunkMetadata

class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
