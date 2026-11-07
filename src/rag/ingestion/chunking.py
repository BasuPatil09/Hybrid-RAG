from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.schemas import Chunk, ChunkMetadata, Document


def make_chunk_id(source: str, page: int | None, chunk_index: int) -> str:
    return f"{source}::p{page if page is not None else 0}::c{chunk_index}"


def _build_chunk(doc: Document, content: str, section: str | None, index: int) -> Chunk:
    metadata = ChunkMetadata(
        **doc.metadata.model_dump(),
        section=section,
        chunk_index=index,
        char_count=len(content),
    )
    return Chunk(
        id=make_chunk_id(doc.metadata.source, doc.metadata.page, index),
        content=content,
        metadata=metadata,
    )


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        pass


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for index, piece in enumerate(self._splitter.split_text(doc.content)):
                chunks.append(_build_chunk(doc, piece, section=None, index=index))
        return chunks


class StructureAwareChunker(BaseChunker):
    _HEADER_AWARE_FORMATS = frozenset({"md", "html"})

    def __init__(
        self, chunk_size: int, chunk_overlap: int, header_levels: tuple[str, ...]
    ) -> None:
        headers_to_split_on = [
            (level, f"Header {i + 1}") for i, level in enumerate(header_levels)
        ]
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=True
        )
        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            if doc.metadata.format in self._HEADER_AWARE_FORMATS:
                sections = [
                    (
                        s.page_content,
                        " > ".join(s.metadata.values()) if s.metadata else None,
                    )
                    for s in self._header_splitter.split_text(doc.content)
                ]
            else:
                sections = [(doc.content, None)]

            index = 0
            for section_text, section_path in sections:
                for piece in self._size_splitter.split_text(section_text):
                    chunks.append(
                        _build_chunk(doc, piece, section=section_path, index=index)
                    )
                    index += 1
        return chunks


def get_chunker(
    strategy: Literal["structure_aware", "fixed_size"],
    *,
    chunk_size: int,
    chunk_overlap: int,
    header_levels: tuple[str, ...] = ("#", "##", "###"),
) -> BaseChunker:

    if strategy == "structure_aware":
        return StructureAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            header_levels=header_levels,
        )
    if strategy == "fixed_size":
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
