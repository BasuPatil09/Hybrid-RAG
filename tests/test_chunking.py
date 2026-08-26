from __future__ import annotations

import pytest

from rag.ingestion.chunking import (
    FixedSizeChunker,
    StructureAwareChunker,
    get_chunker,
    make_chunk_id,
)
from rag.schemas import Document, DocumentMetadata


def _doc(
    content: str, *, source: str = "f.txt", fmt: str = "txt", page: int | None = None
) -> Document:
    return Document(
        content=content, metadata=DocumentMetadata(source=source, format=fmt, page=page)
    )


class TestMakeChunkId:
    def test_deterministic_and_readable(self) -> None:
        assert make_chunk_id("docs/a.md", None, 2) == "docs/a.md::p0::c2"
        assert make_chunk_id("report.pdf", 5, 0) == "report.pdf::p5::c0"

    def test_same_inputs_produce_same_id(self) -> None:
        assert make_chunk_id("a.md", 1, 3) == make_chunk_id("a.md", 1, 3)


class TestFixedSizeChunker:
    def test_splits_long_text_into_bounded_pieces(self) -> None:
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        long_text = "Sentence providing filler content for the test case. " * 10
        chunks = chunker.chunk([_doc(long_text)])

        assert len(chunks) > 1
        assert all(c.metadata.char_count <= 55 for c in chunks)
        assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.metadata.section is None for c in chunks)

    def test_char_count_matches_content_length(self) -> None:
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk([_doc("short text under the chunk size")])

        assert chunks[0].metadata.char_count == len(chunks[0].content)

    def test_empty_document_produces_no_chunks(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk([_doc("")]) == []

    def test_chunk_id_incorporates_page(self) -> None:
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(
            [_doc("page level text", source="doc.pdf", fmt="pdf", page=3)]
        )
        assert chunks[0].id == "doc.pdf::p3::c0"


class TestStructureAwareChunker:
    def _chunker(
        self, chunk_size: int = 200, chunk_overlap: int = 20
    ) -> StructureAwareChunker:
        return StructureAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            header_levels=("#", "##", "###"),
        )

    def test_tags_chunks_with_header_path(self) -> None:
        md = "# Intro\n\nWelcome.\n\n## Setup\n\nInstall steps go here.\n\n### Install\n\nRun pip install."
        chunks = self._chunker().chunk([_doc(md, source="guide.md", fmt="md")])

        sections = [c.metadata.section for c in chunks]
        assert "Intro" in sections
        assert "Intro > Setup" in sections
        assert "Intro > Setup > Install" in sections

    def test_resplits_oversized_section_into_multiple_chunks_same_section(self) -> None:
        body = "Filler sentence for a single long section. " * 15
        md = f"# Only Header\n\n{body}"
        chunks = self._chunker(chunk_size=100, chunk_overlap=15).chunk(
            [_doc(md, source="long.md", fmt="md")]
        )

        assert len(chunks) > 1
        assert all(c.metadata.section == "Only Header" for c in chunks)
        assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_headerless_content_falls_back_with_section_none(self) -> None:

        plain = "Just a plain paragraph with no headers at all, repeated. " * 5
        chunks = self._chunker().chunk(
            [_doc(plain, source="report.pdf", fmt="pdf", page=2)]
        )

        assert len(chunks) >= 1
        assert all(c.metadata.section is None for c in chunks)
        assert all(c.metadata.page == 2 for c in chunks)

    def test_pdf_code_comments_are_not_misdetected_as_headers(self) -> None:

        pdf_text = (
            "# =====================\n"
            "def train_step(model, batch):\n"
            "    # Add current thread to the list\n"
            "    optimizer.zero_grad()\n"
        )
        chunks = self._chunker().chunk(
            [_doc(pdf_text, source="slides.pdf", fmt="pdf", page=7)]
        )

        assert all(c.metadata.section is None for c in chunks)
        assert any("Add current thread to the list" in c.content for c in chunks)

    def test_html_headers_still_detected_after_pdf_exclusion(self) -> None:

        html_like_md = "# Real Heading\n\nBody text under the heading."
        chunks = self._chunker().chunk(
            [_doc(html_like_md, source="page.html", fmt="html")]
        )

        assert chunks[0].metadata.section == "Real Heading"

    def test_empty_document_produces_no_chunks(self) -> None:
        assert self._chunker().chunk([_doc("")]) == []

    def test_multiple_documents_reset_chunk_index_per_document(self) -> None:
        docs = [
            _doc("# A\n\nfirst doc body", source="a.md"),
            _doc("# B\n\nsecond doc body", source="b.md"),
        ]
        chunks = self._chunker().chunk(docs)

        a_indices = [
            c.metadata.chunk_index for c in chunks if c.metadata.source == "a.md"
        ]
        b_indices = [
            c.metadata.chunk_index for c in chunks if c.metadata.source == "b.md"
        ]
        assert a_indices[0] == 0
        assert b_indices[0] == 0


class TestGetChunker:
    def test_returns_structure_aware(self) -> None:
        chunker = get_chunker("structure_aware", chunk_size=800, chunk_overlap=120)
        assert isinstance(chunker, StructureAwareChunker)

    def test_returns_fixed_size(self) -> None:
        chunker = get_chunker("fixed_size", chunk_size=800, chunk_overlap=120)
        assert isinstance(chunker, FixedSizeChunker)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            get_chunker("semantic", chunk_size=800, chunk_overlap=120)
