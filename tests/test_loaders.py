from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from rag.ingestion.loaders import (
    DocumentLoaderRegistry,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    UnsupportedFormatError,
)


def _make_pdf(path: Path, pages_text: list[str]) -> None:
    c = canvas.Canvas(str(path))
    for text in pages_text:
        c.drawString(72, 700, text)
        c.showPage()
    c.save()


class TestTextLoader:
    def test_reads_content_and_tags_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("plain content here")

        docs = TextLoader().load(f)

        assert len(docs) == 1
        assert docs[0].content == "plain content here"
        assert docs[0].metadata.format == "txt"
        assert docs[0].metadata.page is None


class TestMarkdownLoader:
    def test_passes_through_unmodified(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nSome body text.")

        docs = MarkdownLoader().load(f)

        assert docs[0].content == "# Title\n\nSome body text."
        assert docs[0].metadata.format == "md"


class TestHTMLLoader:
    def test_converts_headers_and_paragraphs(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text(
            "<html><body>"
            "<h1>Intro</h1><p>Welcome text.</p>"
            "<h2>Setup</h2><p>Install steps.</p>"
            "<ul><li>Step one</li><li>Step two</li></ul>"
            "</body></html>"
        )

        docs = HTMLLoader().load(f)
        content = docs[0].content

        assert "# Intro" in content
        assert "Welcome text." in content
        assert "## Setup" in content
        assert "- Step one" in content
        assert docs[0].metadata.format == "html"

    def test_strips_script_and_style(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text(
            "<html><head><style>body{color:red}</style></head>"
            "<body><script>alert('x')</script><p>Real content.</p></body></html>"
        )

        content = HTMLLoader().load(f)[0].content

        assert "Real content." in content
        assert "alert" not in content
        assert "color:red" not in content

    def test_falls_back_to_flat_text_when_no_structural_tags(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "page.html"
        f.write_text(
            "<html><body><div>Just a div with text, no p or li.</div></body></html>"
        )

        content = HTMLLoader().load(f)[0].content

        assert "Just a div with text" in content


class TestPDFLoader:
    def test_one_document_per_page_with_page_numbers(self, tmp_path: Path) -> None:
        f = tmp_path / "report.pdf"
        _make_pdf(
            f,
            [
                "First page content, long enough to clear the min_chars floor",
                "Second page content, also long enough to clear the floor",
                "Third page content, likewise well over the threshold",
            ],
        )

        docs = PDFLoader().load(f)

        assert len(docs) == 3
        assert [d.metadata.page for d in docs] == [1, 2, 3]
        assert all(d.metadata.format == "pdf" for d in docs)
        assert "First page" in docs[0].content
        assert "Second page" in docs[1].content

    def test_filters_near_empty_pages_below_min_chars(self, tmp_path: Path) -> None:
        f = tmp_path / "report.pdf"
        _make_pdf(
            f, ["A full page of real substantive content worth indexing", "CampusX"]
        )

        docs = PDFLoader(min_chars=20).load(f)

        assert len(docs) == 1
        assert docs[0].metadata.page == 1


class TestDocumentLoaderRegistry:
    def test_dispatches_by_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hi")
        registry = DocumentLoaderRegistry()

        docs = registry.load_document(f)

        assert docs[0].content == "hi"

    def test_raises_on_unsupported_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "a.docx"
        f.write_text("hi")
        registry = DocumentLoaderRegistry()

        with pytest.raises(UnsupportedFormatError):
            registry.load_document(f)

    def test_load_corpus_walks_nested_dirs_and_relativizes_source(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("root file")
        (tmp_path / "sub" / "b.md").write_text("# nested file")
        (tmp_path / "ignore.bin").write_text("should be skipped")

        registry = DocumentLoaderRegistry()
        docs = registry.load_corpus(tmp_path)

        sources = sorted(d.metadata.source for d in docs)
        assert sources == ["a.txt", str(Path("sub") / "b.md")]

    def test_load_corpus_skips_unreadable_file_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.txt"
        good.write_text("fine")
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a real pdf")

        registry = DocumentLoaderRegistry()
        docs = registry.load_corpus(tmp_path)

        assert len(docs) == 1
        assert docs[0].metadata.source == "good.txt"

    def test_supported_extensions_reflects_registered_loaders(self) -> None:
        registry = DocumentLoaderRegistry()
        assert registry.supported_extensions == {
            ".txt",
            ".md",
            ".markdown",
            ".html",
            ".htm",
            ".pdf",
        }
