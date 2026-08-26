from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Sequence

from bs4 import BeautifulSoup
from pypdf import PdfReader

from rag.schemas import Document, DocumentMetadata

logger = logging.getLogger(__name__)

_HEADING_PREFIXES: dict[str, str] = {
    "h1": "#",
    "h2": "##",
    "h3": "###",
    "h4": "####",
    "h5": "#####",
    "h6": "######",
}

class UnsupportedFormatError(ValueError):
    pass

class BaseLoader(ABC):
    extensions: ClassVar[frozenset[str]]
    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        pass

class TextLoader(BaseLoader):
    extensions: ClassVar[frozenset[str]] = frozenset({".txt"})

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(
                content=content,
                metadata=DocumentMetadata(source=str(path), format="txt"),
            )
        ]

class MarkdownLoader(BaseLoader):
    extensions: ClassVar[frozenset[str]] = frozenset({".md", ".markdown"})

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(
                content=content,
                metadata=DocumentMetadata(source=str(path), format="md"),
            )
        ]

class HTMLLoader(BaseLoader):
    extensions: ClassVar[frozenset[str]] = frozenset({".html", ".htm"})

    def load(self, path: Path) -> list[Document]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        content = self._to_pseudo_markdown(raw)
        return [
            Document(
                content=content,
                metadata=DocumentMetadata(source=str(path), format="html"),
            )
        ]

    @staticmethod
    def _to_pseudo_markdown(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()

        blocks: list[str] = []
        for el in soup.find_all(list(_HEADING_PREFIXES) + ["p", "li"]):
            text = el.get_text(strip=True)
            if not text:
                continue
            if el.name in _HEADING_PREFIXES:
                blocks.append(f"{_HEADING_PREFIXES[el.name]} {text}")
            elif el.name == "li":
                blocks.append(f"- {text}")
            else:
                blocks.append(text)

        if not blocks:
            return soup.get_text(separator="\n\n", strip=True)
        return "\n\n".join(blocks)


class PDFLoader(BaseLoader):
    extensions: ClassVar[frozenset[str]] = frozenset({".pdf"})
    def __init__(self, min_chars: int = 20) -> None:
        self._min_chars = min_chars

    def load(self, path: Path) -> list[Document]:
        reader = PdfReader(str(path))
        documents: list[Document] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < self._min_chars:
                logger.debug(
                    "Page %d of %s has %d chars of extractable text (< min_chars=%d); skipping.",
                    page_num,
                    path,
                    len(text),
                    self._min_chars,
                )
                continue
            documents.append(
                Document(
                    content=text,
                    metadata=DocumentMetadata(
                        source=str(path), format="pdf", page=page_num
                    ),
                )
            )
        return documents

def _default_loaders(*, pdf_min_page_chars: int = 20) -> list[BaseLoader]:
    return [
        TextLoader(),
        MarkdownLoader(),
        HTMLLoader(),
        PDFLoader(min_chars=pdf_min_page_chars),
    ]

class DocumentLoaderRegistry:
    def __init__(
        self,
        loaders: Sequence[BaseLoader] | None = None,
        *,
        pdf_min_page_chars: int = 20,
    ) -> None:
        self._by_extension: dict[str, BaseLoader] = {}
        for loader in loaders or _default_loaders(
            pdf_min_page_chars=pdf_min_page_chars
        ):
            for ext in loader.extensions:
                self._by_extension[ext] = loader

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._by_extension)

    def load_document(self, path: Path) -> list[Document]:

        loader = self._by_extension.get(path.suffix.lower())
        if loader is None:
            raise UnsupportedFormatError(
                f"No loader registered for extension '{path.suffix}' ({path})"
            )
        return loader.load(path)

    def load_corpus(self, directory: Path) -> list[Document]:

        documents: list[Document] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self._by_extension:
                continue
            try:
                for doc in self.load_document(path):
                    doc.metadata.source = str(path.relative_to(directory))
                    documents.append(doc)
            except Exception:
                logger.exception("Failed to load %s; skipping.", path)
        return documents
