from __future__ import annotations

from pathlib import Path

import pytest

from rag.eval.relevance import (
    load_relevance_labels,
    normalize_source,
    write_draft_labels,
)
from rag.schemas import Chunk, ChunkMetadata

_HEADER = "question_id,source,page,llm_relevant,human_relevant,snippet\n"


def _csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _chunk(source: str, page: int | None) -> Chunk:
    return Chunk(
        id=f"{source}::{page}",
        content="x",
        metadata=ChunkMetadata(
            source=source, format="pdf", page=page, chunk_index=0, char_count=1
        ),
    )


class TestLoad:
    def test_only_human_approved_rows_count_as_ground_truth(self, tmp_path) -> None:
        path = _csv(
            tmp_path,
            [
                "q1,a.pdf,3,yes,yes,s",
                "q1,a.pdf,4,yes,,s",
                "q1,a.pdf,5,no,no,s",
            ],
        )
        units = load_relevance_labels(path).units_for("q1")
        assert [u.page for u in units] == [3]

    def test_blank_page_means_any_page_of_that_source(self, tmp_path) -> None:
        labels = load_relevance_labels(_csv(tmp_path, ["q1,a.pdf,,yes,yes,s"]))
        assert labels.matches("q1", _chunk("a.pdf", 99))
        assert not labels.matches("q1", _chunk("b.pdf", 1))

    def test_page_label_matches_only_that_page(self, tmp_path) -> None:
        labels = load_relevance_labels(_csv(tmp_path, ["q1,a.pdf,3,yes,yes,s"]))
        assert labels.matches("q1", _chunk("a.pdf", 3))
        assert not labels.matches("q1", _chunk("a.pdf", 4))

    def test_source_match_ignores_case_and_path_separators(self, tmp_path) -> None:
        labels = load_relevance_labels(_csv(tmp_path, ["q1,Sub/A.pdf,1,yes,yes,s"]))
        assert labels.matches("q1", _chunk("sub\\a.PDF", 1))

    def test_labels_are_scoped_to_their_question(self, tmp_path) -> None:
        labels = load_relevance_labels(_csv(tmp_path, ["q1,a.pdf,1,yes,yes,s"]))
        assert not labels.matches("q2", _chunk("a.pdf", 1))

    def test_handles_utf8_bom(self, tmp_path) -> None:
        path = tmp_path / "labels.csv"
        path.write_bytes(("\ufeff" + _HEADER + "q1,a.pdf,1,yes,yes,s\n").encode())
        assert load_relevance_labels(path).question_ids == {"q1"}

    def test_row_missing_source_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="malformed"):
            load_relevance_labels(_csv(tmp_path, ["q1,,1,yes,yes,s"]))


def test_normalize_source() -> None:
    assert normalize_source(" Folder\\File.PDF ") == "folder/file.pdf"


def test_draft_sheet_leaves_human_column_blank_so_nothing_counts_until_reviewed(
    tmp_path,
) -> None:
    path = tmp_path / "out" / "labels.csv"
    n = write_draft_labels(
        path,
        [{"question_id": "q1", "source": "a.pdf", "page": 3, "llm_relevant": "yes"}],
    )
    assert n == 1
    assert load_relevance_labels(path).question_ids == set()
