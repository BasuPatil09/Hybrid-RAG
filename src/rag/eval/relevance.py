from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from rag.schemas import Chunk

_TRUTHY = {"yes", "y", "true", "1"}
LABEL_COLUMNS = (
    "question_id",
    "source",
    "page",
    "llm_relevant",
    "human_relevant",
    "snippet",
)


def normalize_source(source: str) -> str:
    return source.replace("\\", "/").strip().lower()


class RelevanceUnit(BaseModel, frozen=True):
    question_id: str
    source: str
    page: int | None = None

    def label(self) -> str:
        return f"{self.source}" + (f" p{self.page}" if self.page is not None else "")

    def matches(self, chunk: Chunk) -> bool:
        if normalize_source(chunk.metadata.source) != self.source:
            return False
        return self.page is None or chunk.metadata.page == self.page


class RelevanceLabels:
    def __init__(self, units: Iterable[RelevanceUnit]) -> None:
        self._by_question: dict[str, list[RelevanceUnit]] = {}
        for unit in units:
            self._by_question.setdefault(unit.question_id, []).append(unit)

    @property
    def question_ids(self) -> set[str]:
        return set(self._by_question)

    def units_for(self, question_id: str) -> list[RelevanceUnit]:
        return list(self._by_question.get(question_id, []))

    def matches(self, question_id: str, chunk: Chunk) -> frozenset[RelevanceUnit]:
        return frozenset(
            u for u in self._by_question.get(question_id, []) if u.matches(chunk)
        )


def load_relevance_labels(csv_path: Path) -> RelevanceLabels:
    units: list[RelevanceUnit] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("human_relevant", "").strip().lower() not in _TRUTHY:
                continue
            question_id = row.get("question_id", "").strip()
            source = row.get("source", "").strip()
            if not question_id or not source:
                raise ValueError(f"malformed label row (id/source missing): {row}")
            page_text = (row.get("page") or "").strip()
            units.append(
                RelevanceUnit(
                    question_id=question_id,
                    source=normalize_source(source),
                    page=int(page_text) if page_text else None,
                )
            )
    return RelevanceLabels(units)


def existing_question_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return {
            r["question_id"].strip() for r in csv.DictReader(f) if r.get("question_id")
        }


def write_draft_labels(
    csv_path: Path, rows: Iterable[dict[str, object]], *, append: bool = False
) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    appending = append and csv_path.exists()
    with open(
        csv_path, "a" if appending else "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        if not appending:
            writer.writeheader()
        for row in rows:
            writer.writerow({**{c: "" for c in LABEL_COLUMNS}, **row})
            count += 1
    return count
