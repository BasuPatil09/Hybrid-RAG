from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel

_TRUTHY = {"yes", "y", "true", "1"}
_VALID_CATEGORIES = {
    "direct_lookup",
    "multi_hop",
    "unanswerable",
    "ambiguous",
}
_VALID_BEHAVIORS = {
    "answerable",
    "insufficient_context",
    "ambiguous",
}

class GoldenQuestion(BaseModel):
    id: str
    category: str
    question: str
    expected_behavior: str
    source_hint: str = ""

def load_golden_questions(
    csv_path: Path,
    *,
    approved_only: bool = True,
) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id", "<unknown>")

            category = row.get("category", "").strip()
            if category not in _VALID_CATEGORIES:
                raise ValueError(
                    f"{row_id}: malformed row, "
                    f"category={category!r}; "
                    f"expected one of {sorted(_VALID_CATEGORIES)}"
                )

            expected_behavior = row.get("expected_behavior", "").strip()
            if expected_behavior not in _VALID_BEHAVIORS:
                raise ValueError(
                    f"{row_id}: malformed row, "
                    f"expected_behavior={expected_behavior!r}; "
                    f"expected one of {sorted(_VALID_BEHAVIORS)}"
                )

            if approved_only and row.get("approved", "").strip().lower() not in _TRUTHY:
                continue
            question = row.get("question", "").strip()
            correction = row.get("your_correction", "").strip()
            question_text = correction or question

            if not question_text:
                raise ValueError(f"{row_id}: malformed row, question is empty")

            questions.append(
                GoldenQuestion(
                    id=row_id,
                    category=category,
                    question=question_text,
                    expected_behavior=expected_behavior,
                    source_hint=row.get("source_hint", "").strip(),
                )
            )
    return questions