from __future__ import annotations

from pathlib import Path

from rag.eval.golden_qa import load_golden_questions

_HEADER = (
    "id,category,question,expected_behavior,source_hint,your_correction,approved\n"
)


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "golden.csv"
    path.write_text(_HEADER + "\n".join(rows) + "\n")
    return path


class TestLoadGoldenQuestions:
    def test_skips_unapproved_rows_by_default(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            [
                "q1,direct_lookup,Approved question?,answerable,hint,,yes",
                "q2,direct_lookup,Unapproved question?,answerable,hint,,",
            ],
        )
        questions = load_golden_questions(path)
        assert [q.id for q in questions] == ["q1"]

    def test_approved_only_false_includes_everything(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            [
                "q1,direct_lookup,Approved?,answerable,hint,,yes",
                "q2,direct_lookup,Unapproved?,answerable,hint,,",
            ],
        )
        questions = load_golden_questions(path, approved_only=False)
        assert {q.id for q in questions} == {"q1", "q2"}

    def test_your_correction_overrides_original_question(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            [
                "q1,direct_lookup,Original wording?,answerable,hint,Corrected wording?,yes"
            ],
        )
        questions = load_golden_questions(path)
        assert questions[0].question == "Corrected wording?"

    def test_blank_correction_keeps_original_question(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path, ["q1,direct_lookup,Original wording?,answerable,hint,,yes"]
        )
        questions = load_golden_questions(path)
        assert questions[0].question == "Original wording?"

    def test_various_truthy_approved_values_all_work(self, tmp_path: Path) -> None:
        for value in ("yes", "Yes", "YES", "y", "true", "1"):
            path = _write_csv(
                tmp_path, [f"q1,direct_lookup,Q?,answerable,hint,,{value}"]
            )
            assert len(load_golden_questions(path)) == 1, (
                f"{value!r} should count as approved"
            )

    def test_no_and_other_values_are_not_approved(self, tmp_path: Path) -> None:
        for value in ("no", "No", "0", "maybe"):
            path = _write_csv(
                tmp_path, [f"q1,direct_lookup,Q?,answerable,hint,,{value}"]
            )
            assert load_golden_questions(path) == [], (
                f"{value!r} should not count as approved"
            )

    def test_preserves_category_and_expected_behavior(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path, ["q1,multi_hop,A question?,insufficient_context,some hint,,yes"]
        )
        q = load_golden_questions(path)[0]
        assert q.category == "multi_hop"
        assert q.expected_behavior == "insufficient_context"
        assert q.source_hint == "some hint"

    def test_handles_utf8_bom_from_excel_windows_export(self, tmp_path: Path) -> None:

        path = tmp_path / "golden.csv"
        path.write_bytes(
            ("\ufeff" + _HEADER + "q1,direct_lookup,Q?,answerable,hint,,yes\n").encode(
                "utf-8"
            )
        )
        questions = load_golden_questions(path)
        assert questions[0].id == "q1"
