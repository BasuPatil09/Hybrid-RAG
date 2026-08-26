from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from rag.generation.confidence import (
    ConfidenceScore,
    GroqCompletenessJudge,
    _parse_completeness,
    _sigmoid,
    compute_retrieval_confidence,
    get_completeness_judge,
    score_answer,
)
from rag.generation.generator import GeneratedAnswer
from rag.generation.verification import VerifiedAnswer
from rag.schemas import Chunk, ChunkMetadata, ScoredChunk


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        content="x",
        metadata=ChunkMetadata(source="f.md", format="md", chunk_index=0, char_count=1),
    )


def _scored(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(cid), score=score)


class TestSigmoid:
    def test_zero_maps_to_half(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self) -> None:
        assert _sigmoid(10.0) > 0.99

    def test_large_negative_approaches_zero(self) -> None:
        assert _sigmoid(-10.0) < 0.01

    def test_matches_observed_reranker_score_scale(self) -> None:

        assert _sigmoid(8.74) > 0.99
        assert _sigmoid(-6.96) < 0.01


class TestParseCompleteness:
    def test_parses_score_and_explanation(self) -> None:
        score, explanation = _parse_completeness(
            "SCORE: 85 - covers most of the question"
        )
        assert score == pytest.approx(0.85)
        assert "covers most" in explanation

    def test_clamps_out_of_range_score(self) -> None:
        score, _ = _parse_completeness("SCORE: 150 - overclaimed")
        assert score == 1.0

    def test_clamps_negative_score(self) -> None:
        score, _ = _parse_completeness("SCORE: -20 - nonsensical")
        assert score == 0.0

    def test_malformed_response_fails_closed_at_zero(self) -> None:
        score, explanation = _parse_completeness("I think it's pretty good.")
        assert score == 0.0
        assert "did not match" in explanation.lower()

    def test_normalizes_unicode_dash_separator_and_explanation_punctuation(
        self,
    ) -> None:

        score, explanation = _parse_completeness(
            "SCORE: 90 \u2013 covers the higher\u2011level point well"
        )
        assert score == pytest.approx(0.9)
        assert explanation == "covers the higher-level point well"


class _FakeGroqClient:
    def __init__(self, responses: list[str] | None = None, fail_times: int = 0) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses or ["SCORE: 100 - default fake"])
        self._fail_times = fail_times
        self._call_count = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(
        self, *, model: str, messages: list[dict], temperature: float
    ) -> SimpleNamespace:
        self._call_count += 1
        self.calls.append(
            {"model": model, "messages": messages, "temperature": temperature}
        )
        if self._call_count <= self._fail_times:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            )
        content = self._responses[min(self._call_count - 1, len(self._responses) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _judge_with_fake_client(responses=None, fail_times=0):
    judge = GroqCompletenessJudge(api_key="fake-key-not-used", sleep_fn=lambda _: None)
    fake_client = _FakeGroqClient(responses=responses, fail_times=fail_times)
    judge._client = fake_client
    return judge, fake_client


class TestGroqCompletenessJudge:
    def test_sends_question_and_answer(self) -> None:
        judge, fake_client = _judge_with_fake_client()
        judge.judge("my question", "my answer text")
        content = fake_client.calls[0]["messages"][1]["content"]
        assert "my question" in content and "my answer text" in content

    def test_uses_zero_temperature(self) -> None:
        judge, fake_client = _judge_with_fake_client()
        judge.judge("q", "a")
        assert fake_client.calls[0]["temperature"] == 0.0

    def test_returns_parsed_score(self) -> None:
        judge, _ = _judge_with_fake_client(responses=["SCORE: 60 - partial coverage"])
        score, explanation = judge.judge("q", "a")
        assert score == pytest.approx(0.6)
        assert "partial" in explanation

    def test_recovers_from_transport_error(self) -> None:
        judge, fake_client = _judge_with_fake_client(fail_times=1)
        judge.judge("q", "a")
        assert fake_client._call_count == 2


class TestGetCompletenessJudge:
    def test_returns_groq_judge(self) -> None:
        assert isinstance(
            get_completeness_judge(
                "groq", api_key="fake", model="llama-3.3-70b-versatile"
            ),
            GroqCompletenessJudge,
        )

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            get_completeness_judge("cohere", api_key="fake", model="x")


class TestComputeRetrievalConfidence:
    def test_sigmoid_scales_cited_chunk_scores(self) -> None:
        reranked = [_scored("a", 10.0)]
        confidence = compute_retrieval_confidence(["a"], reranked)
        assert confidence == pytest.approx(_sigmoid(10.0))

    def test_averages_multiple_cited_chunks(self) -> None:
        reranked = [_scored("a", 10.0), _scored("b", -10.0)]
        confidence = compute_retrieval_confidence(["a", "b"], reranked)
        assert confidence == pytest.approx((_sigmoid(10.0) + _sigmoid(-10.0)) / 2)

    def test_only_considers_cited_chunks_not_every_reranked_one(self) -> None:
        reranked = [_scored("a", 10.0), _scored("b", -10.0)]
        confidence = compute_retrieval_confidence(["a"], reranked)
        assert confidence == pytest.approx(_sigmoid(10.0))

    def test_empty_cited_ids_scores_zero_not_one(self) -> None:

        assert compute_retrieval_confidence([], [_scored("a", 10.0)]) == 0.0

    def test_cited_id_missing_from_reranked_list_is_skipped(self) -> None:

        confidence = compute_retrieval_confidence(["missing"], [_scored("a", 10.0)])
        assert confidence == 0.0


class TestConfidenceScoreComposite:
    def test_averages_all_three_components_equally(self) -> None:
        score = ConfidenceScore(
            retrieval_confidence=0.9,
            citation_coverage=0.6,
            answer_completeness=0.3,
            completeness_explanation="x",
        )
        assert score.composite == pytest.approx((0.9 + 0.6 + 0.3) / 3)


class TestScoreAnswer:
    def test_combines_all_three_signals(self) -> None:
        answer = GeneratedAnswer(
            answer="A claim [1].", is_insufficient=False, cited_chunk_ids=["a"]
        )
        verified = VerifiedAnswer(answer=answer, claim_verdicts=[])
        reranked = [_scored("a", 5.0)]

        class _FakeJudge:
            def judge(self, query: str, answer_text: str) -> tuple[float, str]:
                return 0.75, "mostly complete"

        result = score_answer("q", answer, verified, reranked, _FakeJudge())

        assert result.retrieval_confidence == pytest.approx(_sigmoid(5.0))
        assert result.citation_coverage == 1.0
        assert result.answer_completeness == pytest.approx(0.75)
        assert result.completeness_explanation == "mostly complete"
