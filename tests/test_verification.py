from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from rag.generation.generator import GeneratedAnswer
from rag.generation.verification import (
    GroqCitationVerifier,
    _combine_chunk_contents,
    _parse_verdict,
    extract_claims,
    get_citation_verifier,
    verify_answer,
)
from rag.schemas import Chunk, ChunkMetadata


def _chunk(cid: str, content: str) -> Chunk:
    return Chunk(
        id=cid,
        content=content,
        metadata=ChunkMetadata(
            source="f.md", format="md", chunk_index=0, char_count=len(content)
        ),
    )


class TestExtractClaims:
    def test_splits_into_one_claim_per_sentence(self) -> None:
        chunks = [_chunk("a", "x"), _chunk("b", "y")]
        claims = extract_claims("First claim [1]. Second claim [2].", chunks)
        assert [c.text for c in claims] == ["First claim.", "Second claim."]
        assert [[ch.id for ch in c.chunks] for c in claims] == [["a"], ["b"]]

    def test_sentences_without_citations_are_skipped(self) -> None:
        chunks = [_chunk("a", "x")]
        claims = extract_claims("No citation here. Cited claim [1].", chunks)
        assert len(claims) == 1
        assert claims[0].text == "Cited claim."

    def test_multiple_citations_in_one_sentence_produce_one_claim_with_all_chunks(
        self,
    ) -> None:

        chunks = [_chunk("a", "x"), _chunk("b", "y")]
        claims = extract_claims("A claim citing two sources [1][2].", chunks)
        assert len(claims) == 1
        assert [c.id for c in claims[0].chunks] == ["a", "b"]
        assert claims[0].text == "A claim citing two sources."

    def test_citation_markers_are_stripped_from_claim_text(self) -> None:
        claims = extract_claims("The claim [1].", [_chunk("a", "x")])
        assert "[1]" not in claims[0].text

    def test_out_of_range_citation_index_is_skipped_not_errored(self) -> None:
        chunks = [_chunk("a", "x")]
        claims = extract_claims("A hallucinated citation [99].", chunks)
        assert claims == []

    def test_out_of_range_citation_mixed_with_valid_keeps_only_the_valid_chunk(
        self,
    ) -> None:
        chunks = [_chunk("a", "x")]
        claims = extract_claims("Partially valid citation [1][99].", chunks)
        assert len(claims) == 1
        assert [c.id for c in claims[0].chunks] == ["a"]

    def test_repeated_identical_citation_in_one_sentence_deduplicates(self) -> None:
        chunks = [_chunk("a", "x")]
        claims = extract_claims("Same source cited twice [1][1].", chunks)
        assert len(claims) == 1
        assert len(claims[0].chunks) == 1

    def test_empty_answer_returns_no_claims(self) -> None:
        assert extract_claims("", [_chunk("a", "x")]) == []


class TestExtractClaimsCitationAfterPeriod:
    def test_marker_after_the_period_stays_attached_to_its_sentence(self) -> None:
        chunks = [_chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")]
        claims = extract_claims(
            "I answer questions rather than act autonomously. [2][3]", chunks
        )
        assert len(claims) == 1
        assert claims[0].text == "I answer questions rather than act autonomously."
        assert [c.id for c in claims[0].chunks] == ["b", "c"]

    def test_never_produces_a_claim_with_empty_text(self) -> None:
        claims = extract_claims("[1]", [_chunk("a", "x")])
        assert claims == []

    def test_following_sentence_still_splits_normally(self) -> None:
        chunks = [_chunk("a", "x"), _chunk("b", "y")]
        claims = extract_claims("First claim. [1] Second claim. [2]", chunks)
        assert [c.text for c in claims] == ["First claim.", "Second claim."]


class TestCombineChunkContents:
    def test_single_chunk_passes_through_unchanged(self) -> None:

        assert _combine_chunk_contents([_chunk("a", "solo content")]) == "solo content"

    def test_multiple_chunks_are_joined_with_a_visible_separator(self) -> None:
        combined = _combine_chunk_contents(
            [_chunk("a", "first"), _chunk("b", "second")]
        )
        assert "first" in combined
        assert "second" in combined
        assert "---" in combined

    def test_preserves_chunk_order(self) -> None:
        combined = _combine_chunk_contents(
            [_chunk("a", "AAA"), _chunk("b", "BBB"), _chunk("c", "CCC")]
        )
        assert combined.index("AAA") < combined.index("BBB") < combined.index("CCC")


class TestParseVerdict:
    def test_supported_response(self) -> None:
        supported, explanation = _parse_verdict(
            "SUPPORTED: the source directly states this."
        )
        assert supported is True
        assert "directly states" in explanation

    def test_not_supported_response(self) -> None:
        supported, explanation = _parse_verdict(
            "NOT_SUPPORTED: the source never mentions this."
        )
        assert supported is False
        assert "never mentions" in explanation

    def test_malformed_response_fails_closed_as_not_supported(self) -> None:

        supported, explanation = _parse_verdict("I'm not sure, maybe?")
        assert supported is False
        assert "did not match" in explanation.lower()

    def test_case_insensitive_prefix_matching(self) -> None:
        supported, _ = _parse_verdict("supported: lowercase still works")
        assert supported is True

    def test_normalizes_punctuation_in_explanation(self) -> None:

        supported, explanation = _parse_verdict(
            "SUPPORTED: mentions higher\u2011level features directly."
        )
        assert explanation == "mentions higher-level features directly."

    def test_extracts_verdict_even_when_preceded_by_reasoning(self) -> None:

        raw = "Let me check each part of this claim first. NOT_SUPPORTED: the second part is missing."
        supported, explanation = _parse_verdict(raw)
        assert supported is False
        assert "second part is missing" in explanation

    def test_takes_the_last_verdict_when_more_than_one_appears(self) -> None:
        raw = "SUPPORTED: initial thought. Actually, on reflection, NOT_SUPPORTED: this is missing something."
        supported, _ = _parse_verdict(raw)
        assert supported is False


class _FakeGroqClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        fail_times: int = 0,
        fail_with: Exception | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses or ["SUPPORTED: default fake verdict"])
        self._fail_times = fail_times
        self._fail_with = fail_with or httpx.RemoteProtocolError(
            "Server disconnected without sending a response."
        )
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
            raise self._fail_with
        content = self._responses[min(self._call_count - 1, len(self._responses) - 1)]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _verifier_with_fake_client(responses=None, fail_times=0, fail_with=None):
    verifier = GroqCitationVerifier(
        api_key="fake-key-not-used", sleep_fn=lambda _: None
    )
    fake_client = _FakeGroqClient(
        responses=responses, fail_times=fail_times, fail_with=fail_with
    )
    verifier._client = fake_client
    return verifier, fake_client


class TestGroqCitationVerifier:
    def test_sends_claim_and_source_in_user_message(self) -> None:
        verifier, fake_client = _verifier_with_fake_client()
        verifier.verify_claim("my claim", "my source text")
        content = fake_client.calls[0]["messages"][1]["content"]
        assert "my claim" in content and "my source text" in content

    def test_uses_zero_temperature(self) -> None:
        verifier, fake_client = _verifier_with_fake_client()
        verifier.verify_claim("c", "s")
        assert fake_client.calls[0]["temperature"] == 0.0

    def test_returns_parsed_verdict(self) -> None:
        verifier, _ = _verifier_with_fake_client(
            responses=["NOT_SUPPORTED: contradicts the claim"]
        )
        supported, explanation = verifier.verify_claim("c", "s")
        assert supported is False
        assert "contradicts" in explanation

    def test_recovers_from_transport_error(self) -> None:
        verifier, fake_client = _verifier_with_fake_client(fail_times=1)
        verifier.verify_claim("c", "s")
        assert fake_client._call_count == 2


class TestGetCitationVerifier:
    def test_returns_groq_verifier(self) -> None:
        assert isinstance(
            get_citation_verifier(
                "groq", api_key="fake", model="llama-3.3-70b-versatile"
            ),
            GroqCitationVerifier,
        )

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            get_citation_verifier("cohere", api_key="fake", model="x")


class _FakeVerifier:
    def __init__(self, supported: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self._supported = supported

    def verify_claim(self, claim_text: str, source_text: str) -> tuple[bool, str]:
        self.calls.append((claim_text, source_text))
        return self._supported, "fake verdict"


class TestVerifyAnswer:
    def test_insufficient_context_answer_is_never_sent_to_the_verifier(self) -> None:
        answer = GeneratedAnswer(
            answer="INSUFFICIENT_CONTEXT: nothing relevant [1].",
            is_insufficient=True,
            cited_chunk_ids=["a"],
        )
        verifier = _FakeVerifier()

        result = verify_answer(answer, [_chunk("a", "x")], verifier)

        assert result.claim_verdicts == []
        assert verifier.calls == []

    def test_normal_answer_verifies_each_claim(self) -> None:
        answer = GeneratedAnswer(
            answer="First claim [1]. Second claim [2].",
            is_insufficient=False,
            cited_chunk_ids=["a", "b"],
        )
        verifier = _FakeVerifier(supported=True)

        result = verify_answer(answer, [_chunk("a", "x"), _chunk("b", "y")], verifier)

        assert len(result.claim_verdicts) == 2
        assert len(verifier.calls) == 2

    def test_multi_citation_sentence_calls_verifier_once_not_once_per_chunk(
        self,
    ) -> None:

        answer = GeneratedAnswer(
            answer="A synthesized claim [1][2].",
            is_insufficient=False,
            cited_chunk_ids=["a", "b"],
        )
        verifier = _FakeVerifier(supported=True)

        result = verify_answer(
            answer,
            [_chunk("a", "alpha content"), _chunk("b", "beta content")],
            verifier,
        )

        assert len(verifier.calls) == 1
        claim_text, source_text = verifier.calls[0]
        assert "alpha content" in source_text and "beta content" in source_text
        assert len(result.claim_verdicts) == 1

    def test_multi_citation_verdict_carries_every_cited_chunk_id(self) -> None:
        answer = GeneratedAnswer(
            answer="A synthesized claim [1][2].",
            is_insufficient=False,
            cited_chunk_ids=["a", "b"],
        )
        result = verify_answer(
            answer, [_chunk("a", "x"), _chunk("b", "y")], _FakeVerifier(supported=True)
        )
        assert result.claim_verdicts[0].chunk_ids == ["a", "b"]

    def test_citation_accuracy_is_one_when_no_claims(self) -> None:
        answer = GeneratedAnswer(
            answer="INSUFFICIENT_CONTEXT: n/a [1].",
            is_insufficient=True,
            cited_chunk_ids=["a"],
        )
        result = verify_answer(answer, [_chunk("a", "x")], _FakeVerifier())
        assert result.citation_accuracy == 1.0

    def test_citation_accuracy_reflects_mixed_verdicts(self) -> None:
        answer = GeneratedAnswer(
            answer="Claim one [1]. Claim two [2].",
            is_insufficient=False,
            cited_chunk_ids=["a", "b"],
        )

        class _MixedVerifier:
            def __init__(self) -> None:
                self._n = 0

            def verify_claim(
                self, claim_text: str, source_text: str
            ) -> tuple[bool, str]:
                self._n += 1
                return self._n == 1, "verdict"

        result = verify_answer(
            answer, [_chunk("a", "x"), _chunk("b", "y")], _MixedVerifier()
        )
        assert result.citation_accuracy == 0.5
