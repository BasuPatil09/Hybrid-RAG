from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from rag.generation.generator import (
    GroqGenerator,
    _build_user_prompt,
    _parse_response,
    get_generator,
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


class _FakeGroqClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        fail_times: int = 0,
        fail_with: Exception | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses or ["default fake answer [1]"])
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
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _generator_with_fake_client(
    responses: list[str] | None = None,
    fail_times: int = 0,
    fail_with: Exception | None = None,
):
    generator = GroqGenerator(api_key="fake-key-not-used", sleep_fn=lambda _: None)
    fake_client = _FakeGroqClient(
        responses=responses, fail_times=fail_times, fail_with=fail_with
    )
    generator._client = fake_client
    return generator, fake_client


class TestBuildUserPrompt:
    def test_numbers_excerpts_starting_at_one(self) -> None:
        chunks = [_chunk("a", "first content"), _chunk("b", "second content")]
        prompt = _build_user_prompt("my question", chunks)
        assert "[1] first content" in prompt
        assert "[2] second content" in prompt
        assert "Question: my question" in prompt


class TestParseResponse:
    def test_extracts_cited_chunk_ids_in_first_cited_order(self) -> None:
        chunks = [_chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")]
        result = _parse_response(
            "Some claim [2]. Another claim [1]. Same again [2].", chunks
        )
        assert result.cited_chunk_ids == ["b", "a"]
        assert result.is_insufficient is False

    def test_detects_insufficient_context_prefix(self) -> None:
        chunks = [_chunk("a", "x")]
        result = _parse_response(
            "INSUFFICIENT_CONTEXT: the excerpts don't mention this topic.", chunks
        )
        assert result.is_insufficient is True

    def test_out_of_range_citation_indices_are_ignored_not_errored(self) -> None:
        chunks = [_chunk("a", "x")]
        result = _parse_response("A claim [1] and a bogus one [99].", chunks)
        assert result.cited_chunk_ids == ["a"]

    def test_no_citations_returns_empty_list(self) -> None:
        chunks = [_chunk("a", "x")]
        result = _parse_response("An answer with no citation markers at all.", chunks)
        assert result.cited_chunk_ids == []

    def test_answer_is_stripped_of_surrounding_whitespace(self) -> None:
        result = _parse_response("  \n  answer text [1]  \n  ", [_chunk("a", "x")])
        assert result.answer == "answer text [1]"

    def test_normalizes_non_breaking_hyphen_to_ascii(self) -> None:

        result = _parse_response("higher\u2011level features [1]", [_chunk("a", "x")])
        assert result.answer == "higher-level features [1]"

    def test_normalizes_curly_quotes_and_non_breaking_space(self) -> None:
        result = _parse_response(
            "a \u201cquoted\u201d word\u00a0here [1]", [_chunk("a", "x")]
        )
        assert result.answer == 'a "quoted" word here [1]'

    def test_normalization_does_not_affect_citation_parsing(self) -> None:
        result = _parse_response(
            "claim one\u2013still cited [1][2]", [_chunk("a", "x"), _chunk("b", "y")]
        )
        assert result.cited_chunk_ids == ["a", "b"]

    def test_normalization_does_not_affect_insufficient_context_detection(self) -> None:
        result = _parse_response(
            "INSUFFICIENT_CONTEXT: missing the \u2018key\u2019 detail.",
            [_chunk("a", "x")],
        )
        assert result.is_insufficient is True


class TestGroqGenerator:
    def test_sends_system_and_user_messages(self) -> None:
        generator, fake_client = _generator_with_fake_client()
        generator.generate("my question", [_chunk("a", "some content")])
        messages = fake_client.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "my question" in messages[1]["content"]

    def test_uses_zero_temperature_for_deterministic_grounded_answers(self) -> None:
        generator, fake_client = _generator_with_fake_client()
        generator.generate("q", [_chunk("a", "x")])
        assert fake_client.calls[0]["temperature"] == 0.0

    def test_returns_parsed_answer_with_citations(self) -> None:
        generator, _ = _generator_with_fake_client(responses=["The answer is X [1]."])
        result = generator.generate("q", [_chunk("target", "supporting content")])
        assert result.cited_chunk_ids == ["target"]
        assert result.is_insufficient is False

    def test_recovers_from_a_transport_error(self) -> None:
        generator, fake_client = _generator_with_fake_client(fail_times=1)
        result = generator.generate("q", [_chunk("a", "x")])
        assert fake_client._call_count == 2
        assert result.answer

    def test_gives_up_after_max_attempts(self) -> None:
        generator, _ = _generator_with_fake_client(fail_times=99)
        with pytest.raises(httpx.TransportError):
            generator.generate("q", [_chunk("a", "x")])

    def test_non_transport_errors_are_not_retried(self) -> None:

        generator, fake_client = _generator_with_fake_client(
            fail_times=99, fail_with=RuntimeError("rate limited")
        )
        with pytest.raises(RuntimeError):
            generator.generate("q", [_chunk("a", "x")])
        assert fake_client._call_count == 1


class TestFormatComplianceRetry:
    def test_retries_once_when_answer_has_no_citations(self) -> None:
        generator, fake_client = _generator_with_fake_client(
            responses=[
                "Deep learning uses layers 【L1-L4】.",
                "Deep learning uses layers [1].",
            ]
        )
        result = generator.generate("q", [_chunk("target", "supporting content")])
        assert fake_client._call_count == 2
        assert result.cited_chunk_ids == ["target"]

    def test_retry_prompt_includes_format_correction_note(self) -> None:
        generator, fake_client = _generator_with_fake_client(
            responses=["no citations here", "still none"]
        )
        generator.generate("q", [_chunk("a", "x")])
        retry_user_message = fake_client.calls[1]["messages"][1]["content"]
        assert "required [N] format" in retry_user_message

    def test_does_not_retry_when_citations_already_present(self) -> None:
        generator, fake_client = _generator_with_fake_client(
            responses=["The answer is X [1]."]
        )
        generator.generate("q", [_chunk("target", "x")])
        assert fake_client._call_count == 1

    def test_does_not_retry_insufficient_context_answers(self) -> None:
        generator, fake_client = _generator_with_fake_client(
            responses=["INSUFFICIENT_CONTEXT: the excerpts don't cover this."]
        )
        generator.generate("q", [_chunk("a", "x")])
        assert fake_client._call_count == 1

    def test_fails_closed_if_retry_also_has_no_citations(self) -> None:
        generator, fake_client = _generator_with_fake_client(
            responses=["still no brackets here", "and still none here either"]
        )
        result = generator.generate("q", [_chunk("a", "x")])
        assert fake_client._call_count == 2
        assert result.cited_chunk_ids == []


class TestGetGenerator:
    def test_returns_groq_generator(self) -> None:
        generator = get_generator(
            "groq", api_key="fake", model="llama-3.3-70b-versatile"
        )
        assert isinstance(generator, GroqGenerator)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            get_generator("cohere", api_key="fake", model="x")
