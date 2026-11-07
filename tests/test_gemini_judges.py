from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from rag.config import Settings
from rag.generation.confidence import (
    GeminiCompletenessJudge,
    GroqCompletenessJudge,
    get_completeness_judge,
)
from rag.generation.gemini_client import GeminiTextClient
from rag.generation.judges import build_judges, missing_credentials_message
from rag.generation.verification import (
    GeminiCitationVerifier,
    GroqCitationVerifier,
    _parse_verdict,
    get_citation_verifier,
)


class _FakeTextClient:
    def __init__(self, response: str = "SUPPORTED: fake") -> None:
        self.calls: list[tuple[str, str]] = []
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._response


class TestGeminiCitationVerifier:
    def test_sends_claim_and_source_and_parses_verdict(self) -> None:
        client = _FakeTextClient("NOT_SUPPORTED: second part missing")
        supported, explanation = GeminiCitationVerifier(client).verify_claim(
            "my claim", "my source"
        )
        system_prompt, user_prompt = client.calls[0]
        assert "fact-checking judge" in system_prompt
        assert (
            "CLAIM: my claim" in user_prompt and "SOURCE TEXT: my source" in user_prompt
        )
        assert supported is False
        assert "second part missing" in explanation

    def test_tolerates_markdown_bold_around_the_verdict(self) -> None:
        supported, explanation = _parse_verdict("**SUPPORTED:** the source says so.")
        assert supported is True
        assert explanation == "the source says so."

    def test_empty_response_fails_closed(self) -> None:
        supported, explanation = GeminiCitationVerifier(
            _FakeTextClient("")
        ).verify_claim("c", "s")
        assert supported is False
        assert "did not match" in explanation.lower()


class TestGeminiCompletenessJudge:
    def test_sends_question_and_answer_and_parses_score(self) -> None:
        client = _FakeTextClient("SCORE: 80 - covers most of it")
        score, explanation = GeminiCompletenessJudge(client).judge("q?", "an answer")
        _, user_prompt = client.calls[0]
        assert "QUESTION: q?" in user_prompt and "ANSWER: an answer" in user_prompt
        assert score == pytest.approx(0.8)
        assert "covers most" in explanation

    def test_tolerates_markdown_bold_around_the_score(self) -> None:
        score, explanation = GeminiCompletenessJudge(
            _FakeTextClient("**SCORE: 90** - complete")
        ).judge("q", "a")
        assert score == pytest.approx(0.9)
        assert explanation == "complete"

    def test_empty_response_fails_closed_at_zero(self) -> None:
        score, _ = GeminiCompletenessJudge(_FakeTextClient("")).judge("q", "a")
        assert score == 0.0


class TestFactories:
    def test_gemini_provider_uses_the_injected_client(self) -> None:
        client = _FakeTextClient()
        verifier = get_citation_verifier(
            "gemini", api_key="k", model="m", text_client=client
        )
        judge = get_completeness_judge(
            "gemini", api_key="k", model="m", text_client=client
        )
        assert isinstance(verifier, GeminiCitationVerifier)
        assert isinstance(judge, GeminiCompletenessJudge)

    def test_gemini_provider_builds_its_own_client_when_none_injected(self) -> None:
        assert isinstance(
            get_citation_verifier("gemini", api_key="fake", model="m"),
            GeminiCitationVerifier,
        )

    def test_groq_provider_is_unchanged(self) -> None:
        assert isinstance(
            get_citation_verifier("groq", api_key="fake", model="m"),
            GroqCitationVerifier,
        )
        assert isinstance(
            get_completeness_judge("groq", api_key="fake", model="m"),
            GroqCompletenessJudge,
        )


def _settings(**overrides) -> Settings:
    base = {"groq_api_key": "g", "gemini_api_key": "x", "_env_file": None}
    return Settings(**{**base, **overrides})


class TestBuildJudges:
    def test_gemini_judges_share_one_client_so_the_throttle_is_global(self) -> None:
        verifier, judge = build_judges(_settings(judge_requests_per_minute=10))
        assert isinstance(verifier, GeminiCitationVerifier)
        assert isinstance(judge, GeminiCompletenessJudge)
        assert verifier._client is judge._client

    def test_groq_judges_when_configured(self) -> None:
        verifier, judge = build_judges(
            _settings(judge_provider="groq", judge_model="llama-3.3-70b-versatile")
        )
        assert isinstance(verifier, GroqCitationVerifier)
        assert isinstance(judge, GroqCompletenessJudge)

    def test_warns_when_judge_is_the_generator(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            build_judges(
                _settings(
                    judge_provider="groq",
                    judge_model="llama-3.3-70b-versatile",
                    generation_model="llama-3.3-70b-versatile",
                )
            )
        assert "grading its own output" in caplog.text

    def test_groq_provider_with_a_gemini_model_id_is_rejected_at_startup(self) -> None:
        with pytest.raises(ValueError, match="Groq JUDGE_MODEL"):
            _settings(judge_provider="groq", judge_model="gemini-3.5-flash-lite")


class TestMissingKeyDoesNotCrashStartup:
    def test_building_judges_without_a_gemini_key_succeeds(self) -> None:
        verifier, judge = build_judges(_settings(gemini_api_key=""))
        assert isinstance(verifier, GeminiCitationVerifier)

    def test_the_missing_key_only_surfaces_when_a_call_is_made(self) -> None:
        verifier, _ = build_judges(_settings(gemini_api_key=""))
        with pytest.raises(ValueError, match="API key"):
            verifier.verify_claim("claim", "source")


class TestMissingCredentials:
    def test_groq_key_required(self) -> None:
        assert "GROQ_API_KEY" in missing_credentials_message(_settings(groq_api_key=""))

    def test_gemini_key_required_only_for_gemini_judge(self) -> None:
        assert "GEMINI_API_KEY" in missing_credentials_message(
            _settings(gemini_api_key="")
        )
        assert (
            missing_credentials_message(
                _settings(
                    gemini_api_key="",
                    judge_provider="groq",
                    judge_model="llama-3.3-70b-versatile",
                )
            )
            is None
        )

    def test_all_present(self) -> None:
        assert missing_credentials_message(_settings()) is None


class _FakeGenaiModels:
    def __init__(self, outcomes: list) -> None:
        self.calls: list[dict] = []
        self._outcomes = list(outcomes)

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def _client(outcomes, **kwargs) -> tuple[GeminiTextClient, _FakeGenaiModels, list]:
    sleeps: list[float] = []
    client = GeminiTextClient(
        api_key="fake", model="gemini-test", sleep_fn=sleeps.append, **kwargs
    )
    fake = _FakeGenaiModels(outcomes)
    client._client = SimpleNamespace(models=fake)
    return client, fake, sleeps


def _api_error(code: int, status: str, **extra) -> genai_errors.APIError:
    body = {"error": {"code": code, "message": "boom", "status": status, **extra}}
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, body)


class TestGeminiTextClient:
    def test_sends_model_prompt_and_deterministic_config(self) -> None:
        client, fake, _ = _client(["hello"], thinking_level="low")
        assert client.complete("sys prompt", "user prompt") == "hello"
        call = fake.calls[0]
        assert call["model"] == "gemini-test"
        assert call["contents"] == "user prompt"
        assert call["config"].system_instruction == "sys prompt"
        assert call["config"].temperature == 0.0
        assert call["config"].thinking_config.thinking_level.value == "LOW"

    def test_no_thinking_config_when_level_is_empty(self) -> None:
        client, fake, _ = _client(["hello"], thinking_level="")
        client.complete("s", "u")
        assert fake.calls[0]["config"].thinking_config is None

    def test_none_text_from_a_blocked_response_becomes_empty_string(self) -> None:
        client, _, _ = _client([None])
        assert client.complete("s", "u") == ""

    def test_retries_a_429_using_the_servers_retry_delay(self) -> None:
        rate_limited = _api_error(
            429,
            "RESOURCE_EXHAUSTED",
            details=[
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "12s",
                }
            ],
        )
        client, fake, sleeps = _client([rate_limited, "ok"])
        assert client.complete("s", "u") == "ok"
        assert len(fake.calls) == 2
        assert sleeps == [pytest.approx(12.25)]

    def test_retries_a_503(self) -> None:
        client, fake, _ = _client([_api_error(503, "UNAVAILABLE"), "ok"])
        assert client.complete("s", "u") == "ok"
        assert len(fake.calls) == 2

    def test_gives_up_after_max_attempts(self) -> None:
        client, fake, _ = _client([_api_error(503, "UNAVAILABLE")], max_attempts=3)
        with pytest.raises(genai_errors.ServerError):
            client.complete("s", "u")
        assert len(fake.calls) == 3

    def test_auth_error_is_not_retried(self) -> None:
        client, fake, _ = _client([_api_error(403, "PERMISSION_DENIED")])
        with pytest.raises(genai_errors.ClientError):
            client.complete("s", "u")
        assert len(fake.calls) == 1

    def test_daily_quota_429_is_not_retried(self) -> None:
        exhausted = _api_error(
            429,
            "RESOURCE_EXHAUSTED",
            message="Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        )
        client, fake, _ = _client([exhausted])
        with pytest.raises(genai_errors.ClientError):
            client.complete("s", "u")
        assert len(fake.calls) == 1


class TestThrottle:
    def _throttled(self, rpm: int):
        now = [100.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        client = GeminiTextClient(
            api_key="fake",
            model="m",
            requests_per_minute=rpm,
            sleep_fn=sleep,
            clock_fn=lambda: now[0],
        )
        client._client = SimpleNamespace(models=_FakeGenaiModels(["ok"]))
        return client, sleeps

    def test_spaces_calls_to_the_configured_rate(self) -> None:
        client, sleeps = self._throttled(rpm=10)
        for _ in range(3):
            client.complete("s", "u")
        assert sleeps == [pytest.approx(6.0), pytest.approx(6.0)]

    def test_zero_disables_throttling(self) -> None:
        client, sleeps = self._throttled(rpm=0)
        for _ in range(3):
            client.complete("s", "u")
        assert sleeps == []
