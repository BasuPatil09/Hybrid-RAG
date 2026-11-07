from __future__ import annotations

import logging

from rag.config import Settings
from rag.generation.confidence import CompletenessJudge, get_completeness_judge
from rag.generation.gemini_client import GeminiTextClient
from rag.generation.verification import CitationVerifier, get_citation_verifier

logger = logging.getLogger(__name__)


def missing_credentials_message(settings: Settings) -> str | None:
    if not settings.groq_api_key:
        return "GROQ_API_KEY is not configured on the server."
    if settings.judge_provider == "gemini" and not settings.gemini_api_key:
        return "GEMINI_API_KEY is not configured on the server (JUDGE_PROVIDER=gemini)."
    return None


def build_gemini_client(settings: Settings) -> GeminiTextClient:
    return GeminiTextClient(
        api_key=settings.gemini_api_key,
        model=settings.judge_model,
        thinking_level=settings.judge_thinking_level,
        requests_per_minute=settings.judge_requests_per_minute,
        max_attempts=settings.judge_max_attempts,
        timeout_seconds=settings.judge_timeout_seconds,
    )


def build_judges(settings: Settings) -> tuple[CitationVerifier, CompletenessJudge]:
    if (
        settings.judge_provider == settings.generation_provider
        and settings.judge_model == settings.generation_model
    ):
        logger.warning(
            "Judge model equals generator model (%s); the generator is grading "
            "its own output. Set JUDGE_PROVIDER/JUDGE_MODEL to a different model.",
            settings.judge_model,
        )

    if settings.judge_provider == "gemini":
        client = build_gemini_client(settings)
        return (
            get_citation_verifier(
                "gemini",
                api_key=settings.gemini_api_key,
                model=settings.judge_model,
                text_client=client,
            ),
            get_completeness_judge(
                "gemini",
                api_key=settings.gemini_api_key,
                model=settings.judge_model,
                text_client=client,
            ),
        )

    return (
        get_citation_verifier(
            "groq", api_key=settings.groq_api_key, model=settings.judge_model
        ),
        get_completeness_judge(
            "groq", api_key=settings.groq_api_key, model=settings.judge_model
        ),
    )
