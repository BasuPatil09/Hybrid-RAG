from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel

from rag.generation.gemini_client import GeminiTextClient, TextClient
from rag.generation.generator import GeneratedAnswer
from rag.generation.text_utils import normalize_punctuation, strip_markdown_emphasis
from rag.retry_utils import call_with_transport_retry
from rag.schemas import Chunk

_JUDGE_SYSTEM_PROMPT = """You are a fact-checking judge. You will be given a CLAIM and a SOURCE TEXT. Your only job is to decide whether the SOURCE TEXT actually supports the CLAIM.

Respond with EXACTLY one line, one of these two formats, nothing else:
SUPPORTED: <one sentence saying what in the source supports the claim>
NOT_SUPPORTED: <one sentence saying what is missing or contradicted>

Use only the SOURCE TEXT to judge — not outside knowledge, not what seems plausible. A claim is SUPPORTED only if the source text directly states or clearly implies it; topical relatedness alone is NOT support. The SOURCE TEXT may be a concatenation of several excerpts separated by "---" — treat it as one combined body of evidence: a fact may be stated in any one of the excerpts, or assembled from parts stated across more than one.

If the CLAIM contains multiple parts (for example, joined by "and"), every part must be directly supported by the SOURCE TEXT for the verdict to be SUPPORTED. If any part is missing from the source, the verdict is NOT_SUPPORTED even if the rest of the claim is correct — do not fill in a missing part with what is "commonly understood" or "usually true." That is outside knowledge, and using it here is exactly the failure mode this judgment exists to prevent.
"""


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CITATION_WITH_LEADING_SPACE_PATTERN = re.compile(r"\s*\[(\d+)\]")
_CHUNK_SEPARATOR = "\n\n---\n\n"
_TRAILING_CITATION_PATTERN = re.compile(r"([.!?])\s*((?:\[\d+\])+)")


class Claim(BaseModel):
    text: str
    chunks: list[Chunk]


class ClaimVerdict(BaseModel):
    claim: str
    chunk_ids: list[str]
    supported: bool
    explanation: str


class VerifiedAnswer(BaseModel):
    answer: GeneratedAnswer
    claim_verdicts: list[ClaimVerdict]

    @property
    def citation_accuracy(self) -> float:

        if not self.claim_verdicts:
            return 1.0
        return sum(1 for v in self.claim_verdicts if v.supported) / len(
            self.claim_verdicts
        )


def _combine_chunk_contents(chunks: list[Chunk]) -> str:

    if len(chunks) == 1:
        return chunks[0].content
    return _CHUNK_SEPARATOR.join(c.content for c in chunks)


def extract_claims(answer_text: str, chunks: list[Chunk]) -> list[Claim]:

    claims: list[Claim] = []
    answer_text = _TRAILING_CITATION_PATTERN.sub(r" \2\1", answer_text)
    for sentence in _SENTENCE_SPLIT_PATTERN.split(answer_text.strip()):
        indices = _CITATION_PATTERN.findall(sentence)
        if not indices:
            continue
        clean_text = _CITATION_WITH_LEADING_SPACE_PATTERN.sub("", sentence).strip()
        if not clean_text:
            continue
        cited_chunks = [
            chunks[int(index_str) - 1]
            for index_str in dict.fromkeys(indices)
            if 1 <= int(index_str) <= len(chunks)
        ]
        if cited_chunks:
            claims.append(Claim(text=clean_text, chunks=cited_chunks))
    return claims


_VERDICT_PATTERN = re.compile(
    r"\b(NOT_SUPPORTED|SUPPORTED)\s*:\s*(.*?)(?=\b(?:NOT_SUPPORTED|SUPPORTED)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_verdict(raw: str) -> tuple[bool, str]:

    raw = strip_markdown_emphasis(normalize_punctuation(raw))
    matches = list(_VERDICT_PATTERN.finditer(raw.strip()))
    if matches:
        last = matches[-1]
        explanation = last.group(2).strip()
        return last.group(1).upper() == "SUPPORTED", explanation or raw.strip()[:200]

    return (
        False,
        f"Judge response did not match the expected format: {raw.strip()[:200]!r}",
    )


class CitationVerifier(ABC):
    @abstractmethod
    def verify_claim(self, claim_text: str, source_text: str) -> tuple[bool, str]:
        pass


class GroqCitationVerifier(CitationVerifier):
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_attempts: int = 3,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model
        self._max_attempts = max_attempts
        self._sleep = sleep_fn

    def verify_claim(self, claim_text: str, source_text: str) -> tuple[bool, str]:
        def call() -> str:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"CLAIM: {claim_text}\n\nSOURCE TEXT: {source_text}",
                    },
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content

        raw = call_with_transport_retry(
            call, max_attempts=self._max_attempts, sleep_fn=self._sleep
        )
        return _parse_verdict(raw)


class GeminiCitationVerifier(CitationVerifier):
    def __init__(self, client: TextClient) -> None:
        self._client = client

    def verify_claim(self, claim_text: str, source_text: str) -> tuple[bool, str]:
        raw = self._client.complete(
            _JUDGE_SYSTEM_PROMPT,
            f"CLAIM: {claim_text}\n\nSOURCE TEXT: {source_text}",
        )
        return _parse_verdict(raw)


def get_citation_verifier(
    provider: str,
    *,
    api_key: str,
    model: str,
    text_client: TextClient | None = None,
) -> CitationVerifier:
    if provider == "groq":
        return GroqCitationVerifier(api_key=api_key, model=model)
    if provider == "gemini":
        client = text_client or GeminiTextClient(api_key=api_key, model=model)
        return GeminiCitationVerifier(client)
    raise ValueError(f"Unknown citation verification provider: {provider!r}")


def verify_answer(
    answer: GeneratedAnswer, chunks: list[Chunk], verifier: CitationVerifier
) -> VerifiedAnswer:

    if answer.is_insufficient:
        return VerifiedAnswer(answer=answer, claim_verdicts=[])

    claims = extract_claims(answer.answer, chunks)
    verdicts = []
    for claim in claims:
        combined_source = _combine_chunk_contents(claim.chunks)
        supported, explanation = verifier.verify_claim(claim.text, combined_source)
        verdicts.append(
            ClaimVerdict(
                claim=claim.text,
                chunk_ids=[c.id for c in claim.chunks],
                supported=supported,
                explanation=explanation,
            )
        )
    return VerifiedAnswer(answer=answer, claim_verdicts=verdicts)
