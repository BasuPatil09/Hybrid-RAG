from __future__ import annotations
import math
import re
from abc import ABC, abstractmethod
from typing import Callable
from pydantic import BaseModel, computed_field
from rag.generation.generator import GeneratedAnswer
from rag.generation.text_utils import normalize_punctuation
from rag.generation.verification import VerifiedAnswer
from rag.retry_utils import call_with_transport_retry
from rag.schemas import ScoredChunk

_COMPLETENESS_PROMPT = """You are evaluating whether an ANSWER fully addresses a QUESTION.

Respond with EXACTLY one line in this format:
SCORE: <integer 0-100> - <one sentence explanation>

100 means the answer fully and directly addresses every part of the question. 0 means it does not address the question at all. Score based only on whether the ANSWER covers what the QUESTION asked — not writing quality, not whether you personally know a better answer.

If the ANSWER explicitly says the available context is insufficient to answer, score it 100. Correctly declining to guess is a complete, correct response to a question the source material can't support — not an incomplete one.
"""
_SCORE_PATTERN = re.compile(r"SCORE\s*:\s*(\d+)\s*-?\s*(.*)", re.IGNORECASE | re.DOTALL)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _parse_completeness(raw: str) -> tuple[float, str]:
    raw = normalize_punctuation(raw)
    match = _SCORE_PATTERN.search(raw.strip())
    if match:
        score = max(0, min(100, int(match.group(1))))
        explanation = match.group(2).strip()
        return score / 100.0, explanation or raw.strip()[:200]

    return (
        0.0,
        f"Judge response did not match the expected format: {raw.strip()[:200]!r}",
    )

class CompletenessJudge(ABC):
    @abstractmethod
    def judge(self, query: str, answer_text: str) -> tuple[float, str]:
        pass

class GroqCompletenessJudge(CompletenessJudge):
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

    def judge(self, query: str, answer_text: str) -> tuple[float, str]:
        def call() -> str:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _COMPLETENESS_PROMPT},
                    {
                        "role": "user",
                        "content": f"QUESTION: {query}\n\nANSWER: {answer_text}",
                    },
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content

        raw = call_with_transport_retry(
            call, max_attempts=self._max_attempts, sleep_fn=self._sleep
        )
        return _parse_completeness(raw)

def get_completeness_judge(
    provider: str, *, api_key: str, model: str
) -> CompletenessJudge:
    if provider == "groq":
        return GroqCompletenessJudge(api_key=api_key, model=model)
    raise ValueError(f"Unknown completeness judge provider: {provider!r}")

def compute_retrieval_confidence(
    cited_chunk_ids: list[str], reranked: list[ScoredChunk]
) -> float:

    scores_by_id = {sc.chunk.id: sc.score for sc in reranked}
    relevant_scores = [
        scores_by_id[cid] for cid in cited_chunk_ids if cid in scores_by_id
    ]
    if not relevant_scores:
        return 0.0
    return sum(_sigmoid(s) for s in relevant_scores) / len(relevant_scores)

class ConfidenceScore(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    completeness_explanation: str

    @computed_field
    @property
    def composite(self) -> float:

        return (
            self.retrieval_confidence
            + self.citation_coverage
            + self.answer_completeness
        ) / 3

def score_answer(
    query: str,
    answer: GeneratedAnswer,
    verified: VerifiedAnswer,
    reranked: list[ScoredChunk],
    completeness_judge: CompletenessJudge,
) -> ConfidenceScore:
    retrieval_confidence = compute_retrieval_confidence(
        answer.cited_chunk_ids, reranked
    )
    citation_coverage = verified.citation_accuracy
    answer_completeness, explanation = completeness_judge.judge(query, answer.answer)
    return ConfidenceScore(
        retrieval_confidence=retrieval_confidence,
        citation_coverage=citation_coverage,
        answer_completeness=answer_completeness,
        completeness_explanation=explanation,
    )
