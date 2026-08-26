from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

logger = logging.getLogger("rag.observability")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)\S+"),
    re.compile(r"(?i)((?:x-api-key|api[_-]?key|groq_api_key)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)(bearer\s+)\S+"),
)


def redact_error(exc: Exception, *known_secrets: str) -> str:

    message = str(exc)
    for secret in known_secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        message = pattern.sub(r"\1[REDACTED]", message)
    return f"{type(exc).__name__}: {message}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ObservabilityEvent(BaseModel):
    timestamp: str
    event_type: Literal["ask", "ingest", "eval"]
    latency_ms: float
    success: bool
    error: str | None = None
    question: str | None = None
    is_insufficient: bool | None = None
    retrieval_confidence: float | None = None
    citation_coverage: float | None = None
    answer_completeness: float | None = None
    composite_confidence: float | None = None
    claims_verified: int | None = None
    claims_supported: int | None = None
    chunks_indexed: int | None = None
    eval_questions: int | None = None
    eval_behavior_match_rate: float | None = None


class ObservabilitySummary(BaseModel):
    total_events: int
    ask_count: int
    ingest_count: int
    eval_count: int
    error_count: int
    mean_latency_ms: float
    mean_composite_confidence: float
    insufficient_context_rate: float


class ObservabilityLog:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._events: list[ObservabilityEvent] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._events = self._read_all_from_disk()

    def record(self, event: ObservabilityEvent) -> None:
        logger.info(event.model_dump_json())
        self._events.append(event)
        if self._path is not None:
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(event.model_dump_json() + "\n")
            except OSError as exc:
                logger.warning(
                    "Failed to persist observability event to %s: %s", self._path, exc
                )

    def read_recent(self, limit: int = 200) -> list[ObservabilityEvent]:
        return self._events[-limit:]

    def _read_all_from_disk(self) -> list[ObservabilityEvent]:
        if not self._path or not self._path.exists():
            return []
        events: list[ObservabilityEvent] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(ObservabilityEvent.model_validate_json(line))
        return events


def summarize(events: list[ObservabilityEvent]) -> ObservabilitySummary:
    if not events:
        return ObservabilitySummary(
            total_events=0,
            ask_count=0,
            ingest_count=0,
            eval_count=0,
            error_count=0,
            mean_latency_ms=0.0,
            mean_composite_confidence=0.0,
            insufficient_context_rate=0.0,
        )

    successful_asks = [e for e in events if e.event_type == "ask" and e.success]
    composite_scores = [
        e.composite_confidence
        for e in successful_asks
        if e.composite_confidence is not None
    ]
    insufficient_flags = [
        e.is_insufficient for e in successful_asks if e.is_insufficient is not None
    ]

    return ObservabilitySummary(
        total_events=len(events),
        ask_count=sum(1 for e in events if e.event_type == "ask"),
        ingest_count=sum(1 for e in events if e.event_type == "ingest"),
        eval_count=sum(1 for e in events if e.event_type == "eval"),
        error_count=sum(1 for e in events if not e.success),
        mean_latency_ms=sum(e.latency_ms for e in events) / len(events),
        mean_composite_confidence=(sum(composite_scores) / len(composite_scores))
        if composite_scores
        else 0.0,
        insufficient_context_rate=(sum(insufficient_flags) / len(insufficient_flags))
        if insufficient_flags
        else 0.0,
    )
