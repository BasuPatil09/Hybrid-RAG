from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

import httpx
from google.genai import errors as genai_errors
from groq import RateLimitError

T = TypeVar("T")

_RETRY_AFTER_PATTERN = re.compile(r"try again in ([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

_GEMINI_RETRY_DELAY_PATTERN = re.compile(
    r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s|retry in ([\d.]+)s", re.IGNORECASE
)
_RETRYABLE_GEMINI_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 65.0


def _rate_limit_wait_seconds(exc: RateLimitError, default: float) -> float:
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _RETRY_AFTER_PATTERN.search(str(exc))
    if match:
        value, unit = match.groups()
        return float(value) / 1000.0 if unit.lower() == "ms" else float(value)
    return default


def _is_retryable_gemini_error(exc: genai_errors.APIError) -> bool:
    if exc.code not in _RETRYABLE_GEMINI_CODES:
        return False
    return "perday" not in str(exc).lower().replace(" ", "").replace("_", "")


def _gemini_wait_seconds(exc: genai_errors.APIError, default: float) -> float:
    match = _GEMINI_RETRY_DELAY_PATTERN.search(str(exc))
    if match:
        return float(match.group(1) or match.group(2))
    return default


def call_with_transport_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    sleep_fn: Callable[[float], None] | None = None,
    rate_limit_default_wait: float = 5.0,
) -> T:
    sleep = sleep_fn or time.sleep
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                sleep(2.0 * (attempt + 1))
        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                sleep(_rate_limit_wait_seconds(exc, rate_limit_default_wait) + 0.25)
        except genai_errors.APIError as exc:
            if not _is_retryable_gemini_error(exc):
                raise
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = _gemini_wait_seconds(
                    exc, rate_limit_default_wait * (attempt + 1)
                )
                sleep(min(wait, _MAX_BACKOFF_SECONDS) + 0.25)
    assert last_exc is not None
    raise last_exc
