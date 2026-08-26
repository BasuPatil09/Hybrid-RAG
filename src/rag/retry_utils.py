from __future__ import annotations
import re
import time
from typing import Callable, TypeVar
import httpx
from groq import RateLimitError

T = TypeVar("T")
_RETRY_AFTER_PATTERN = re.compile(r"try again in ([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

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
    assert last_exc is not None
    raise last_exc
