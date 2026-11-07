from __future__ import annotations

import httpx
import pytest
from google.genai import errors as genai_errors

from rag.retry_utils import (
    _gemini_wait_seconds,
    _is_retryable_gemini_error,
    call_with_transport_retry,
)


def _err(code: int, status: str = "X", message: str = "m", details=None):
    body = {"error": {"code": code, "message": message, "status": status}}
    if details is not None:
        body["error"]["details"] = details
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, body)


class TestGeminiErrorClassification:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_transient_codes_are_retryable(self, code: int) -> None:
        assert _is_retryable_gemini_error(_err(code)) is True

    @pytest.mark.parametrize("code", [400, 401, 403, 404])
    def test_client_errors_are_not_retryable(self, code: int) -> None:
        assert _is_retryable_gemini_error(_err(code)) is False

    def test_per_day_quota_is_not_retryable(self) -> None:
        exc = _err(429, "RESOURCE_EXHAUSTED", message="...PerDayPerProject...")
        assert _is_retryable_gemini_error(exc) is False


class TestGeminiWait:
    def test_reads_retry_delay_detail(self) -> None:
        exc = _err(429, details=[{"retryDelay": "34s"}])
        assert _gemini_wait_seconds(exc, default=5.0) == 34.0

    def test_reads_retry_in_from_message_text(self) -> None:
        exc = _err(429, message="Quota exceeded. Please retry in 7.5s.")
        assert _gemini_wait_seconds(exc, default=5.0) == 7.5

    def test_falls_back_to_default(self) -> None:
        assert _gemini_wait_seconds(_err(429), default=5.0) == 5.0


class TestCallWithTransportRetry:
    def test_wait_is_capped_so_a_request_thread_never_sleeps_for_minutes(self) -> None:
        sleeps: list[float] = []
        outcomes = [_err(429, details=[{"retryDelay": "900s"}]), "ok"]

        def fn():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        assert call_with_transport_retry(fn, sleep_fn=sleeps.append) == "ok"
        assert sleeps == [pytest.approx(65.25)]

    def test_existing_transport_retry_behaviour_is_unchanged(self) -> None:
        sleeps: list[float] = []
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.RemoteProtocolError("disconnected")
            return "ok"

        assert call_with_transport_retry(fn, sleep_fn=sleeps.append) == "ok"
        assert sleeps == [2.0]
