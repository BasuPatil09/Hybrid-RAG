from __future__ import annotations

import threading
import time
from typing import Any, Callable, Protocol

from rag.retry_utils import call_with_transport_retry
from rag.tracing import span, usage_metadata


class TextClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class GeminiTextClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        thinking_level: str = "",
        requests_per_minute: int = 0,
        max_attempts: int = 4,
        timeout_seconds: float = 60.0,
        sleep_fn: Callable[[float], None] | None = None,
        clock_fn: Callable[[], float] | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client: Any = None
        self._model = model
        self._thinking_level = thinking_level
        self._max_attempts = max_attempts
        self._sleep = sleep_fn or time.sleep
        self._clock = clock_fn or time.monotonic
        self._min_interval = (
            60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        )
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        config = self._build_config(system_prompt)

        def call() -> str:
            self._throttle()
            with span(
                "gemini.generate_content",
                "llm",
                inputs={"system": system_prompt, "user": user_prompt},
                metadata={
                    "ls_provider": "google_genai",
                    "ls_model_name": self._model,
                    "ls_temperature": 0.0,
                },
            ) as run:
                response = self._get_client().models.generate_content(
                    model=self._model, contents=user_prompt, config=config
                )
                text = response.text or ""
                usage = getattr(response, "usage_metadata", None)
                output_tokens = (
                    getattr(usage, "candidates_token_count", None) or 0
                ) + (getattr(usage, "thoughts_token_count", None) or 0)
                run.end(
                    outputs={
                        "output": text,
                        "usage_metadata": usage_metadata(
                            getattr(usage, "prompt_token_count", None),
                            output_tokens if usage is not None else None,
                        ),
                    }
                )
            return text

        return call_with_transport_retry(
            call, max_attempts=self._max_attempts, sleep_fn=self._sleep
        )

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(
                        timeout=int(self._timeout_seconds * 1000)
                    ),
                )
            return self._client

    def _build_config(self, system_prompt: str):
        from google.genai import types

        thinking = (
            types.ThinkingConfig(thinking_level=self._thinking_level)
            if self._thinking_level
            else None
        )
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            thinking_config=thinking,
        )

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = self._clock()
            start = max(now, self._next_slot)
            self._next_slot = start + self._min_interval
        wait = start - now
        if wait > 0:
            self._sleep(wait)
