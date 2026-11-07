from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Any, ContextManager, Protocol, TypeVar

from rag.config import Settings

try:
    from langsmith import Client as _LangSmithClient
    from langsmith import run_helpers as _rh
    from langsmith.utils import tracing_is_enabled as _tracing_is_enabled
except ImportError:  # pragma: no cover
    _LangSmithClient = None
    _rh = None

    def _tracing_is_enabled() -> bool:
        return False


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def traced(
    name: str,
    *,
    run_type: str = "chain",
    process_inputs: Callable[[dict], dict] | None = None,
    process_outputs: Callable[[Any], dict] | None = None,
) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        if _rh is None:  # pragma: no cover
            return fn
        options: dict[str, Any] = {"name": name, "run_type": run_type}
        if process_inputs is not None:
            options["process_inputs"] = process_inputs
        if process_outputs is not None:
            options["process_outputs"] = process_outputs
        return _rh.traceable(**options)(fn)

    return decorator


class _NullSpan:
    def end(self, **_: Any) -> None:
        return None


@contextlib.contextmanager
def span(
    name: str,
    run_type: str = "chain",
    *,
    inputs: dict | None = None,
    metadata: dict | None = None,
) -> Iterator[Any]:
    if _rh is None or not _tracing_is_enabled():
        yield _NullSpan()
        return
    with _rh.trace(
        name, run_type, inputs=inputs or {}, metadata=metadata
    ) as run:  # pragma: no cover - exercised via the recording-client tests
        yield run


def current_trace_id() -> str | None:
    if _rh is None:
        return None
    run = _rh.get_current_run_tree()
    return str(run.trace_id) if run is not None else None


def usage_metadata(
    input_tokens: int | None, output_tokens: int | None
) -> dict[str, int]:
    if input_tokens is None and output_tokens is None:
        return {}
    i, o = input_tokens or 0, output_tokens or 0
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def query_only_inputs(inputs: dict) -> dict:
    return {"query": inputs.get("query")}


def scored_chunks_outputs(result: Any) -> dict:
    return {
        "documents": [
            {
                "page_content": sc.chunk.content,
                "type": "Document",
                "metadata": {
                    "chunk_id": sc.chunk.id,
                    "score": sc.score,
                    "source": sc.chunk.metadata.source,
                    "page": sc.chunk.metadata.page,
                },
            }
            for sc in result
        ]
    }


class Tracer(Protocol):
    enabled: bool

    def scope(
        self, *, tags: list[str] | None = None, metadata: dict | None = None
    ) -> ContextManager[None]: ...

    def log_feedback(
        self, trace_id: str, key: str, score: float, comment: str | None = None
    ) -> None: ...

    def flush(self) -> None: ...


class NullTracer:
    enabled = False

    def scope(
        self, *, tags: list[str] | None = None, metadata: dict | None = None
    ) -> ContextManager[None]:
        return contextlib.nullcontext()

    def log_feedback(
        self, trace_id: str, key: str, score: float, comment: str | None = None
    ) -> None:
        return None

    def flush(self) -> None:
        return None


class LangSmithTracer:
    enabled = True

    def __init__(self, client: Any, project: str) -> None:
        self._client = client
        self._project = project

    def scope(
        self, *, tags: list[str] | None = None, metadata: dict | None = None
    ) -> ContextManager[None]:
        return _rh.tracing_context(
            enabled=True,
            client=self._client,
            project_name=self._project,
            tags=tags,
            metadata=metadata,
        )

    def log_feedback(
        self, trace_id: str, key: str, score: float, comment: str | None = None
    ) -> None:
        try:
            self._client.create_feedback(
                run_id=trace_id, key=key, score=score, comment=comment
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangSmith feedback %r not recorded: %s", key, exc)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangSmith flush failed: %s", exc)


def build_tracer(settings: Settings) -> Tracer:
    if not settings.tracing_active:
        return NullTracer()
    if _LangSmithClient is None:  # pragma: no cover
        logger.warning("LANGSMITH_TRACING is on but `langsmith` is not installed.")
        return NullTracer()
    hide = not settings.trace_include_content
    client = _LangSmithClient(
        api_key=settings.langsmith_api_key,
        api_url=settings.langsmith_endpoint or None,
        hide_inputs=hide,
        hide_outputs=hide,
    )
    return LangSmithTracer(client, settings.langsmith_project)
