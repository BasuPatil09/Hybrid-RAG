from __future__ import annotations

from types import SimpleNamespace

import pytest
from langsmith import Client

from rag.config import Settings
from rag.generation.gemini_client import GeminiTextClient
from rag.generation.generator import GroqGenerator
from rag.pipeline import answer_question
from rag.schemas import Chunk, ChunkMetadata
from rag.tracing import (
    LangSmithTracer,
    NullTracer,
    build_tracer,
    current_trace_id,
    span,
    usage_metadata,
)
from tests.pipeline_fakes import CORPUS, RETRIEVAL_KWARGS, make_components


class RecordingClient(Client):
    def __init__(self) -> None:
        super().__init__(
            api_url="http://localhost:1", api_key="k", auto_batch_tracing=False
        )
        self.recorded_runs: list[dict] = []
        self.recorded_updates: dict[str, dict] = {}
        self.recorded_feedback: list[dict] = []

    def create_run(self, name, inputs, run_type, **kwargs):
        self.recorded_runs.append(
            {
                "name": name,
                "inputs": inputs,
                "run_type": run_type,
                "id": str(kwargs.get("id")),
                "parent": str(kwargs.get("parent_run_id")),
                "tags": kwargs.get("tags"),
                "metadata": (kwargs.get("extra") or {}).get("metadata", {}),
            }
        )

    def update_run(self, run_id, **kwargs):
        self.recorded_updates[str(run_id)] = kwargs

    def create_feedback(self, **kwargs):
        self.recorded_feedback.append(kwargs)

    def by_name(self, name: str) -> dict:
        return next(r for r in self.recorded_runs if r["name"] == name)


def _ask(components, question="gradient descent"):
    return answer_question(
        question,
        components["embedder"],
        components["vector_store"],
        components["sparse_index"],
        components["reranker"],
        components["generator"],
        components["verifier"],
        components["completeness_judge"],
        **RETRIEVAL_KWARGS,
    )


class TestPipelineTracing:
    def test_no_scope_means_no_trace_and_no_id(self) -> None:
        assert _ask(make_components()).trace_id is None

    def test_scope_produces_a_root_span_with_a_child_per_stage(self) -> None:
        client = RecordingClient()
        tracer = LangSmithTracer(client, "proj")
        with tracer.scope(tags=["eval"], metadata={"question_id": "q1"}):
            result = _ask(make_components())

        names = [r["name"] for r in client.recorded_runs]
        for expected in (
            "rag.answer_question",
            "retrieve",
            "dense_retrieve",
            "sparse_retrieve",
            "rerank",
            "generate",
            "verify_citations",
            "score_confidence",
        ):
            assert expected in names, expected

        root = client.by_name("rag.answer_question")
        assert root["parent"] == "None"
        assert root["tags"] == ["eval"]
        assert root["metadata"]["question_id"] == "q1"
        assert result.trace_id == root["id"]
        for child in ("retrieve", "rerank", "generate", "verify_citations"):
            assert client.by_name(child)["parent"] == root["id"], child

    def test_retriever_spans_use_the_retriever_run_type(self) -> None:
        client = RecordingClient()
        with LangSmithTracer(client, "p").scope():
            _ask(make_components())
        assert client.by_name("retrieve")["run_type"] == "retriever"
        assert client.by_name("dense_retrieve")["run_type"] == "retriever"

    def test_inputs_never_include_stores_models_or_clients(self) -> None:
        client = RecordingClient()
        with LangSmithTracer(client, "p").scope():
            _ask(make_components(), "gradient descent")
        for run in client.recorded_runs:
            blob = repr(run["inputs"])
            assert "ChromaVectorStore" not in blob and "FakeEmbedder" not in blob, run
        assert client.by_name("rag.answer_question")["inputs"] == {
            "query": "gradient descent"
        }

    def test_retrieval_output_is_rendered_as_documents_with_scores(self) -> None:
        client = RecordingClient()
        with LangSmithTracer(client, "p").scope():
            _ask(make_components())
        out = client.recorded_updates[client.by_name("retrieve")["id"]]["outputs"]
        doc = out["documents"][0]
        assert doc["type"] == "Document" and "chunk_id" in doc["metadata"]
        assert doc["page_content"] in CORPUS.values()

    def test_citation_verdicts_are_visible_in_the_verify_span(self) -> None:
        client = RecordingClient()
        with LangSmithTracer(client, "p").scope():
            _ask(make_components())
        out = client.recorded_updates[client.by_name("verify_citations")["id"]][
            "outputs"
        ]
        assert out["citation_accuracy"] == 1.0
        assert out["claims"][0]["supported"] is True


class TestLlmSpans:
    def test_groq_call_records_provider_model_and_token_usage(self) -> None:
        generator = GroqGenerator(api_key="fake", sleep_fn=lambda _: None)
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=4)
        message = SimpleNamespace(content="Answer [1].")
        generator._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        choices=[SimpleNamespace(message=message)], usage=usage
                    )
                )
            )
        )
        client = RecordingClient()
        chunk = Chunk(
            id="a",
            content="x",
            metadata=ChunkMetadata(
                source="f", format="md", chunk_index=0, char_count=1
            ),
        )
        with LangSmithTracer(client, "p").scope():
            generator.generate("q", [chunk])

        run = client.by_name("groq.chat")
        assert run["run_type"] == "llm"
        assert run["metadata"]["ls_provider"] == "groq"
        assert run["metadata"]["ls_model_name"] == generator._model
        assert client.recorded_updates[run["id"]]["outputs"]["usage_metadata"] == {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
        }

    def test_gemini_call_counts_thinking_tokens_as_output(self) -> None:
        gemini = GeminiTextClient(
            api_key="fake", model="gemini-test", sleep_fn=lambda _: None
        )
        usage = SimpleNamespace(
            prompt_token_count=10, candidates_token_count=3, thoughts_token_count=20
        )
        gemini._client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kw: SimpleNamespace(
                    text="SUPPORTED: ok", usage_metadata=usage
                )
            )
        )
        client = RecordingClient()
        with LangSmithTracer(client, "p").scope():
            assert gemini.complete("sys", "user") == "SUPPORTED: ok"

        run = client.by_name("gemini.generate_content")
        assert run["metadata"]["ls_provider"] == "google_genai"
        assert client.recorded_updates[run["id"]]["outputs"]["usage_metadata"] == {
            "input_tokens": 10,
            "output_tokens": 23,
            "total_tokens": 33,
        }

    def test_each_retry_attempt_gets_its_own_span(self) -> None:
        import httpx

        gemini = GeminiTextClient(
            api_key="fake", model="m", sleep_fn=lambda _: None, max_attempts=3
        )
        calls = {"n": 0}

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.RemoteProtocolError("dropped")
            return SimpleNamespace(text="ok")

        gemini._client = SimpleNamespace(models=SimpleNamespace(generate_content=flaky))
        client = RecordingClient()
        with LangSmithTracer(client, "p").scope():
            gemini.complete("s", "u")
        assert [r["name"] for r in client.recorded_runs].count(
            "gemini.generate_content"
        ) == 3


class TestTracers:
    def test_null_tracer_is_inert(self) -> None:
        tracer = NullTracer()
        with tracer.scope(tags=["x"]):
            pass
        tracer.log_feedback("id", "k", 1.0)
        tracer.flush()
        assert tracer.enabled is False

    def test_span_without_scope_is_a_noop_handle(self) -> None:
        with span("anything", "llm", inputs={"a": 1}) as handle:
            handle.end(outputs={"b": 2})
        assert current_trace_id() is None

    def test_feedback_is_forwarded_to_the_client(self) -> None:
        client = RecordingClient()
        LangSmithTracer(client, "p").log_feedback("run-1", "recall@5", 0.5, "why")
        assert client.recorded_feedback == [
            {"run_id": "run-1", "key": "recall@5", "score": 0.5, "comment": "why"}
        ]

    def test_feedback_failure_never_raises(self) -> None:
        class Broken(RecordingClient):
            def create_feedback(self, **kw):
                raise RuntimeError("langsmith is down")

        LangSmithTracer(Broken(), "p").log_feedback("r", "k", 1.0)

    def test_flush_failure_never_raises(self) -> None:
        class Broken(RecordingClient):
            def flush(self):
                raise RuntimeError("down")

        LangSmithTracer(Broken(), "p").flush()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"langsmith_tracing": False, "langsmith_api_key": "k"},
            {"langsmith_tracing": True, "langsmith_api_key": ""},
        ],
    )
    def test_tracing_needs_both_the_flag_and_a_key(self, kwargs) -> None:
        settings = Settings(_env_file=None, **kwargs)
        assert settings.tracing_active is False
        assert isinstance(build_tracer(settings), NullTracer)

    def test_enabled_settings_build_a_langsmith_tracer(self, monkeypatch) -> None:
        created: list[dict] = []

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                created.append(kwargs)

        monkeypatch.setattr("rag.tracing._LangSmithClient", FakeClient)
        settings = Settings(
            _env_file=None,
            langsmith_tracing=True,
            langsmith_api_key="lsv2_fake",
            langsmith_endpoint="https://eu.api.smith.langchain.com",
        )
        tracer = build_tracer(settings)

        assert isinstance(tracer, LangSmithTracer)
        assert created[0]["api_key"] == "lsv2_fake"
        assert created[0]["api_url"] == "https://eu.api.smith.langchain.com"
        assert created[0]["hide_inputs"] is False

    def test_blank_endpoint_falls_back_to_the_sdk_default(self, monkeypatch) -> None:
        created: list[dict] = []
        monkeypatch.setattr(
            "rag.tracing._LangSmithClient", lambda **kw: created.append(kw) or object()
        )
        build_tracer(
            Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="k")
        )
        assert created[0]["api_url"] is None

    def test_trace_content_flag_maps_to_client_hide_flags(self, monkeypatch) -> None:
        created: list[dict] = []
        monkeypatch.setattr(
            "rag.tracing._LangSmithClient", lambda **kw: created.append(kw) or object()
        )
        build_tracer(
            Settings(
                _env_file=None,
                langsmith_tracing=True,
                langsmith_api_key="k",
                trace_include_content=False,
            )
        )
        assert created[0]["hide_inputs"] is True
        assert created[0]["hide_outputs"] is True


def test_usage_metadata_shapes() -> None:
    assert usage_metadata(None, None) == {}
    assert usage_metadata(5, None) == {
        "input_tokens": 5,
        "output_tokens": 0,
        "total_tokens": 5,
    }
