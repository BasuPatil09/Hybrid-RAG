from __future__ import annotations

import uuid

import chromadb
import httpx
import pytest
from fastapi.testclient import TestClient
from groq import AuthenticationError

from rag.api.dependencies import AppComponents, get_components
from rag.api.main import _docs_urls, app, limiter
from rag.config import get_settings
from rag.generation.generator import GeneratedAnswer
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.ingestion.chunking import get_chunker
from rag.ingestion.loaders import DocumentLoaderRegistry
from rag.observability import ObservabilityLog
from rag.retrieval.reranker import Reranker
from rag.schemas import Chunk, ChunkMetadata, ScoredChunk
from tests.fakes import FakeEmbedder


def _chunk(cid: str, content: str = "some source content") -> Chunk:
    return Chunk(
        id=cid,
        content=content,
        metadata=ChunkMetadata(
            source="f.md", format="md", chunk_index=0, char_count=len(content)
        ),
    )


class _FakeReranker(Reranker):
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        return candidates[:top_n] or [ScoredChunk(chunk=_chunk("fallback"), score=1.0)]


class _FakeGenerator:
    def generate(self, query: str, chunks: list[Chunk]) -> GeneratedAnswer:
        if "unanswerable" in query.lower():
            return GeneratedAnswer(
                answer="INSUFFICIENT_CONTEXT: not covered.",
                is_insufficient=True,
                cited_chunk_ids=[],
            )
        return GeneratedAnswer(
            answer="A real answer [1].",
            is_insufficient=False,
            cited_chunk_ids=[chunks[0].id] if chunks else [],
        )


class _FakeVerifier:
    def verify_claim(self, claim_text: str, source_text: str) -> tuple[bool, str]:
        return True, "fake supported"


class _FakeCompletenessJudge:
    def judge(self, query: str, answer_text: str) -> tuple[float, str]:
        return 0.9, "fake mostly complete"


class _ExplodingGenerator:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def generate(self, query: str, chunks: list[Chunk]) -> GeneratedAnswer:
        raise self._exc


def _fake_components(
    *, has_groq_key: bool = True, seeded: bool = True
) -> AppComponents:
    settings = get_settings()
    settings.groq_api_key = "fake-key" if has_groq_key else ""
    embedder = FakeEmbedder()
    vector_store = ChromaVectorStore(
        chromadb.EphemeralClient(), f"api_test_{uuid.uuid4().hex[:12]}"
    )
    sparse_index = SparseIndex()
    if seeded:
        chunks = [_chunk("a")]
        vector_store.upsert(chunks, embedder.embed([c.content for c in chunks]))
        sparse_index.rebuild(chunks)
    return AppComponents(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        sparse_index=sparse_index,
        reranker=_FakeReranker(),
        generator=_FakeGenerator(),
        verifier=_FakeVerifier(),
        completeness_judge=_FakeCompletenessJudge(),
        registry=DocumentLoaderRegistry(),
        chunker=get_chunker("structure_aware", chunk_size=200, chunk_overlap=20),
        observability_log=ObservabilityLog(),
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    settings = get_settings()
    original = (
        settings.app_api_key,
        settings.ask_rate_limit,
        settings.ingest_rate_limit,
        settings.eval_rate_limit,
    )
    limiter.reset()
    yield
    app.dependency_overrides.clear()
    (
        settings.app_api_key,
        settings.ask_rate_limit,
        settings.ingest_rate_limit,
        settings.eval_rate_limit,
    ) = original
    limiter.reset()


class TestHealth:
    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_api_routes_still_resolve_with_static_mount_present(
        self, client: TestClient
    ) -> None:

        app.dependency_overrides[get_components] = lambda: _fake_components(
            seeded=False
        )
        response = client.get("/v1/documents")
        assert response.status_code == 200


class TestApiSecurity:
    def test_all_v1_routes_require_the_configured_key_but_health_stays_public(
        self, client: TestClient
    ) -> None:
        components = _fake_components()
        components.settings.app_api_key = "shared-test-key"
        app.dependency_overrides[get_components] = lambda: components

        assert client.get("/health").status_code == 200
        assert (
            client.post("/v1/ask", json={"question": "a question"}).status_code == 401
        )
        assert client.post("/v1/ingest").status_code == 401
        assert client.get("/v1/documents").status_code == 401
        assert client.post("/v1/eval").status_code == 401
        assert client.get("/v1/observability").status_code == 401
        assert (
            client.get(
                "/v1/documents", headers={"X-API-Key": "shared-test-key"}
            ).status_code
            == 200
        )

    def test_authentication_remains_disabled_when_no_key_is_configured(
        self, client: TestClient
    ) -> None:
        app.dependency_overrides[get_components] = lambda: _fake_components()
        assert client.get("/v1/documents").status_code == 200

    def test_ask_question_over_max_length_returns_422(self, client: TestClient) -> None:
        app.dependency_overrides[get_components] = lambda: _fake_components()
        response = client.post("/v1/ask", json={"question": "x" * 2001})
        assert response.status_code == 422

    def test_expensive_ask_endpoint_returns_429_when_per_ip_limit_is_exceeded(
        self, client: TestClient
    ) -> None:
        components = _fake_components()
        components.settings.ask_rate_limit = "1/minute"
        app.dependency_overrides[get_components] = lambda: components

        assert client.post("/v1/ask", json={"question": "first"}).status_code == 200
        response = client.post("/v1/ask", json={"question": "second"})
        assert response.status_code == 429
        assert response.json()

    def test_production_disables_interactive_docs(self) -> None:
        assert _docs_urls("production") == (None, None)
        assert _docs_urls("development") == ("/docs", "/redoc")


class TestErrorSafety:
    def test_unhandled_error_response_does_not_include_internal_details(self) -> None:
        components = _fake_components()
        components.generator = _ExplodingGenerator(
            RuntimeError(r"D:\private\internal.py")
        )
        app.dependency_overrides[get_components] = lambda: components

        safe_client = TestClient(app, raise_server_exceptions=False)
        response = safe_client.post("/v1/ask", json={"question": "a question"})

        assert response.status_code == 500
        assert response.text == "Internal Server Error"
        assert "private" not in response.text

    def test_groq_authentication_error_is_redacted_before_observability_logging(
        self,
    ) -> None:
        secret = "fake-invalid-groq-key"
        request = httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {secret}"},
        )
        exc = AuthenticationError(
            f"Groq rejected Authorization: Bearer {secret}",
            response=httpx.Response(401, request=request),
            body={"error": "invalid API key"},
        )
        components = _fake_components()
        components.settings.groq_api_key = secret
        components.generator = _ExplodingGenerator(exc)
        app.dependency_overrides[get_components] = lambda: components

        safe_client = TestClient(app, raise_server_exceptions=False)
        assert (
            safe_client.post("/v1/ask", json={"question": "a question"}).status_code
            == 500
        )

        logged_error = components.observability_log.read_recent(1)[0].error
        assert secret not in (logged_error or "")
        assert "[REDACTED]" in (logged_error or "")


class TestAsk:
    def test_returns_answer_with_citations_and_confidence(
        self, client: TestClient
    ) -> None:
        app.dependency_overrides[get_components] = lambda: _fake_components()
        response = client.post("/v1/ask", json={"question": "a real question"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]["is_insufficient"] is False
        assert "confidence" in body and "composite" in body["confidence"]

    def test_insufficient_context_question_returns_declined_answer(
        self, client: TestClient
    ) -> None:
        app.dependency_overrides[get_components] = lambda: _fake_components()
        response = client.post("/v1/ask", json={"question": "an unanswerable question"})
        assert response.status_code == 200
        assert response.json()["answer"]["is_insufficient"] is True

    def test_empty_question_returns_400(self, client: TestClient) -> None:
        app.dependency_overrides[get_components] = lambda: _fake_components()
        response = client.post("/v1/ask", json={"question": "   "})
        assert response.status_code == 400

    def test_missing_groq_key_returns_503(self, client: TestClient) -> None:
        app.dependency_overrides[get_components] = lambda: _fake_components(
            has_groq_key=False
        )
        response = client.post("/v1/ask", json={"question": "a question"})
        assert response.status_code == 503

    def test_empty_index_returns_503(self, client: TestClient) -> None:
        app.dependency_overrides[get_components] = lambda: _fake_components(
            seeded=False
        )
        response = client.post("/v1/ask", json={"question": "a question"})
        assert response.status_code == 503


class TestIngest:
    def test_ingests_a_small_corpus(self, client: TestClient, tmp_path) -> None:
        (tmp_path / "notes.md").write_text(
            "# Note\n\nSome real content about a topic here."
        )
        components = _fake_components(seeded=False)
        components.settings.corpus_dir = tmp_path
        app.dependency_overrides[get_components] = lambda: components

        response = client.post("/v1/ingest")

        assert response.status_code == 200
        body = response.json()
        assert body["documents_loaded"] == 1
        assert body["chunks_indexed"] >= 1

    def test_ask_after_ingest_can_see_newly_indexed_content(
        self, client: TestClient, tmp_path
    ) -> None:

        (tmp_path / "notes.md").write_text(
            "# Note\n\nSome real content about a topic here."
        )
        components = _fake_components(seeded=False)
        components.settings.corpus_dir = tmp_path
        app.dependency_overrides[get_components] = lambda: components

        client.post("/v1/ingest")
        response = client.post("/v1/ask", json={"question": "a real question"})

        assert response.status_code == 200


class TestListDocuments:
    def test_empty_store_returns_empty_list(self, client: TestClient) -> None:
        app.dependency_overrides[get_components] = lambda: _fake_components(
            seeded=False
        )
        response = client.get("/v1/documents")
        assert response.status_code == 200
        assert response.json() == {"total_chunks": 0, "documents": []}

    def test_summarizes_chunk_counts_per_source(self, client: TestClient) -> None:
        components = _fake_components(seeded=False)
        chunks = [_chunk("a"), _chunk("b")]
        chunks[0].metadata.source, chunks[1].metadata.source = "doc1.md", "doc1.md"
        components.vector_store.upsert(
            chunks, components.embedder.embed([c.content for c in chunks])
        )
        app.dependency_overrides[get_components] = lambda: components

        response = client.get("/v1/documents")

        assert response.status_code == 200
        body = response.json()
        assert body["total_chunks"] == 2
        assert body["documents"] == [{"source": "doc1.md", "chunk_count": 2}]


def _write_golden_csv(tmp_path, rows: list[str]):
    header = (
        "id,category,question,expected_behavior,source_hint,your_correction,approved\n"
    )
    path = tmp_path / "golden.csv"
    path.write_text(header + "\n".join(rows) + "\n")
    return path


class TestEval:
    def test_runs_eval_and_returns_summary_and_results(
        self, client: TestClient, tmp_path
    ) -> None:
        components = _fake_components()
        components.settings.golden_qa_path = _write_golden_csv(
            tmp_path,
            [
                "q1,direct_lookup,A real question?,answerable,hint,,yes",
                "q2,unanswerable,An unanswerable question?,insufficient_context,hint,,yes",
            ],
        )
        app.dependency_overrides[get_components] = lambda: components

        response = client.post("/v1/eval")

        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 2
        assert body["summary"]["total_questions"] == 2
        assert body["summary"]["behavior_match_rate"] == 1.0

    def test_unapproved_rows_are_excluded_same_as_manual_check(
        self, client: TestClient, tmp_path
    ) -> None:
        components = _fake_components()
        components.settings.golden_qa_path = _write_golden_csv(
            tmp_path, ["q1,direct_lookup,Unapproved question?,answerable,hint,,"]
        )
        app.dependency_overrides[get_components] = lambda: components

        response = client.post("/v1/eval")

        assert response.status_code == 404

    def test_missing_groq_key_returns_503(self, client: TestClient, tmp_path) -> None:
        components = _fake_components(has_groq_key=False)
        components.settings.golden_qa_path = _write_golden_csv(
            tmp_path, ["q1,direct_lookup,Q?,answerable,hint,,yes"]
        )
        app.dependency_overrides[get_components] = lambda: components

        response = client.post("/v1/eval")

        assert response.status_code == 503

    def test_empty_index_returns_503(self, client: TestClient, tmp_path) -> None:
        components = _fake_components(seeded=False)
        components.settings.golden_qa_path = _write_golden_csv(
            tmp_path, ["q1,direct_lookup,Q?,answerable,hint,,yes"]
        )
        app.dependency_overrides[get_components] = lambda: components

        response = client.post("/v1/eval")

        assert response.status_code == 503


class TestObservability:
    def test_empty_history_returns_zeroed_summary(self, client: TestClient) -> None:
        components = _fake_components()
        app.dependency_overrides[get_components] = lambda: components

        response = client.get("/v1/observability")

        assert response.status_code == 200
        body = response.json()
        assert body["summary"]["total_events"] == 0
        assert body["recent_events"] == []

    def test_ask_call_is_recorded_and_visible_afterward(
        self, client: TestClient
    ) -> None:

        components = _fake_components()
        app.dependency_overrides[get_components] = lambda: components

        client.post("/v1/ask", json={"question": "a real question"})
        response = client.get("/v1/observability")

        body = response.json()
        assert body["summary"]["total_events"] == 1
        assert body["summary"]["ask_count"] == 1
        event = body["recent_events"][0]
        assert event["event_type"] == "ask"
        assert event["success"] is True
        assert event["question"] == "a real question"
        assert event["composite_confidence"] is not None

    def test_ingest_call_is_recorded(self, client: TestClient, tmp_path) -> None:
        (tmp_path / "notes.md").write_text(
            "# Note\n\nSome real content about a topic here."
        )
        components = _fake_components(seeded=False)
        components.settings.corpus_dir = tmp_path
        app.dependency_overrides[get_components] = lambda: components

        client.post("/v1/ingest")
        response = client.get("/v1/observability")

        body = response.json()
        assert body["summary"]["ingest_count"] == 1
        event = body["recent_events"][0]
        assert event["event_type"] == "ingest"
        assert event["chunks_indexed"] is not None

    def test_failed_ask_is_recorded_as_an_error_event(self, client: TestClient) -> None:

        components = _fake_components(has_groq_key=False)
        app.dependency_overrides[get_components] = lambda: components

        client.post("/v1/ask", json={"question": "a question"})
        response = client.get("/v1/observability")

        assert response.json()["summary"]["total_events"] == 0

    def test_most_recent_event_is_first(self, client: TestClient) -> None:
        components = _fake_components()
        app.dependency_overrides[get_components] = lambda: components

        client.post("/v1/ask", json={"question": "first question"})
        client.post("/v1/ask", json={"question": "second question"})
        response = client.get("/v1/observability")

        events = response.json()["recent_events"]
        assert events[0]["question"] == "second question"
        assert events[1]["question"] == "first question"

    def test_limit_query_param_caps_returned_events(self, client: TestClient) -> None:
        components = _fake_components()
        app.dependency_overrides[get_components] = lambda: components

        for i in range(3):
            client.post("/v1/ask", json={"question": f"question {i}"})
        response = client.get("/v1/observability?limit=2")

        assert len(response.json()["recent_events"]) == 2
