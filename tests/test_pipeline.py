from __future__ import annotations

import uuid
from pathlib import Path

import chromadb

from rag.indexing.sparse_index import SparseIndex
from rag.indexing.vector_store import ChromaVectorStore
from rag.ingestion.chunking import get_chunker
from rag.ingestion.loaders import DocumentLoaderRegistry
from rag.pipeline import retrieve, run_ingestion
from tests.fakes import FailAfterNCallsEmbedder, FakeEmbedder


def _new_pipeline_components(collection_name: str | None = None):
    name = collection_name or f"test_{uuid.uuid4().hex[:12]}"
    return {
        "registry": DocumentLoaderRegistry(),
        "chunker": get_chunker("structure_aware", chunk_size=200, chunk_overlap=20),
        "embedder": FakeEmbedder(),
        "vector_store": ChromaVectorStore(chromadb.EphemeralClient(), name),
        "sparse_index": SparseIndex(),
    }


class TestRunIngestion:
    def test_ingests_a_small_corpus_end_to_end(self, tmp_path: Path) -> None:
        (tmp_path / "notes.md").write_text(
            "# Gradient Descent\n\nGradient descent iteratively updates parameters "
            "to minimize a loss function using the negative gradient direction."
        )
        components = _new_pipeline_components("ingest_e2e_test")

        report = run_ingestion(tmp_path, dedup_threshold=0.95, **components)

        assert report.documents_loaded == 1
        assert report.chunks_created >= 1
        assert report.chunks_indexed == report.chunks_created
        assert report.duplicates_skipped == 0
        assert components["vector_store"].count() == report.chunks_indexed

    def test_empty_corpus_produces_zero_chunk_report(self, tmp_path: Path) -> None:
        components = _new_pipeline_components("ingest_empty_test")
        report = run_ingestion(tmp_path, dedup_threshold=0.95, **components)
        assert report == type(report)(
            documents_loaded=0,
            chunks_created=0,
            chunks_skipped_already_indexed=0,
            duplicates_skipped=0,
            chunks_indexed=0,
            stopped_early=None,
        )

    def test_near_duplicate_documents_are_deduped_at_insert_time(
        self, tmp_path: Path
    ) -> None:

        (tmp_path / "a.md").write_text(
            "# Same Note\n\nExactly the same explanatory content, word for word."
        )
        (tmp_path / "b.md").write_text(
            "# Same Note\n\nExactly the same explanatory content, word for word."
        )
        components = _new_pipeline_components("ingest_dedup_test")

        report = run_ingestion(tmp_path, dedup_threshold=0.95, **components)

        assert report.duplicates_skipped >= 1
        assert report.chunks_indexed < report.chunks_created

    def test_bm25_index_stays_in_sync_after_ingestion(self, tmp_path: Path) -> None:

        (tmp_path / "notes.md").write_text(
            "# Backpropagation\n\nBackpropagation computes gradients via the chain rule across layers."
        )
        (tmp_path / "filler1.md").write_text(
            "# Weather\n\nIt was sunny with a light breeze in the afternoon."
        )
        (tmp_path / "filler2.md").write_text(
            "# Cooking\n\nSimmer the sauce for twenty minutes before serving."
        )
        (tmp_path / "filler3.md").write_text(
            "# Finance\n\nThe quarterly report showed steady revenue growth."
        )
        components = _new_pipeline_components("ingest_bm25_sync_test")
        run_ingestion(tmp_path, dedup_threshold=0.95, **components)

        results = components["sparse_index"].query(
            "backpropagation chain rule", top_k=5
        )
        assert len(results) >= 1

    def test_second_ingestion_run_is_idempotent_via_deterministic_ids(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "notes.md").write_text(
            "# Note\n\nSome stable content that will not change between runs."
        )
        components = _new_pipeline_components("ingest_idempotent_test")

        run_ingestion(tmp_path, dedup_threshold=0.95, **components)
        first_count = components["vector_store"].count()
        run_ingestion(tmp_path, dedup_threshold=0.95, **components)
        second_count = components["vector_store"].count()

        assert first_count == second_count

    def test_second_run_skips_already_indexed_chunks_without_re_embedding(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "notes.md").write_text(
            "# Note\n\nStable content that will not change between these two runs."
        )
        components = _new_pipeline_components("ingest_skip_indexed_test")

        run_ingestion(tmp_path, dedup_threshold=0.95, **components)
        second_report = run_ingestion(tmp_path, dedup_threshold=0.95, **components)

        assert (
            second_report.chunks_skipped_already_indexed == second_report.chunks_created
        )
        assert second_report.stopped_early is None

    def test_embedder_failure_partway_through_preserves_earlier_checkpoint_groups(
        self, tmp_path: Path
    ) -> None:

        for i in range(6):
            (tmp_path / f"f{i}.md").write_text(
                f"# Topic {i}\n\nDistinct filler content about topic number {i}."
            )
        components = _new_pipeline_components("ingest_partial_failure_test")
        components["embedder"] = FailAfterNCallsEmbedder(fail_after=1)

        report = run_ingestion(
            tmp_path, dedup_threshold=0.95, checkpoint_every=1, **components
        )

        assert report.stopped_early is not None
        assert "simulated quota exhaustion" in report.stopped_early
        assert 0 < report.chunks_indexed < report.chunks_created
        assert components["vector_store"].count() == report.chunks_indexed

    def test_resumes_after_partial_failure_once_embedder_recovers(
        self, tmp_path: Path
    ) -> None:
        for i in range(6):
            (tmp_path / f"f{i}.md").write_text(
                f"# Topic {i}\n\nDistinct filler content about topic number {i}."
            )
        components = _new_pipeline_components("ingest_resume_test")
        components["embedder"] = FailAfterNCallsEmbedder(fail_after=1)

        first_report = run_ingestion(
            tmp_path, dedup_threshold=0.95, checkpoint_every=1, **components
        )
        assert first_report.stopped_early is not None

        components["embedder"] = FakeEmbedder()
        second_report = run_ingestion(
            tmp_path, dedup_threshold=0.95, checkpoint_every=1, **components
        )

        assert second_report.stopped_early is None
        assert second_report.chunks_indexed == first_report.chunks_created
        assert (
            second_report.chunks_skipped_already_indexed == first_report.chunks_indexed
        )


class TestRetrieve:
    def test_query_returns_fused_ranked_chunks(self, tmp_path: Path) -> None:

        (tmp_path / "a.md").write_text(
            "# Regularization\n\nL2 regularization adds a penalty proportional to squared weight magnitude."
        )
        (tmp_path / "b.md").write_text(
            "# Unrelated\n\nBaking bread requires flour, water, yeast, and time for the dough to rise."
        )
        (tmp_path / "c.md").write_text(
            "# Weather\n\nIt was sunny with a light breeze in the afternoon."
        )
        (tmp_path / "d.md").write_text(
            "# Finance\n\nThe quarterly report showed steady revenue growth."
        )
        (tmp_path / "e.md").write_text(
            "# Travel\n\nThe flight was delayed by two hours due to weather."
        )
        components = _new_pipeline_components("retrieve_e2e_test")
        run_ingestion(tmp_path, dedup_threshold=0.95, **components)

        results = retrieve(
            "L2 regularization penalty weight magnitude",
            components["embedder"],
            components["vector_store"],
            components["sparse_index"],
            dense_top_k=10,
            sparse_top_k=10,
            rrf_k=60,
            rrf_dense_weight=0.7,
            rrf_sparse_weight=0.3,
        )

        assert len(results) >= 1
        assert results == sorted(results, key=lambda sc: sc.score, reverse=True)
        assert "regularization" in results[0].chunk.content.lower()

    def test_retrieve_against_empty_index_returns_empty(self, tmp_path: Path) -> None:
        components = _new_pipeline_components("retrieve_empty_test")
        results = retrieve(
            "anything",
            components["embedder"],
            components["vector_store"],
            components["sparse_index"],
            dense_top_k=10,
            sparse_top_k=10,
            rrf_k=60,
            rrf_dense_weight=0.7,
            rrf_sparse_weight=0.3,
        )
        assert results == []
