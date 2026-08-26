from __future__ import annotations

import numpy as np
import pytest

from rag.indexing.embeddings import LocalEmbedder, get_embedder


class _FakeSentenceTransformerModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, texts, *, batch_size, show_progress_bar, normalize_embeddings):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.array([[float(len(t)), 0.0] for t in texts])


class TestLocalEmbedder:
    def test_empty_input_returns_empty_without_calling_model(self) -> None:
        model = _FakeSentenceTransformerModel()
        embedder = LocalEmbedder(model, dimensions=384)
        assert embedder.embed([]) == []
        assert model.calls == []

    def test_returns_plain_lists_not_numpy_arrays(self) -> None:
        model = _FakeSentenceTransformerModel()
        embedder = LocalEmbedder(model, dimensions=384)
        results = embedder.embed(["a", "bb"])
        assert results == [[1.0, 0.0], [2.0, 0.0]]
        assert isinstance(results, list) and isinstance(results[0], list)

    def test_requests_normalized_embeddings_for_correct_cosine_similarity(self) -> None:

        model = _FakeSentenceTransformerModel()
        embedder = LocalEmbedder(model, dimensions=384)
        embedder.embed(["text"])
        assert model.calls[0]["normalize_embeddings"] is True

    def test_passes_configured_batch_size_through(self) -> None:
        model = _FakeSentenceTransformerModel()
        embedder = LocalEmbedder(model, dimensions=384, batch_size=8)
        embedder.embed(["text"])
        assert model.calls[0]["batch_size"] == 8

    def test_dimensions_property_reflects_constructor_arg(self) -> None:
        embedder = LocalEmbedder(_FakeSentenceTransformerModel(), dimensions=384)
        assert embedder.dimensions == 384

    def test_device_property_reports_underlying_model_device(self) -> None:
        model = _FakeSentenceTransformerModel()
        model.device = "cuda:0"
        embedder = LocalEmbedder(model, dimensions=384)
        assert embedder.device == "cuda:0"


class TestGetEmbedder:
    def test_returns_local_embedder_without_needing_an_api_key(self) -> None:
        model = _FakeSentenceTransformerModel()
        embedder = LocalEmbedder(model, dimensions=384)
        assert isinstance(embedder, LocalEmbedder)

    def test_forwards_device_only_for_local_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict] = []

        def fake_load_local_embedder(**kwargs):
            calls.append(kwargs)
            return LocalEmbedder(
                _FakeSentenceTransformerModel(),
                dimensions=kwargs.get("dimensions", 384),
            )

        monkeypatch.setattr(
            "rag.indexing.embeddings.load_local_embedder", fake_load_local_embedder
        )

        get_embedder(
            "local", model="BAAI/bge-small-en-v1.5", dimensions=384, device="cuda"
        )
        assert calls[0]["device"] == "cuda"

    def test_device_omitted_from_local_kwargs_when_not_specified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict] = []

        def fake_load_local_embedder(**kwargs):
            calls.append(kwargs)
            return LocalEmbedder(
                _FakeSentenceTransformerModel(),
                dimensions=kwargs.get("dimensions", 384),
            )

        monkeypatch.setattr(
            "rag.indexing.embeddings.load_local_embedder", fake_load_local_embedder
        )

        get_embedder("local", model="BAAI/bge-small-en-v1.5", dimensions=384)
        assert "device" not in calls[0]

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            get_embedder("cohere", model="x", dimensions=1)
