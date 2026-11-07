from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass


class LocalEmbedder(Embedder):
    def __init__(self, model: object, dimensions: int, batch_size: int = 32) -> None:
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def device(self) -> str:

        return str(self._model.device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


def load_local_embedder(
    model: str = "BAAI/bge-small-en-v1.5",
    dimensions: int = 384,
    batch_size: int = 32,
    device: str | None = None,
) -> LocalEmbedder:

    from sentence_transformers import SentenceTransformer

    return LocalEmbedder(
        SentenceTransformer(model, device=device),
        dimensions=dimensions,
        batch_size=batch_size,
    )


def get_embedder(
    provider: str,
    *,
    model: str,
    dimensions: int,
    batch_size: int | None = None,
    device: str | None = None,
) -> Embedder:

    if provider == "local":
        kwargs: dict[str, object] = {"model": model, "dimensions": dimensions}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if device is not None:
            kwargs["device"] = device
        return load_local_embedder(**kwargs)
    raise ValueError(f"Unknown embedding provider: {provider!r}")
