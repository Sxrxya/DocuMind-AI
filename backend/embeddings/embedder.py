"""
DocuMind-AI — Embedding Generator

Wraps sentence-transformers to produce dense vector embeddings for text.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    """Lazy-loaded sentence-transformer embedding model."""

    _instance: "Embedder | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "Embedder":
        """Singleton — only one model loaded in memory."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(
                settings.embedding_model,
                cache_folder=settings.models_dir,
            )
            logger.info(
                f"Model loaded — dimension: {self._model.get_sentence_embedding_dimension()}"
            )
        return self._model

    @property
    def model(self) -> SentenceTransformer:
        return self._load_model()

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (e.g. 384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors → cosine = dot product
        )
        logger.info(f"Embedded {len(texts)} texts → shape {embeddings.shape}")
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string → shape (1, dimension)."""
        return self.embed([query])
