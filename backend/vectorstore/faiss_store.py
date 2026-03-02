"""
DocuMind-AI — FAISS Vector Store

Build, search, save, and load a FAISS index with parallel chunk metadata.
"""

import os
import json
from pathlib import Path

import faiss
import numpy as np

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "index.faiss"
META_FILENAME = "metadata.json"


class FAISSStore:
    """
    Manages a FAISS flat index and an aligned list of chunk metadata.

    Usage:
        store = FAISSStore(dimension=384)
        store.add(embeddings, chunks)    # add vectors + metadata
        results = store.search(query_vec, top_k=5)
        store.save()                     # persist to disk
        store.load()                     # reload from disk
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)  # cosine (normalised)
        self.metadata: list[dict] = []  # aligned with index vectors

        self.save_dir = Path(settings.vectorstore_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Add
    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Add embeddings and their corresponding chunk metadata to the store.

        Args:
            embeddings: float32 array of shape (N, dimension)
            chunks:     list of N chunk dicts (from chunker)
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
            )
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        logger.info(
            f"Added {len(chunks)} vectors — total: {self.index.ntotal}"
        )

    # ------------------------------------------------------------------ Search
    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[dict]:
        """
        Search for the top-k most similar vectors.

        Args:
            query_embedding: float32 array of shape (1, dimension)
            top_k:           number of results (default from settings)

        Returns:
            List of dicts: { "text", "score", "metadata" }
        """
        top_k = top_k or settings.top_k

        if self.index.ntotal == 0:
            logger.warning("Index is empty — no results")
            return []

        # Clamp top_k to available vectors
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.metadata[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "score": float(score),
                    "metadata": chunk.get("metadata", {}),
                }
            )

        logger.info(f"Search returned {len(results)} results (top_k={top_k})")
        return results

    # ------------------------------------------------------------------ Persistence
    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_path = str(self.save_dir / INDEX_FILENAME)
        meta_path = str(self.save_dir / META_FILENAME)

        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved index ({self.index.ntotal} vectors) to {self.save_dir}"
        )

    def load(self) -> bool:
        """
        Load a previously saved index from disk.
        Returns True if loaded successfully, False if no saved index exists.
        """
        index_path = self.save_dir / INDEX_FILENAME
        meta_path = self.save_dir / META_FILENAME

        if not index_path.exists() or not meta_path.exists():
            logger.info("No saved index found — starting fresh")
            return False

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.dimension = self.index.d
        logger.info(
            f"Loaded index ({self.index.ntotal} vectors) from {self.save_dir}"
        )
        return True

    # ------------------------------------------------------------------ Utility
    def clear(self) -> None:
        """Reset the index and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Index cleared")

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    @property
    def document_sources(self) -> list[str]:
        """Return unique source filenames in the index."""
        sources = set()
        for m in self.metadata:
            src = m.get("metadata", {}).get("source", "unknown")
            sources.add(src)
        return sorted(sources)
