"""
DocuMind-AI — Retriever

Embeds a user query, searches FAISS, and returns the top-k matching chunks.
"""

from backend.embeddings.embedder import Embedder
from backend.vectorstore.faiss_store import FAISSStore
from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

# Module-level singletons (initialised at startup in main.py)
embedder = Embedder()
store = FAISSStore()


def init_store() -> None:
    """Load persisted FAISS index from disk (call on app startup)."""
    loaded = store.load()
    if loaded:
        logger.info(f"Retriever ready — {store.total_vectors} vectors in index")
    else:
        logger.info("Retriever ready — empty index (no documents yet)")


def index_document(file_path: str, filename: str) -> int:
    """
    Full ingest pipeline for one document:
    extract → chunk → embed → add to FAISS → save.

    Returns the number of chunks indexed.
    """
    from backend.ingestion.extractor import extract_text
    from backend.ingestion.chunker import chunk_text

    # 1. Extract
    text = extract_text(file_path)
    if not text.strip():
        logger.warning(f"No text extracted from {filename}")
        return 0

    # 2. Chunk
    chunks = chunk_text(text, source=filename)
    if not chunks:
        return 0

    # 3. Embed
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)

    # 4. Add to index & save
    store.add(embeddings, chunks)
    store.save()

    logger.info(f"Indexed '{filename}': {len(chunks)} chunks")
    return len(chunks)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    Returns list of { "text", "score", "metadata" }.
    """
    top_k = top_k or settings.top_k
    query_vec = embedder.embed_query(query)
    results = store.search(query_vec, top_k=top_k)
    return results
