"""
DocuMind-AI — Text Chunking

Splits raw text into overlapping chunks suitable for embedding and retrieval.
"""

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    source: str = "unknown",
) -> list[dict]:
    """
    Split text into overlapping chunks.

    Args:
        text:       The full document text.
        chunk_size: Max characters per chunk (default from settings).
        overlap:    Overlap between consecutive chunks (default from settings).
        source:     Source filename for metadata.

    Returns:
        List of dicts: { "text": str, "index": int, "metadata": { ... } }
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if not text or not text.strip():
        logger.warning("Empty text provided — returning no chunks")
        return []

    if overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary (look for last '. ' in chunk)
        if end < len(text):
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.3:  # only if past 30% of chunk
                chunk = chunk[: break_point + 1]
                end = start + break_point + 1

        chunk = chunk.strip()
        if chunk:
            chunks.append(
                {
                    "text": chunk,
                    "index": idx,
                    "metadata": {
                        "source": source,
                        "chunk_index": idx,
                        "start_char": start,
                        "end_char": start + len(chunk),
                    },
                }
            )
            idx += 1

        # Advance, accounting for overlap
        start = end - overlap if end < len(text) else len(text)

    logger.info(
        f"Chunked '{source}' → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap})"
    )
    return chunks
