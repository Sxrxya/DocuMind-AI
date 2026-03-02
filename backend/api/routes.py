"""
DocuMind-AI — API Routes

All REST endpoints: /upload, /ask, /documents, /health
"""

import os
import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.rag.retriever import retrieve, index_document, store
from backend.rag.prompt_builder import build_prompt
from backend.rag.generator import LLMGenerator

logger = get_logger(__name__)
router = APIRouter()

# LLM generator singleton
llm = LLMGenerator()


# ------------------------------------------------------------------ Schemas
class AskRequest(BaseModel):
    question: str
    history: list[dict] | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    time_taken: float


# ------------------------------------------------------------------ /upload
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, TXT) for indexing."""
    # Validate file type
    allowed = {".pdf", ".docx", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(allowed)}"
        )

    # Save uploaded file
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / file.filename

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved uploaded file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Index the document
    try:
        num_chunks = index_document(str(file_path), file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": num_chunks,
        "total_vectors": store.total_vectors,
    }


# ------------------------------------------------------------------ /ask
@router.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question about indexed documents."""
    if store.total_vectors == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload a document first."
        )

    start = time.time()

    # 1. Retrieve relevant chunks
    results = retrieve(req.question)

    # 2. Build prompt
    messages = build_prompt(req.question, results, req.history)

    # 3. Generate answer
    try:
        answer = llm.generate(messages)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    elapsed = time.time() - start

    return AskResponse(
        answer=answer,
        sources=[
            {
                "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "score": round(r["score"], 4),
                "source": r["metadata"].get("source", "unknown"),
            }
            for r in results
        ],
        time_taken=round(elapsed, 2),
    )


# ------------------------------------------------------------------ /ask/stream
@router.post("/ask/stream")
async def ask_question_stream(req: AskRequest):
    """Stream the answer token-by-token (SSE)."""
    if store.total_vectors == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload a document first."
        )

    results = retrieve(req.question)
    messages = build_prompt(req.question, results, req.history)

    def event_stream():
        try:
            for token in llm.generate_stream(messages):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------ /documents
@router.get("/documents")
async def list_documents():
    """List all indexed documents."""
    return {
        "documents": store.document_sources,
        "total_vectors": store.total_vectors,
    }


# ------------------------------------------------------------------ /health
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "indexed_documents": len(store.document_sources),
        "total_vectors": store.total_vectors,
        "llm_provider": settings.llm_provider,
    }
