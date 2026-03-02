"""
DocuMind-AI — FastAPI Application Entry Point

Run with:  uvicorn backend.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.api.routes import router
from backend.api.middleware import register_middleware
from backend.rag.retriever import init_store

logger = get_logger(__name__)

# ------------------------------------------------------------------ App
app = FastAPI(
    title="DocuMind-AI",
    description="RAG-powered document Q&A system",
    version="0.1.0",
)

# ------------------------------------------------------------------ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ Middleware
register_middleware(app)

# ------------------------------------------------------------------ Routes
app.include_router(router)

# ------------------------------------------------------------------ Static (frontend)
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ------------------------------------------------------------------ Startup
@app.on_event("startup")
async def startup():
    logger.info("DocuMind-AI starting up …")
    logger.info(f"LLM provider : {settings.llm_provider}")
    logger.info(f"Embedding model: {settings.embedding_model}")

    # Load persisted FAISS index
    init_store()
    logger.info("Startup complete ✓")
