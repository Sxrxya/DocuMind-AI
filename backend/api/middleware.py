"""
DocuMind-AI — Middleware

Request timing, structured error handling, and logging middleware.
"""

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.logging import get_logger

logger = get_logger(__name__)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Log the processing time for every request."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start

        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} "
            f"({elapsed:.3f}s)"
        )
        response.headers["X-Process-Time"] = f"{elapsed:.3f}"
        return response


async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return structured JSON."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


def register_middleware(app: FastAPI) -> None:
    """Register all custom middleware on the FastAPI app."""
    app.add_middleware(RequestTimingMiddleware)
    app.add_exception_handler(Exception, global_exception_handler)
