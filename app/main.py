from __future__ import annotations

import logging
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse

from app.env import load_dotenv

load_dotenv()

from app.router import router


app = FastAPI(title="URA EXACT Challenge QA")
app.include_router(router)


# Uvicorn sets up its own loggers; keep ours simple and compatible.
logging.getLogger("ura").setLevel(logging.INFO)


@app.get("/")
def index() -> RedirectResponse:
    # Convenience: open the UI immediately.
    return RedirectResponse(url="/demo")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "request_id": str(uuid.uuid4()), "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "request_id": str(uuid.uuid4()), "details": type(exc).__name__},
    )


@app.get("/health")
def health() -> dict[str, str]:
    # Expose a tiny bit of runtime info for debugging wrong-port/proxy issues.
    return {
        "status": "ok",
        "pid": str(os.getpid()),
    }
