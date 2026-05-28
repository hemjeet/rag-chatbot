"""
FastAPI backend for the RAG system.
Uses lifespan to initialize the RAG engine (vector store, models, etc.) at startup.
Gradio is mounted as a sub-application at "/" for a single-process deployment.
"""

from __future__ import annotations

import os
import shutil
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from rag_engine import RAGEngine
from memory import memory_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-14s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("api")


# ──────────────────────────────────────────────
# Lifespan — load everything on startup
# ──────────────────────────────────────────────

engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Called once when the server starts.
    Initializes the RAG engine, loads any existing vector store & models.
    """
    global engine
    logger.info("=" * 60)
    logger.info("  RAG System — Starting up")
    logger.info("=" * 60)

    engine = RAGEngine()
    engine.initialize()

    logger.info("🌐 Mounting Gradio app...")
    _mount_gradio(app)

    logger.info("=" * 60)
    logger.info("  RAG System — Ready")
    logger.info("=" * 60)

    yield  # ← App runs here

    logger.info("RAG System — Shutting down")


def _mount_gradio(app: FastAPI):
    """Import and mount the Gradio UI at root."""
    import gradio as gr
    from gradio_app import build_interface

    gradio_app = build_interface(engine)
    # Gradio 5 requires explicit queue/startup when mounted in FastAPI
    gradio_app.queue()
    if hasattr(gradio_app, "startup_events"):
        gradio_app.startup_events()
    elif hasattr(gradio_app, "run_startup_events"):
        gradio_app.run_startup_events()
    app = gr.mount_gradio_app(app, gradio_app, path="")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API with multi-document support and conversation memory.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    rerank: bool = Field(False, description="Enable NVIDIA reranking")


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: Optional[str]


class ConfigUpdateRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: Optional[float] = None
    openai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None


class UploadResponse(BaseModel):
    files_processed: List[str]
    new_chunks: int
    total_chunks: int
    total_files: int


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "engine_ready": engine.is_ready if engine else False,
    }


# ──────────────────────────────────────────────
# Document Endpoints
# ──────────────────────────────────────────────

@app.post("/api/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more documents to the knowledge base."""
    if not engine:
        raise HTTPException(500, "Engine not initialized")

    saved_paths = []
    try:
        for file in files:
            # Validate extension
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in {".txt", ".pdf", ".docx", ".md", ".csv", ".html", ".htm"}:
                raise HTTPException(
                    400,
                    f"Unsupported file type: '{ext}'. Supported: txt, pdf, docx, md, csv, html",
                )

            # Save to uploads directory
            dest = os.path.join(settings.upload_dir, file.filename)
            with open(dest, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_paths.append(dest)

        # Ingest all uploaded files
        result = engine.ingest_documents(saved_paths)
        return UploadResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, f"Failed to process documents: {e}")


@app.get("/api/documents", tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    return engine.list_documents()


@app.delete("/api/documents", tags=["Documents"])
async def clear_documents():
    """Clear the entire knowledge base."""
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    return engine.clear_index()


# ──────────────────────────────────────────────
# Query Endpoints
# ──────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(req: QueryRequest):
    """Ask a question about the indexed documents."""
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    if not engine.is_ready:
        raise HTTPException(400, "No documents indexed. Upload documents first.")

    try:
        result = engine.query(
            question=req.question,
            session_id=req.session_id,
            rerank=req.rerank,
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(500, f"Query failed: {e}")


@app.post("/api/query/stream", tags=["Query"])
async def query_stream(req: QueryRequest):
    """Streaming query — returns answer tokens via Server-Sent Events."""
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    if not engine.is_ready:
        raise HTTPException(400, "No documents indexed. Upload documents first.")

    async def event_generator():
        try:
            async for token in engine.query_stream(
                question=req.question,
                session_id=req.session_id,
                rerank=req.rerank,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ──────────────────────────────────────────────
# Session / Memory Endpoints
# ──────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/history", tags=["Sessions"])
async def get_session_history(session_id: str):
    """Get the chat history for a session."""
    session = memory_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return {"session_id": session_id, "history": session.get_history()}


@app.delete("/api/sessions/{session_id}/history", tags=["Sessions"])
async def clear_session_history(session_id: str):
    """Clear the chat history for a session."""
    deleted = memory_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return {"status": "cleared", "session_id": session_id}


# ──────────────────────────────────────────────
# Configuration Endpoints
# ──────────────────────────────────────────────

@app.get("/api/config", tags=["Configuration"])
async def get_config():
    """Get the current RAG system configuration."""
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    return engine.get_config()


@app.put("/api/config", tags=["Configuration"])
async def update_config(req: ConfigUpdateRequest):
    """Update the RAG system configuration at runtime."""
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    return engine.update_config(**req.model_dump(exclude_none=True))


# ──────────────────────────────────────────────
# Run Directly
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
