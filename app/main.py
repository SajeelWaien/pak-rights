import json
import logging
import os

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag import ask, ask_stream


logger = logging.getLogger(__name__)

INDEX_DIR = Path("./storage/vector_index")

ChatModel = Literal[
    "paranine/gpt-oss-120b(high)",
    "paranine/qwen-3.8-27b"
]


# Request schema
class ChatRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=5000
    )

    model: ChatModel = "paranine/gpt-oss-120b(high)"


# Response schemas
class Source(BaseModel):
    file: str | None = None
    page: int | None = None
    score: float | None = None
    snippet: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


app = FastAPI(
    title="Pakistan Legal RAG API",
    description="Legal information assistant powered by Paranine and LlamaIndex",
    version="0.1.0"
)


# Allow your Next.js frontend to communicate with FastAPI
allowed_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_ready": INDEX_DIR.exists()
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if not INDEX_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail="Vector index not found. Run ingestion first."
        )

    try:
        result = ask(
            question=request.question,
            model_name=request.model
        )

        return ChatResponse(**result)

    except Exception:
        logger.exception("Failed to process legal query")

        raise HTTPException(
            status_code=500,
            detail="Failed to process query. Check server logs."
        )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    if not INDEX_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail="Vector index not found. Run ingestion first."
        )

    def event_stream():
        try:
            for event in ask_stream(
                question=request.question,
                model_name=request.model
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            yield 'data: {"type":"done"}\n\n'
        except Exception:
            logger.exception("Failed to stream legal query")
            error = {
                "type": "error",
                "message": "Failed to process query. Check server logs."
            }
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )