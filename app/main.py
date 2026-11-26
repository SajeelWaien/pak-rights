import json
import logging
import os

from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.memory import get_memory
from app.rag import ask, ask_stream
from app.whatsapp_wh import router as whatsapp_router


logger = logging.getLogger(__name__)

INDEX_DIR = Path("./storage/vector_index")

ChatModel = Literal[
    "paranine/gpt-oss-120b(high)",
    "paranine/qwen-3.8-27b"
]


# Request schema
class ChatRequest(BaseModel):
    conversation_id: str | None = None
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
app.include_router(whatsapp_router)


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

    # Reuse existing ID or generate one for a new conversation
    conversation_id = str(
        request.conversation_id or uuid4()
    )

    def event_stream():

        # Send conversation ID immediately, before generating tokens
        payload = json.dumps({
            "type": "conversation",
            "conversation_id": conversation_id,
        })
        yield f"data: {payload}\n\n"
        
        try:
            for event in ask_stream(
                conversation_id=conversation_id,
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

@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_history(conversation_id: str):

    memory = get_memory(conversation_id)

    exists = await memory.sql_store.count_messages(
        conversation_id, status=None
    ) > 0

    if not exists:
        raise HTTPException(
            status_code=404,
            detail=f"No conversation found with ID: {conversation_id}"
        )

    history = await memory.aget_all()

    messages = []

    for message in history:

        role = message.role.value

        # Only return messages visible in the chat UI
        if role not in ["user", "assistant"]:
            continue

        messages.append({
            "role": role,
            "content": message.content or ""
        })

    return {
        "conversation_id": conversation_id,
        "messages": messages
    }