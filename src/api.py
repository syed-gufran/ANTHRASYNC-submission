"""FastAPI service exposing the knowledge assistant over HTTP.

Run:

    uvicorn src.api:app --reload

Endpoints:
    GET  /health        -> liveness probe
    POST /ask           -> ask a question, get a grounded answer + sources
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.rag_graph import answer_question

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG-powered Q&A over internal documents, built with LangGraph.",
    version="1.0.0",
)


class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the refund policy?"])
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Optional prior turns, enabling follow-up questions.",
    )


class Source(BaseModel):
    document: str
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float


@app.get("/health")
def health() -> dict:
    """Liveness probe + which model is configured."""
    return {"status": "ok", "model": settings.llm_model}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question against the indexed knowledge base."""
    try:
        result = answer_question(
            request.question,
            chat_history=[turn.model_dump() for turn in request.chat_history],
        )
    except FileNotFoundError as exc:
        # Index not built yet — actionable 503 rather than an opaque 500.
        raise HTTPException(
            status_code=503,
            detail=f"{exc} Run `python -m src.ingest` to build the index.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(**result)
