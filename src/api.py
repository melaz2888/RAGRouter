"""
RAGRouter v2 API

FastAPI service that orchestrates:
- Routing (router.py)
- Retrieval (retriever.py)
- Generation (generator.py)

Single endpoint: POST /ask
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from .router import route
from .retriever import retrieve, get_collection_stats, Passage
from .generator import generate_direct, generate_rag, check_ollama_health
from .config import API_HOST, API_PORT


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class AskRequest(BaseModel):
    question: str


class PassageResponse(BaseModel):
    text: str
    source: str
    score: float


class AskResponse(BaseModel):
    route: str
    answer: str
    passages: List[PassageResponse]
    timing_ms: int


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    chromadb: Optional[dict]


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="RAGRouter v2",
    description="Latency-aware query routing for RAG systems",
    version="2.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Check health of all services."""
    ollama_ok = check_ollama_health()
    chroma_stats = get_collection_stats()

    status = "healthy" if ollama_ok and "error" not in chroma_stats else "degraded"

    return HealthResponse(
        status=status,
        ollama=ollama_ok,
        chromadb=chroma_stats,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Answer a question using the appropriate route.

    Flow:
    1. Router decides: direct or rag (no retrieval yet)
    2. If direct: generate answer without context
    3. If rag: retrieve passages, then generate with context
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Step 1: Route decision (no retrieval call)
    route_decision = route(question)

    # Step 2: Generate based on route
    if route_decision == "direct":
        result = generate_direct(question)
        return AskResponse(
            route="direct",
            answer=result.answer,
            passages=[],
            timing_ms=result.timing_ms,
        )
    else:
        # RAG path: retrieve then generate
        passages = retrieve(question)
        result = generate_rag(question, passages)

        passage_responses = [
            PassageResponse(text=p.text, source=p.source, score=p.score)
            for p in result.passages
        ]

        return AskResponse(
            route="rag",
            answer=result.answer,
            passages=passage_responses,
            timing_ms=result.timing_ms,
        )


@app.get("/stats")
def stats():
    """Get collection statistics."""
    return get_collection_stats()


# =============================================================================
# CLI RUNNER
# =============================================================================

def run():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    run()
