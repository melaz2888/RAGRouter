"""
RAGRouter v2 Generator

Handles LLM interactions via Ollama:
- Direct generation (no context)
- RAG generation (with retrieved passages)

All prompts in English. All settings from config.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import requests

from .config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    MAX_TOKENS_DIRECT,
    MAX_TOKENS_RAG,
    CONTEXT_TOP_K,
)
from .retriever import Passage


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GenerationResult:
    """Result from LLM generation."""
    answer: str
    timing_ms: int


@dataclass
class RAGResult:
    """Result from RAG generation with passages."""
    answer: str
    passages: List[Passage]
    timing_ms: int


# =============================================================================
# PROMPTS
# =============================================================================

DIRECT_PROMPT = """Answer the following question concisely and factually.

Question: {question}

Answer:"""


RAG_PROMPT = """Answer the question using ONLY the context provided below.
Cite your sources using [1], [2], etc. based on the passage numbers.
If the context doesn't contain the answer, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""


# =============================================================================
# LLM CALLS
# =============================================================================

def _call_ollama(prompt: str, max_tokens: int) -> str:
    """
    Call Ollama API with a prompt.

    Args:
        prompt: The prompt to send
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text

    Raises:
        Exception if Ollama is unavailable or returns an error
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        },
    }

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()

    return response.json().get("response", "").strip()


def generate_direct(question: str) -> GenerationResult:
    """
    Generate an answer without retrieval context.

    Args:
        question: The user's question

    Returns:
        GenerationResult with answer and timing
    """
    start = time.time()

    prompt = DIRECT_PROMPT.format(question=question)
    answer = _call_ollama(prompt, MAX_TOKENS_DIRECT)

    timing_ms = int((time.time() - start) * 1000)
    return GenerationResult(answer=answer, timing_ms=timing_ms)


def generate_rag(question: str, passages: List[Passage]) -> RAGResult:
    """
    Generate an answer using retrieved passages as context.

    Args:
        question: The user's question
        passages: Retrieved passages to use as context

    Returns:
        RAGResult with answer, passages used, and timing
    """
    start = time.time()

    # Limit to CONTEXT_TOP_K passages
    passages = passages[:CONTEXT_TOP_K]

    # Build context string
    if passages:
        context_lines = []
        for i, p in enumerate(passages, 1):
            context_lines.append(f"[{i}] {p.text}")
        context = "\n\n".join(context_lines)
    else:
        context = "(No relevant passages found)"

    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = _call_ollama(prompt, MAX_TOKENS_RAG)

    timing_ms = int((time.time() - start) * 1000)
    return RAGResult(answer=answer, passages=passages, timing_ms=timing_ms)


def check_ollama_health() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        return any(OLLAMA_MODEL in name for name in model_names)
    except Exception:
        return False
