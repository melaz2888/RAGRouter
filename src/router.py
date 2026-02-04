"""
RAGRouter v2 Router

The "meta-model" that decides: direct or rag?

Routing logic:
1. Check keywords first -> if match, return "rag"
2. Check similarity to corpus -> if >= threshold, return "rag"
3. Otherwise -> return "direct"
"""
from __future__ import annotations

import os
from typing import List, Literal

from .config import (
    KEYWORDS_FILE,
    ROUTER_SIMILARITY_THRESHOLD,
)


Route = Literal["direct", "rag"]


# =============================================================================
# KEYWORDS
# =============================================================================

_keywords: List[str] | None = None


def _load_keywords() -> List[str]:
    """Load domain keywords from file."""
    global _keywords
    if _keywords is not None:
        return _keywords

    _keywords = []
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    _keywords.append(line.lower())

    return _keywords


def _check_keywords(question: str) -> bool:
    """Check if question contains any domain keywords."""
    keywords = _load_keywords()
    if not keywords:
        return False

    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)


def reload_keywords():
    """Force reload of keywords file."""
    global _keywords
    _keywords = None
    _load_keywords()


# =============================================================================
# SIMILARITY-BASED ROUTING
# =============================================================================

def _get_corpus_similarity(question: str) -> float:
    """
    Get the highest similarity score between question and corpus.
    Uses ChromaDB to find the most similar document.
    """
    from .retriever import embed_query, _get_collection

    try:
        query_embedding = embed_query(question)
        collection = _get_collection()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["distances"],
        )

        distances = (results.get("distances") or [[]])[0]
        if not distances:
            return 0.0

        # Convert distance to similarity (cosine distance -> similarity)
        distance = distances[0]
        similarity = 1.0 - distance
        return similarity

    except Exception as e:
        print(f"[router] Similarity check error: {e}")
        return 0.0


# =============================================================================
# MAIN ROUTING FUNCTION
# =============================================================================

def route(question: str) -> Route:
    """
    Decide whether to use direct generation or RAG.

    Logic:
    1. If question contains domain keywords -> "rag"
    2. If question is similar to corpus (>= threshold) -> "rag"
    3. Otherwise -> "direct"
    """
    # Step 1: Keyword check (fast path)
    if _check_keywords(question):
        return "rag"

    # Step 2: Similarity-based routing
    similarity = _get_corpus_similarity(question)

    if similarity >= ROUTER_SIMILARITY_THRESHOLD:
        return "rag"
    else:
        return "direct"
