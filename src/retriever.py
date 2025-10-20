# src/retriever.py
"""
CPU retriever for Chroma REST.
- Embeds the query locally with sentence-transformers (same model as ingestion).
- Queries Chroma collection via HTTP and returns scored Passages.
"""
from __future__ import annotations
from typing import List
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .types import Passage
from .config import CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION

# Use the same embedding model as ingestion (fallback to e5-small)
try:
    from .config import EMBEDDINGS_MODEL as _EMB_MODEL
except Exception:
    _EMB_MODEL = "intfloat/e5-small"

_embedder = None  # lazy-loaded


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMB_MODEL, device="cpu")
    return _embedder


def _client():
    return chromadb.HttpClient(
        host=CHROMA_HOST, port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )


def retrieve(q: str, k: int = 8) -> List[Passage]:
    """
    Return top-k passages for the query. Score = 1 - distance (cosine space).
    """
    try:
        emb = _get_embedder().encode([q], device="cpu", normalize_embeddings=False)[0].tolist()
        client = _client()
        try:
            coll = client.get_or_create_collection(
                name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            coll = client.get_or_create_collection(
                name=CHROMA_COLLECTION, metadata={"metric": "cosine"}
            )

        res = coll.query(
            query_embeddings=[emb],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )

        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        out: List[Passage] = []
        for i, doc in enumerate(docs):
            if not doc:
                continue
            dist = float(dists[i]) if i < len(dists) else 1.0
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            src = meta.get("source", "unknown")
            out.append(Passage(text=doc, source=src, score=1.0 - dist))
        return out
    except Exception:
        return []
