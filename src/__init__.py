"""
RAGRouter v2

Latency-aware query routing for RAG systems.
"""
from .router import route, Route
from .retriever import retrieve, Passage, ingest_documents
from .generator import generate_direct, generate_rag

__all__ = [
    "route",
    "Route",
    "retrieve",
    "Passage",
    "ingest_documents",
    "generate_direct",
    "generate_rag",
]
