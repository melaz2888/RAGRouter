"""
RAGRouter v2 Retriever

Handles:
- Text chunking (split documents into smaller pieces)
- Embedding (convert text to vectors using e5-small)
- ChromaDB operations (store and search vectors)

All settings come from config.py — no hardcoded values.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List

import chromadb
import tiktoken
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CORPUS_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_PREFIX_QUERY,
    EMBEDDING_PREFIX_PASSAGE,
    RETRIEVAL_TOP_K,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Passage:
    """A retrieved passage with metadata."""
    text: str
    source: str
    score: float


# =============================================================================
# EMBEDDING
# =============================================================================

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedder


def embed_query(text: str) -> List[float]:
    """
    Embed a query string.
    Uses E5 prefix "query: " as required by the model.
    """
    model = _get_embedder()
    prefixed = f"{EMBEDDING_PREFIX_QUERY}{text}"
    vector = model.encode(prefixed, device="cpu", normalize_embeddings=True)
    return vector.tolist()


def embed_passage(text: str) -> List[float]:
    """
    Embed a passage/document string.
    Uses E5 prefix "passage: " as required by the model.
    """
    model = _get_embedder()
    prefixed = f"{EMBEDDING_PREFIX_PASSAGE}{text}"
    vector = model.encode(prefixed, device="cpu", normalize_embeddings=True)
    return vector.tolist()


def embed_passages_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Embed multiple passages in batches (more efficient).
    """
    model = _get_embedder()
    prefixed = [f"{EMBEDDING_PREFIX_PASSAGE}{t}" for t in texts]
    vectors = model.encode(
        prefixed,
        device="cpu",
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vectors]


# =============================================================================
# CHUNKING
# =============================================================================

_tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Token overlap between chunks

    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []

    tokens = _tokenizer.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    stride = chunk_size - overlap

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start += stride

    return chunks


# =============================================================================
# CHROMADB
# =============================================================================

def _get_chroma_client():
    """Connect to ChromaDB server."""
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client=None):
    """Get or create the main collection."""
    if client is None:
        client = _get_chroma_client()

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(corpus_dir: str = None) -> dict:
    """
    Ingest all .txt and .md files from corpus directory into ChromaDB.

    Args:
        corpus_dir: Directory containing .txt or .md files (default: config.CORPUS_DIR)

    Returns:
        Summary dict with file_count and chunk_count
    """
    if corpus_dir is None:
        corpus_dir = str(CORPUS_DIR)

    # Find all text and markdown files
    txt_pattern = os.path.join(corpus_dir, "**", "*.txt")
    md_pattern = os.path.join(corpus_dir, "**", "*.md")
    files = sorted(glob.glob(txt_pattern, recursive=True) + glob.glob(md_pattern, recursive=True))

    if not files:
        print(f"[ingest] No .txt or .md files found in {corpus_dir}")
        return {"file_count": 0, "chunk_count": 0}

    print(f"[ingest] Found {len(files)} files")

    # Process files into chunks
    all_ids = []
    all_texts = []
    all_metadatas = []

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"[ingest] Skipping {filepath}: {e}")
            continue

        chunks = chunk_text(content)
        source = os.path.abspath(filepath)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}#{i}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append({"source": source})

    if not all_texts:
        print("[ingest] No chunks to ingest")
        return {"file_count": len(files), "chunk_count": 0}

    print(f"[ingest] Embedding {len(all_texts)} chunks...")
    embeddings = embed_passages_batch(all_texts)

    print(f"[ingest] Uploading to ChromaDB collection '{CHROMA_COLLECTION}'...")
    collection = _get_collection()

    # Upsert in batches
    batch_size = 500
    for i in range(0, len(all_ids), batch_size):
        end = min(i + batch_size, len(all_ids))
        collection.upsert(
            ids=all_ids[i:end],
            documents=all_texts[i:end],
            metadatas=all_metadatas[i:end],
            embeddings=embeddings[i:end],
        )

    print(f"[ingest] Done. {len(all_texts)} chunks ingested.")
    return {"file_count": len(files), "chunk_count": len(all_texts)}


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Passage]:
    """
    Retrieve top-k passages for a query.

    Args:
        query: The search query
        top_k: Number of results to return

    Returns:
        List of Passage objects with text, source, and score
    """
    query_embedding = embed_query(query)
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    documents = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    passages = []
    for i, doc in enumerate(documents):
        if not doc:
            continue

        # Convert distance to similarity score (cosine distance -> similarity)
        distance = distances[i] if i < len(distances) else 1.0
        score = 1.0 - distance

        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "unknown")

        passages.append(Passage(text=doc, source=source, score=score))

    return passages


def get_collection_stats() -> dict:
    """Get statistics about the current collection."""
    try:
        collection = _get_collection()
        count = collection.count()
        return {"collection": CHROMA_COLLECTION, "document_count": count}
    except Exception as e:
        return {"error": str(e)}
