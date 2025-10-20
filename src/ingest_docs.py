"""
CPU-only document ingestion into Chroma (REST) for RAGRouter.

Usage (notes):
- Run: python -m src.ingest_docs --corpus data/corpus/ --collection kb --model intfloat/e5-small --batch-size 128
- Steps: load files, tiktoken-chunk (512/128), embed on CPU via SentenceTransformer, upsert into Chroma (cosine).
- Prints final JSON: {"files": N, "chunks": M, "collection": "..."} and exits 0. Skips empty chunks; continues on per-file errors.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings


from src.data_utils import chunk_text, load_corpus



def _connect_chroma(host: str = "localhost", port: int = 8000):
    # Always use HttpClient with telemetry off
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(anonymized_telemetry=False)
    )


def _get_or_create_collection(client, name: str):
    """
    Get or create a collection configured for cosine similarity.
    """
    try:
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    except Exception:
        # Older servers may expect "metric"
        return client.get_or_create_collection(name=name, metadata={"metric": "cosine"})


def _stable_id(source_path: str, idx: int) -> str:
    """
    Build stable ID as "<abs_source_path>#<chunk_idx>".
    """
    return f"{source_path}#{idx}"


def _prepare_chunks(
    items: List[Tuple[str, str, Dict[str, str]]],
    L: int = 512,
    S: int = 128,
) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    """
    From (doc_id, raw_text, meta), produce triplets (ids, docs, metadatas) for Chroma.
    Skips empty chunks.
    """
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, str]] = []

    for _, raw, meta in items:
        source = meta.get("source", "")
        chunks = chunk_text(raw, L=L, S=S)
        for i, ch in enumerate(chunks):
            if not ch.strip():
                continue
            ids.append(_stable_id(source, i))
            docs.append(ch)
            metas.append({"source": source})

    return ids, docs, metas


def _embed_cpu(model_name: str, texts: List[str], batch_size: int) -> List[List[float]]:
    """
    CPU-only embedding with SentenceTransformer.
    """
    model = SentenceTransformer(model_name, device="cpu")
    # No GPU flags; batch_size from CLI; deterministic enough for ingestion.
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        device="cpu",
        show_progress_bar=True,
        normalize_embeddings=False,  # Chroma will store raw vectors; cosine metric is set on the collection
    )
    # Ensure plain Python lists for Chroma client
    return [v.tolist() for v in vectors]


def ingest(
    corpus_dir: str,
    collection: str,
    model_name: str,
    batch_size: int,
    host: str = "localhost",
    port: int = 8000,
) -> Dict[str, int | str]:
    """
    Orchestrate ingestion: load -> chunk -> embed -> upsert -> return summary dict.
    """
    items = load_corpus(corpus_dir)
    n_files = len(items)
    print(f"[info] files found: {n_files}")

    ids, docs, metas = _prepare_chunks(items, L=512, S=128)
    n_chunks = len(docs)
    print(f"[info] chunks prepared (non-empty): {n_chunks}")

    if n_chunks == 0:
        summary = {"files": n_files, "chunks": 0, "collection": collection}
        print(json.dumps(summary, ensure_ascii=False))
        return summary

    print(f"[info] embedding on CPU with '{model_name}', batch_size={batch_size}")
    embeddings = _embed_cpu(model_name, docs, batch_size=batch_size)

    print(f"[info] connecting to Chroma at http://{host}:{port}, collection='{collection}'")
    client = _connect_chroma(host=host, port=port)
    coll = _get_or_create_collection(client, name=collection)

    # Upsert in one call if possible; if not, fall back to smaller chunks transparently.
    try:
        if hasattr(coll, "upsert"):
            coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        else:
            coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    except Exception as e:
        print(f"[warn] bulk upsert failed: {e}. Trying per-batch backup.")
        # Conservative fallback: split into safe batches
        step = max(1, min(512, len(ids)))
        for start in range(0, len(ids), step):
            end = start + step
            try:
                if hasattr(coll, "upsert"):
                    coll.upsert(
                        ids=ids[start:end],
                        documents=docs[start:end],
                        metadatas=metas[start:end],
                        embeddings=embeddings[start:end],
                    )
                else:
                    coll.add(
                        ids=ids[start:end],
                        documents=docs[start:end],
                        metadatas=metas[start:end],
                        embeddings=embeddings[start:end],
                    )
            except Exception as e2:
                print(f"[warn] partial upsert failed for range [{start}:{end}]: {e2}")

    summary = {"files": n_files, "chunks": n_chunks, "collection": collection}
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser("Ingest *.txt into Chroma (CPU-only).")
    parser.add_argument("--collection", default="kb_main",
                    help="Chroma collection name (>=3 chars, default: kb_main)")
    parser.add_argument("--corpus", required=True, help="Corpus directory containing *.txt (recursive).")
    # parser.add_argument("--collection", default="kb", help="Chroma collection name (default: kb).")
    parser.add_argument("--model", default="intfloat/e5-small", help="Sentence-Transformers model name.")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size (CPU).")
    parser.add_argument("--host", default="localhost", help="Chroma host.")
    parser.add_argument("--port", type=int, default=8000, help="Chroma port.")
    args = parser.parse_args()

    ingest(
        corpus_dir=args.corpus,
        collection=args.collection,
        model_name=args.model,
        batch_size=args.batch_size,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
