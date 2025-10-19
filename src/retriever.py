from typing import List
from .types import Passage
from .config import CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION

def retrieve(query: str, k: int = 8) -> List[Passage]:
    """Retrieve top-k passages from Chroma. Returns empty if collection absent."""
    try:
        import chromadb
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        col = client.get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        res = col.query(query_texts=[query], n_results=k, include=["documents", "distances", "metadatas"])
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out = []
        for text, dist, meta in zip(docs, dists, metas):
            score = 1 - float(dist) if dist is not None else 0.0
            source = (meta or {}).get("source", "unknown")
            out.append(Passage(score=score, text=text, source=source))
        return out
    except Exception:
        # Chroma not running or empty index
        return []
