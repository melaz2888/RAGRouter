import time
from typing import List
from .types import RagAnswer, Passage
from .config import OLLAMA_HOST, OLLAMA_MODEL
import requests

def _mmr(passages: List[Passage], k_ctx: int = 5, lam: float = 0.7) -> List[Passage]:
    # Simplified: just top-k by score for skeleton
    return sorted(passages, key=lambda p: p.score, reverse=True)[:k_ctx]

def answer_rag(question: str, topk: int = 8, k_ctx: int = 5) -> RagAnswer:
    from .retriever import retrieve
    t0 = time.time()
    cands = retrieve(question, k=topk)
    ctx = _mmr(cands, k_ctx=k_ctx)
    ctx_text = "\n\n".join([f"[{i}] {p.text}\n(Source: {p.source})" for i, p in enumerate(ctx, 1)])
    prompt = f"""You must answer ONLY using the provided context; cite sources like [1], [2].
Question: {question}

Context:
{ctx_text}

Answer:"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        # "options": {"temperature": 0.1, "num_ctx": 4096},
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    ans = r.json().get("response", "").strip()
    return RagAnswer(answer=ans, passages=ctx, timing_ms=int((time.time() - t0) * 1000))
