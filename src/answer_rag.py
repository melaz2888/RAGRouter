# src/answer_rag.py
"""
RAG answering via Ollama (CPU).
- Retrieve candidates from Chroma, select up to K_CTX passages.
- With context: answer strictly from context with inline [i] citations.
- Without context: answer briefly with NO citations.
- Caps generation for CPU latency: num_predict=128, num_ctx=512.
"""
from __future__ import annotations
import os, re, time, requests
from typing import List, Dict, Any

from .types import Passage, RagAnswer
from .config import OLLAMA_HOST, OLLAMA_MODEL, TOPK_DEFAULT, K_CTX_DEFAULT
from .retriever import retrieve

_USE_MMR = os.getenv("USE_MMR", "false").lower() in {"1", "true", "yes"}

_token_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")


def _lex(s: str) -> set[str]:
    return set(_token_re.findall((s or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _lex(a), _lex(b)
    if not ta or not tb:
        return 0.0
    inter, uni = len(ta & tb), len(ta | tb)
    return inter / uni if uni else 0.0


def _select_ctx(passages: List[Passage], k_ctx: int) -> List[Passage]:
    if not passages:
        return []
    ps = sorted(passages, key=lambda p: float(getattr(p, "score", 0.0)), reverse=True)
    k_ctx = max(1, int(k_ctx))
    if (not _USE_MMR) or len(ps) <= 1:
        return ps[:k_ctx]
    sel, rem, lam = [ps[0]], ps[1:], 0.65
    while rem and len(sel) < k_ctx:
        best_i, best_val = 0, float("-inf")
        for i, cand in enumerate(rem):
            rel = float(getattr(cand, "score", 0.0))
            div = max((_jaccard(cand.text, s.text) for s in sel), default=0.0)
            val = lam * rel - (1.0 - lam) * div
            if val > best_val:
                best_val, best_i = val, i
        sel.append(rem.pop(best_i))
    return sel


def _build_prompt(question: str, ctx: List[Passage]) -> str:
    if ctx:
        lines = [
            "Tu es un assistant strictement extractif.",
            "RÈGLE: Réponds uniquement à partir du contexte ci-dessous et cite les sources comme [i].",
            "",
            "Contexte :",
        ]
        for i, p in enumerate(ctx, 1):
            lines.append(f"[{i}] {p.text} (source: {p.source})")
        lines += [
            "",
            "Réponds de façon concise en citant [i] pour chaque fait clé.",
            f"Question : {question}",
            "Réponse :",
        ]
        return "\n".join(lines)
    return f"Réponds brièvement. Si l'information manque, dis-le.\nQuestion : {question}\nRéponse :"


def _call_ollama(prompt: str) -> str:
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 128, "num_ctx": 512},  # keep CPU latency bounded
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def answer_rag(question: str, topk: int = TOPK_DEFAULT, k_ctx: int = K_CTX_DEFAULT) -> RagAnswer:
    t0 = time.time()
    cands = retrieve(question, k=topk)
    ctx = _select_ctx(cands, k_ctx=k_ctx)
    prompt = _build_prompt(question, ctx)
    ans = _call_ollama(prompt)

    if ctx:
        if not re.search(r"\[\d+]", ans):
            ans += "\n\nSources: " + " ".join(f"[{i}]" for i in range(1, len(ctx) + 1))
    else:
        ans = re.sub(r"\[\d+]", "", ans).strip()

    return RagAnswer(answer=ans, passages=ctx, timing_ms=int((time.time() - t0) * 1000))
