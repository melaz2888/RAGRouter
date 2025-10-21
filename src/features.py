# src/features.py
"""
Feature extraction for Router ML (CPU-only).
- Uses retrieval stats + simple query features + TF-IDF rarity.
- Vectorizer is fit on data/corpus/*.txt and cached under models/tfidf.joblib.
Contract:
    extract_features(q: str) -> Dict[str, float|int|str]
Insertion order of numeric features is FIXED to match train/infer.
"""
from __future__ import annotations
from typing import Dict, List
import os, glob, re, json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .retriever import retrieve

_MODELS_DIR = "models"
_VECT_PATH = os.path.join(_MODELS_DIR, "tfidf.joblib")
_STATS_PATH = os.path.join(_MODELS_DIR, "tfidf_stats.json")

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")

def _tokenize(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())

def _load_corpus_texts(root: str = "data/corpus") -> List[str]:
    files = glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)
    out = []
    for p in files:
        try:
            out.append(open(p, "r", encoding="utf-8", errors="ignore").read())
        except Exception:
            pass
    return out or ["placeholder text"]  # avoid empty fit

def _ensure_vectorizer() -> TfidfVectorizer:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    if os.path.exists(_VECT_PATH):
        return joblib.load(_VECT_PATH)
    texts = _load_corpus_texts()
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=1)
    vect.fit(texts)
    joblib.dump(vect, _VECT_PATH)
    # store idf p95 for rarity metric
    p95 = float(np.percentile(vect.idf_, 95)) if hasattr(vect, "idf_") else 0.0
    json.dump({"idf_p95": p95}, open(_STATS_PATH, "w"))
    return vect

def _idf_rarity(query: str) -> float:
    vect = _ensure_vectorizer()
    stats = {"idf_p95": 0.0}
    if os.path.exists(_STATS_PATH):
        try: stats = json.load(open(_STATS_PATH, "r"))
        except Exception: pass
    p95 = float(stats.get("idf_p95", 0.0))

    vocab = vect.vocabulary_ or {}
    idf = getattr(vect, "idf_", None)
    toks = _tokenize(query)
    if not toks or idf is None: return 0.0
    hits = []
    for t in toks:
        idx = vocab.get(t)
        if idx is not None:
            hits.append(float(idf[idx]))
    if not hits: return 0.0
    # proportion of tokens above p95
    above = sum(1 for v in hits if v > p95)
    return above / max(1, len(hits))

def _wh_bucket(q: str) -> str:
    ql = (q or "").strip().lower()
    for w in ("what", "who", "where", "when", "why", "how"):
        if ql.startswith(w + " ") or ql == w or ql.startswith(w + "?"):
            return w
    return "other"

def extract_features(q: str) -> Dict[str, float | int | str]:
    # simple features
    toks = _tokenize(q)
    len_words = len(toks)
    has_num = int(bool(re.search(r"\d", q or "")))
    tfidf_rare = _idf_rarity(q)

    # retrieval probe
    hits = []
    try:
        hits = retrieve(q, k=8)
    except Exception:
        hits = []

    scores = [float(getattr(h, "score", 0.0)) for h in hits]
    sources = [str(getattr(h, "source", "")) for h in hits]
    top1 = max(scores) if scores else 0.0
    topk_mean = float(np.mean(scores)) if scores else 0.0
    gap12 = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
    uniq_src = len(set(s for s in sources if s))

    # FIXED insertion order for numeric features:
    feats: Dict[str, float | int | str] = {}
    feats["len_words"] = float(len_words)
    feats["has_num"] = float(has_num)
    feats["tfidf_rare"] = float(tfidf_rare)
    feats["top1"] = float(top1)
    feats["topk_mean"] = float(topk_mean)
    feats["gap12"] = float(gap12)
    feats["uniq_src"] = float(uniq_src)
    feats["wh"] = _wh_bucket(q)  # kept as string; one-hot later
    return feats
