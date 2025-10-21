from __future__ import annotations
from typing import Optional
import os, re
from .rag_types import Route
from .retriever import retrieve
try:
    from .features import extract_features
except Exception:
    extract_features = None  # ML optionnel

_model = None
_threshold: Optional[float] = None
_NUMERIC_KEYS = ["len_words","has_num","tfidf_rare","top1","topk_mean","gap12","uniq_src"]
_WH_KEYS      = ["what","who","where","when","why","how","other"]

_tok_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}")
def _tokset(s:str): return set(_tok_re.findall((s or "").lower()))
def _overlap_frac(q:str, t:str)->float:
    tq, tt = _tokset(q), _tokset(t)
    if not tq or not tt: return 0.0
    return len(tq & tt) / len(tq)  # rappel des tokens requête couverts

def _lazy_load():
    global _model, _threshold
    if _model is not None: return
    try:
        import joblib
        _model = joblib.load("models/router.joblib")
        thr_path = "models/threshold.txt"
        _threshold = float(open(thr_path).read().strip()) if os.path.exists(thr_path) else 0.5
    except Exception:
        _model, _threshold = None, None

def _rule_based_route(question: str) -> Route:
    # Sonde retrieval
    try:
        hits = retrieve(question, k=3)
    except Exception:
        return "direct"
    if not hits: return "direct"
    top = max(float(getattr(h, "score", 0.0)) for h in hits)
    max_ov = max(_overlap_frac(question, h.text) for h in hits if getattr(h,"text",None)) if hits else 0.0
    # Garde: exiger score ET recouvrement lexical
    return "rag" if (top >= 0.35 and max_ov >= 0.20) else "direct"

def route(question: str) -> Route:
    _lazy_load()
    if _model is None or extract_features is None:
        return _rule_based_route(question)
    # Chemin ML avec garde de sécurité identique
    try:
        import numpy as np
        feats = extract_features(question)
        x = [float(feats.get(k, 0.0)) for k in _NUMERIC_KEYS]
        wh = str(feats.get("wh","other"))
        x += [1.0 if wh==w else 0.0 for w in _WH_KEYS]
        proba = float(_model.predict_proba(np.array([x], dtype=float))[0,1])
        thr = 0.5 if _threshold is None else _threshold
        pred = "rag" if proba >= thr else "direct"
        # Garde finale: si recouvrement lexical trop faible, basculer direct
        if pred == "rag":
            hits = retrieve(question, k=3) or []
            max_ov = max((_overlap_frac(question, h.text) for h in hits), default=0.0)
            if max_ov < 0.20:
                return "direct"
        return pred
    except Exception:
        return _rule_based_route(question)
