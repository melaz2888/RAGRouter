from typing import Optional
import os
from .types import Route
from .features import extract_features

_model = None
_threshold: Optional[float] = None

def _lazy_load():
    global _model, _threshold
    if _model is not None:
        return
    try:
        import joblib
        _model = joblib.load("models/router.joblib")
        tpath = "models/threshold.txt"
        if os.path.exists(tpath):
            with open(tpath, "r") as f:
                _threshold = float(f.read().strip())
        else:
            _threshold = 0.5
    except Exception:
        _model = None
        _threshold = 1.0  # force RAG by default for safety

def route(question: str) -> Route:
    # return "direct"  # we won't cite 
    _lazy_load()
    if _model is None:
        return "rag"
    feats = extract_features(question)
    import numpy as np
    # Map categorical 'wh' to simple one-hot
    wh = feats.pop("wh", "other")
    wh_map = ["what","who","where","when","why","how","other"]
    wh_vec = [1.0 if wh==w else 0.0 for w in wh_map]
    x = np.array([list(feats.values()) + wh_vec], dtype=float)
    p = _model.predict_proba(x)[0,1]
    return "rag" if p >= (_threshold or 0.5) else "direct"
