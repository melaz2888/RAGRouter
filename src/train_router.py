# src/train_router.py
"""
Train a Logistic Regression router on CPU.
- Loads data/labels.jsonl with {"id","question","route"} where route in {"direct","needs-RAG"}.
- Uses extract_features() for each question.
- One-hot encodes 'wh' and keeps a FIXED numeric feature order from features.extract_features.
- Calibrates a threshold to achieve recall(y=1) >= 0.90 if possible.
Outputs:
    models/router.joblib
    models/threshold.txt
    reports/router_metrics.json
CLI:
    python src/train_router.py
"""
from __future__ import annotations
import os, json, random
from typing import List, Dict, Tuple

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .features import extract_features

LABELS_PATH = "data/qa/labels.jsonl"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
ROUTER_PATH = os.path.join(MODELS_DIR, "router.joblib")
THRESH_PATH = os.path.join(MODELS_DIR, "threshold.txt")
METRICS_PATH = os.path.join(REPORTS_DIR, "router_metrics.json")

_WH_KEYS = ["what", "who", "where", "when", "why", "how", "other"]
_NUMERIC_KEYS = ["len_words", "has_num", "tfidf_rare", "top1", "topk_mean", "gap12", "uniq_src"]

def _load_labels(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("question") and obj.get("route") in {"direct", "needs-RAG"}:
                    out.append(obj)
            except Exception:
                pass
    return out

def _vec_from_feats(feats: Dict[str, float|int|str]) -> List[float]:
    x = [float(feats[k]) for k in _NUMERIC_KEYS]
    wh = str(feats.get("wh", "other"))
    x += [1.0 if wh == w else 0.0 for w in _WH_KEYS]
    return x

def _split_idx(n: int, seed: int = 13, frac_val: float = 0.2) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = max(1, int(n * frac_val))
    return idx[cut:], idx[:cut]

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    data = _load_labels(LABELS_PATH)
    if not data:
        print(f"[train] no labels at {LABELS_PATH}")
        return

    X, y = [], []
    for row in data:
        feats = extract_features(row["question"])
        X.append(_vec_from_feats(feats))
        y.append(1 if row["route"] == "needs-RAG" else 0)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    tr_idx, va_idx = _split_idx(len(y))
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    clf = LogisticRegression(max_iter=300, class_weight={1: 2.0})
    clf.fit(Xtr, ytr)

    # pick threshold for recall >= 0.90 if possible
    proba = clf.predict_proba(Xva)[:, 1]
    thresholds = sorted(set(list(proba) + [i/100 for i in range(5, 96)]))
    best_thr = 0.5
    best = {"recall": -1, "f1": -1, "precision": -1}
    for thr in thresholds:
        yhat = (proba >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(yva, yhat, average="binary", zero_division=0)
        if rec >= 0.90 and f1 > best["f1"]:
            best_thr = float(thr)
            best = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

    if best["recall"] < 0:  # no threshold hit 0.90 recall
        # choose threshold with max f1
        f1_best, thr_best = -1.0, 0.5
        for thr in thresholds:
            yhat = (proba >= thr).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(yva, yhat, average="binary", zero_division=0)
            if f1 > f1_best:
                f1_best, thr_best = float(f1), float(thr)
        best_thr = thr_best
        prec, rec, f1, _ = precision_recall_fscore_support(yva, (proba >= best_thr).astype(int),
                                                           average="binary", zero_division=0)
        best = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

    # save
    joblib.dump(clf, ROUTER_PATH)
    with open(THRESH_PATH, "w") as f:
        f.write(str(best_thr))
    cm = confusion_matrix(yva, (proba >= best_thr).astype(int)).tolist()
    json.dump({"val_metrics": best, "threshold": best_thr, "confusion_matrix": cm},
              open(METRICS_PATH, "w"), indent=2)
    print(f"[train] saved {ROUTER_PATH}, threshold={best_thr:.3f}")
    print(f"[train] val: precision={best['precision']:.3f} recall={best['recall']:.3f} f1={best['f1']:.3f}")
    print(f"[train] confusion_matrix={cm}")

if __name__ == "__main__":
    main()
