# src/label_offline.py
"""
Auto-label QA items into {direct, needs-RAG} with simple heuristics.
Writes data/labels.jsonl with {"id","question","route"}.
CLI:
    python src/label_offline.py --qa data/qa/seed.jsonl [--limit 500]
"""
from __future__ import annotations
import os, json, argparse, sys
from typing import Dict, Iterable, List

try:
    # when run as module: python -m src.label_offline
    from .features import extract_features
except Exception:
    # when run as script: python src/label_offline.py
    from src.features import extract_features

OUT_PATH = "data/labels.jsonl"

def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue

def _heuristic_label(q: str) -> str:
    f = extract_features(q)
    # thresholds tuned for tiny corpora; conservative on needs-RAG recall
    cond_retrieval = (f["top1"] >= 0.60) or (f["topk_mean"] >= 0.55) or (f["uniq_src"] >= 2)
    cond_complex = (f["len_words"] > 10 and f["has_num"] >= 1.0)
    return "needs-RAG" if (cond_retrieval or cond_complex) else "direct"

def main():
    ap = argparse.ArgumentParser("Auto-label QA into {direct, needs-RAG}.")
    ap.add_argument("--qa", required=True, help="Path to a JSONL file under data/qa/")
    ap.add_argument("--limit", type=int, default=0, help="Max items to process (0=all)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for item in _iter_jsonl(args.qa):
            qid = item.get("id") or f"auto_{n}"
            q = item.get("question") or ""
            if not q: continue
            route = _heuristic_label(q)
            out.write(json.dumps({"id": qid, "question": q, "route": route}, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit: break

    print(f"[labels] wrote {n} items to {OUT_PATH}")
    sys.exit(0)

if __name__ == "__main__":
    main()
