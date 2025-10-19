import re
from typing import Dict

_WH = re.compile(r"\b(what|who|where|when|why|how)\b", re.I)

def extract_features(question: str) -> Dict:
    q = question.strip()
    wh = (_WH.search(q) or ["other"])[0].lower() if _WH.search(q) else "other"
    has_num = bool(re.search(r"\d", q))
    length = len(q.split())
    # Placeholders for first run (safe zeros); extend later
    feats = {
        "len": length,
        "wh": wh,
        "has_num": int(has_num),
        "ner_cnt": 0,
        "tfidf_rare": 0.0,
        "top1": 0.0,
        "topk_mean": 0.0,
        "gap12": 0.0,
        "uniq_src": 0,
        "slm_probe": 0,
    }
    return feats
