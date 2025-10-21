"""
Generate synthetic QA data and matching corpus files for router training.

Outputs:
 - Corpus: data/corpus/synth/*.txt (domain docs with facts)
 - QA JSONL: data/qa/synth_all.jsonl (N items)
 - Labels: data/qa/labels.jsonl (routes: direct for general, needs-RAG for rag )

Usage (from repo root):
  python -m scripts.gen_synth_data --general 100 --rag 200 ^
      --corpus-dir data/corpus/synth --out data/qa/synth_all.jsonl --seed 7

Then ingest the new corpus:
  python -m src.ingest_docs --corpus data/corpus/synth --collection kb_main ^
      --model intfloat/e5-small --batch-size 128
"""
from __future__ import annotations
 
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

# ------------------- helpers -------------------

def _slug(n: int) -> str:
    return f"{n:03d}"

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")  

def _jsonl_write(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------- general QA -------------------

_GENERAL_TERMS = [
    "RMSE", "MAE", "R-squared", "TF-IDF", "BM25", "cosine similarity",
    "logistic regression", "linear regression", "k-means", "DBSCAN",
    "PCA", "random forest", "gradient boosting", "ROC curve", "AUC",
    "precision", "recall", "F1 score", "confusion matrix", "cross-validation",
    "learning rate", "batch size", "epoch", "overfitting", "underfitting",
    "regularization", "dropout", "embedding", "tokenization", "attention",
    "transformer", "encoder", "decoder", "JSON", "YAML", "HTTP", "TLS",
    "DNS", "TCP", "UDP", "REST API", "GraphQL", "microservice", "load balancer",
    "cache", "CDN", "SQL", "NoSQL", "data lake", "data warehouse",
    "ETL", "ELT", "time series", "moving average", "stationarity", "ARIMA",
    "seasonality", "lag feature", "one-hot encoding", "standardization", "normalization",
    "Docker", "container", "Kubernetes", "orchestrator", "message queue", "pub/sub",
    "idempotency", "rate limiting", "retry policy", "circuit breaker", "blue/green deployment",
    "canary release", "feature flag", "observability", "tracing", "metrics",
    "logging", "SLA", "SLO", "SLI", "latency", "throughput", "availability",
    "consistency", "partition tolerance", "sharding", "replication", "leader election",
    "hashing", "Bloom filter", "LRU cache", "backpressure", "dead letter queue",
]

def _def_sentence(term: str) -> str:
    t = term.strip()
    base = {
        "RMSE": "Root Mean Square Error, a measure of prediction error.",
        "MAE": "Mean Absolute Error, the average absolute difference.",
        "R-squared": "Coefficient of determination indicating fit quality.",
        "TF-IDF": "Term weighting combining frequency and inverse document frequency.",
        "BM25": "A ranking function for document retrieval.",
    }
    if t in base:
        return base[t]
    lead = t if t.upper() == t else t.capitalize()
    return f"{lead} is a commonly used concept or technique."

def build_general_qas(n: int, rng: random.Random) -> List[Dict]:
    terms = list(_GENERAL_TERMS)
    rng.shuffle(terms)
    if n > len(terms):
        terms = (terms * ((n + len(terms) - 1) // len(terms)))[:n]
    else:
        terms = terms[:n]
    rows: List[Dict] = []
    for i, term in enumerate(terms, 1):
        qid = f"g{_slug(i)}"
        q = f"What is {term}?"
        a = _def_sentence(term)
        rows.append({"id": qid, "question": q, "answer": a, "domain": "general"})
    return rows

# ------------------- synthetic RAG corpora + QAs -------------------

_EMB_CHOICES = [
    "intfloat/e5-small", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2",
    "Alibaba-NLP/gte-small", "microsoft/mpnet-base",
]

_PROJECT_NAMES = [
    "Orion", "Lyra", "Vega", "Altair", "Sirius", "Nova", "Pulsar", "Quasar",
    "Nebula", "Cosmos", "Phoenix", "Draco", "Hydra", "Pegasus", "Aquila",
    "Cygnus", "Argo", "Perseus", "Andromeda", "Helios",
]

@dataclass
class Project:
    name: str
    version: str
    release: str
    chunk: int
    stride: int
    embedding: str
    collection: str
    metric: str

def synth_projects(k: int, rng: random.Random) -> List[Project]:
    names = list(_PROJECT_NAMES)
    if k > len(names):
        names += [f"Project{n}" for n in range(len(names) + 1, k + 1)]
    projs: List[Project] = []
    for i in range(k):
        nm = names[i]
        ver = f"{1 + (i % 3)}.{(i * 2) % 10}"
        rel = f"2024-{(i % 12)+1:02d}-{(10 + i) % 28 + 1:02d}"
        chunk = rng.choice([256, 384, 512, 640])
        stride = rng.choice([64, 96, 128, 160])
        emb = rng.choice(_EMB_CHOICES)
        coll = f"kb_{nm.lower()}"
        metric = rng.choice(["cosine", "dot", "l2"]) if emb != "intfloat/e5-small" else "cosine"
        projs.append(Project(nm, ver, rel, chunk, stride, emb, coll, metric))
    return projs

def project_doc_text(p: Project) -> str:
    return (
        f"Project {p.name} — Release {p.version} ({p.release})\n\n"
        f"Chunking: length {p.chunk} tokens, stride {p.stride} tokens.\n"
        f"Embedding model: {p.embedding}.\n"
        f"Vector store: Chroma collection '{p.collection}' with metric '{p.metric}'.\n"
        f"Notes: This document describes {p.name}'s retrieval defaults and deployment parameters on CPU.\n"
    )

def build_rag_corpus_and_qas(corpus_dir: Path, n_qas: int, rng: random.Random) -> Tuple[List[Path], List[Dict]]:
    projs = synth_projects(24, rng)
    docs_created: List[Path] = []
    qas: List[Dict] = []
    cnt = 0

    for idx, p in enumerate(projs, 1):
        # Write doc
        fname = f"synth_{_slug(idx)}_{p.name.lower()}.txt"
        path = corpus_dir / fname
        _write_text(path, project_doc_text(p))
        docs_created.append(path)

        # 10 QAs per project
        qa_templates = [
            (f"What is the chunk length used by Project {p.name}?", f"{p.chunk} tokens."),
            (f"What stride does Project {p.name} use for chunking?", f"{p.stride} tokens."),
            (f"Which embedding model is configured for Project {p.name}?", p.embedding + "."),
            (f"What Chroma collection does Project {p.name} use?", f"{p.collection}."),
            (f"What similarity metric is set for Project {p.name}?", f"{p.metric}."),
            (f"What is the release date of Project {p.name} {p.version}?", f"{p.release}."),
            (f"Which release version is documented for Project {p.name}?", f"{p.version}."),
            (f"Is Project {p.name} described as CPU-based deployment?", "Yes, CPU-based."),
            (f"What does the documentation of Project {p.name} cover?", "Retrieval defaults and deployment parameters."),
            (f"What is the default retrieval store for Project {p.name}?", "Chroma."),
        ]
        for q, a in qa_templates:
            cnt += 1
            qas.append({"id": f"r{_slug(cnt)}", "question": q, "answer": a, "domain": "synth", "source": str(path)})
            if cnt >= n_qas:
                return docs_created, qas

    return docs_created, qas

# ------------------- main -------------------

def main():
    ap = argparse.ArgumentParser("Generate synthetic QA and corpus for router training")
    ap.add_argument("--general", type=int, default=100, help="Number of general QAs")
    ap.add_argument("--rag", type=int, default=200, help="Number of RAG QAs")
    ap.add_argument("--corpus-dir", default="data/corpus/synth", help="Output corpus directory")
    ap.add_argument("--out", default="data/qa/synth_all.jsonl", help="Output QA JSONL path")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus_dir = Path(args.corpus_dir)
    _ensure_dir(corpus_dir)

    general = build_general_qas(args.general, rng)
    _, rag = build_rag_corpus_and_qas(corpus_dir, args.rag, rng)

    rows = general + rag
    # sanity
    assert len(general) == args.general, "general count mismatch"
    assert len(rag) == args.rag, "rag count mismatch"
    assert all(r.get("id") and r.get("question") and r.get("answer") for r in rows), "empty fields"
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"

    # shuffle to avoid order bias
    rng.shuffle(rows)

    out = Path(args.out)
    _jsonl_write(out, rows)

    # write labels.jsonl next to out
    labels_path = out.with_name("labels.jsonl")
    lab_rows = [
        {"id": r["id"], "question": r["question"], "route": ("direct" if r.get("domain") == "general" else "needs-RAG")}
        for r in rows
    ]
    _jsonl_write(labels_path, lab_rows)

    print(json.dumps({
        "qa_total": len(rows),
        "qa_general": sum(1 for r in rows if r["domain"] == "general"),
        "qa_rag": sum(1 for r in rows if r["domain"] != "general"),
        "corpus_dir": str(corpus_dir),
        "out": str(out),
        "labels": str(labels_path),
        "seed": args.seed,
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
