# RAGRouter - Project Notes

Personal notes and explanations for understanding the project.

---

## Git Basics

### Local vs Remote
- **Local branch**: exists only on your computer
- **Remote branch**: exists on GitHub (called `origin`)

### Upstream / Tracking
- When you create a branch locally, GitHub doesn't know about it
- "Upstream" means: which remote branch should your local branch sync with
- Once set, `git push` and `git pull` know where to go automatically

### Pushing a New Branch
```powershell
# First time (creates branch on GitHub + sets tracking)
git push -u origin branch-name

# After that, just:
git push
```

---

## What is Ingestion?

**Ingestion = preparing your knowledge base BEFORE any user asks questions**

It's the "loading" step - like putting books on library shelves before anyone comes to search.

### The Flow
```
data/corpus/*.txt
       │
       ▼
┌─────────────┐
│  Chunking   │  Split docs into 512-token pieces (128 overlap)
│  (tiktoken) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embedding  │  Convert each chunk to a 384-dim vector
│  (e5-small) │  Uses prefix: "passage: {text}"
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ChromaDB   │  Store vectors + original text
│  (upsert)   │  Now searchable by similarity
└─────────────┘
```

### Key Point
You ingest ONCE, query MANY times. That's why it's separate from the API.

### Command
```powershell
python scripts/ingest.py --corpus data/corpus/synth
```

---

## Router: How We Decide Direct vs RAG

### The Problem
Every RAG system retrieves documents for EVERY query, even when the LLM already knows the answer. This wastes time.

### The Solution
A "meta-model" (router) that decides BEFORE calling retrieval:
- **direct**: LLM answers without context (fast)
- **rag**: Retrieve documents first, then answer (accurate)

### The Hybrid Approach

We use TWO methods combined:

```
Question arrives
       │
       ▼
┌──────────────────┐
│  KEYWORD CHECK   │  ← Rule-based (manual keywords.txt)
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
 found    not found
    │         │
    ▼         ▼
  "rag"   ┌──────────────────┐
          │   CLASSIFIER     │  ← ML-based (trained model)
          └────────┬─────────┘
                   │
             ┌─────┴─────┐
             ▼           ▼
         "direct"      "rag"
```

| Method | How it works | When used |
|--------|--------------|-----------|
| Keywords | If question contains domain terms → rag | First check, always wins |
| Classifier | ML predicts based on question features | Only if no keyword match |

---

## Router Training Pipeline

### Step 1: Generate Labels (scripts/label.py)

For each question in Natural Questions dataset:

1. Ask LLM the question (NO context, NO retrieval)
2. Compare LLM answer to ground truth
3. Assign label:
   - Match (≥50% token overlap) → "direct" (LLM already knows)
   - No match → "rag" (LLM needs help)

```powershell
python scripts/label.py --dataset nq --limit 500
```

Output: `data/labels.jsonl`

### Step 2: Train Classifier (scripts/train.py)

Train a LogisticRegression on query-only features:

| Feature | Description |
|---------|-------------|
| `len_words` | Word count |
| `len_chars` | Character count |
| `has_number` | Contains digits (0/1) |
| `wh_type` | what/who/where/when/why/how/other |
| `num_entities` | Count of capitalized words |
| `has_question_mark` | Ends with ? (0/1) |

```powershell
python scripts/train.py
```

Output: `models/router.joblib`

### Key Insight
All features are derived from the question text ONLY. No retrieval needed during routing = fast decision.

---

## File Structure

```
src/
├── config.py       # All settings (ports, paths, model names)
├── router.py       # Keyword check + classifier
├── retriever.py    # Embedding + ChromaDB + chunking
├── generator.py    # LLM calls (direct + RAG prompts)
└── api.py          # FastAPI /ask endpoint

scripts/
├── ingest.py       # CLI: ingest documents into ChromaDB
├── label.py        # CLI: generate training labels
└── train.py        # CLI: train the router classifier

models/
├── router.joblib   # Trained classifier
└── tfidf.joblib    # (optional) corpus vocabulary

data/
├── corpus/         # Documents to ingest
├── keywords.txt    # Domain keywords (one per line)
└── labels.jsonl    # Training labels
```

---

## Quick Reference Commands

```powershell
# Start services
docker compose -f docker/docker-compose.yml up -d
docker exec -it ollama ollama pull qwen2.5:0.5b-instruct

# Ingest documents
python scripts/ingest.py --corpus data/corpus/synth

# Train router (optional - requires labeling first)
python scripts/label.py --dataset nq --limit 500
python scripts/train.py

# Run API
uvicorn src.api:app --port 8008 --reload

# Run UI
python -m streamlit run ui/app.py
```

---

## Technical Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | qwen2.5:0.5b-instruct | Small, fast, CPU-friendly |
| Embeddings | intfloat/e5-small | Good quality, requires prefixes |
| Vector DB | ChromaDB | Simple, HTTP client, cosine similarity |
| Classifier | LogisticRegression | Fast, interpretable, works with small data |
| Chunking | 512 tokens, 128 overlap | Fits model context, preserves continuity |

---

*Last updated: January 2026*
