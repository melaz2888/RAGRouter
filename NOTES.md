# RAGRouter - Project Notes

Personal notes and explanations for understanding the project.

---

## LATEST UPDATE (February 2026)

### Current Architecture

RAGRouter is a smart Q&A system that decides whether to answer directly (using LLM knowledge) or retrieve context first (RAG).

```
                           USER QUESTION
                                 |
                                 v
                    +------------------------+
                    |     FastAPI (/ask)     |
                    +------------------------+
                                 |
                                 v
+------------------------------------------------------------------------+
|                            ROUTER                                       |
|  +------------------+     +-------------------+                         |
|  | Keyword Check    | --> | Similarity Check  |                         |
|  | (keywords.txt)   |     | (ChromaDB query)  |                         |
|  +------------------+     +-------------------+                         |
|         |                        |                                      |
|      found?                 score >= 0.84?                              |
|         |                        |                                      |
|        YES                  YES     NO                                  |
|         |                    |       |                                  |
|         v                    v       v                                  |
|       "rag"               "rag"   "direct"                              |
+------------------------------------------------------------------------+
                     |                    |
                     v                    v
        +------------------+    +------------------+
        |   RETRIEVER      |    |   GENERATOR      |
        |  - Embed query   |    |  - Direct prompt |
        |  - ChromaDB top5 |    |  - No context    |
        |  - Return chunks |    |  - Fast response |
        +------------------+    +------------------+
                     |
                     v
        +------------------+
        |   GENERATOR      |
        |  - RAG prompt    |
        |  - With context  |
        |  - Accurate      |
        +------------------+
                     |
                     v
                  ANSWER
```

### Project Structure

```
RAGRouter/
|
+-- src/                    # Core application code
|   +-- config.py           # All settings (ports, models, thresholds)
|   +-- router.py           # Routing logic (keywords + similarity)
|   +-- retriever.py        # Embedding, chunking, ChromaDB operations
|   +-- generator.py        # LLM calls (Ollama)
|   +-- api.py              # FastAPI endpoints
|
+-- scripts/                # CLI utilities
|   +-- ingest.py           # Ingest documents into ChromaDB
|   +-- test_routing.py     # Test routing accuracy
|
+-- data/
|   +-- corpus/dynamiqs/docs/   # Knowledge base (22 markdown files)
|   +-- keywords.txt            # Domain keywords for fast routing
|   +-- test_questions.jsonl    # Test set (24 questions)
|
+-- docker/                 # Docker compose for Ollama + ChromaDB
+-- ui/                     # Streamlit interface
```

### How Routing Works

1. **Question arrives** at `/ask` endpoint
2. **Keyword check**: If question contains domain keywords -> RAG
3. **Similarity check**: Embed question, find closest chunk in ChromaDB
   - Score >= 0.84 -> RAG (domain question)
   - Score < 0.84 -> Direct (general knowledge)
4. **Generate answer** using appropriate path

### Key Numbers

| Metric | Value |
|--------|-------|
| Similarity threshold | 0.84 |
| Domain questions score | 0.84 - 0.93 |
| General questions score | 0.77 - 0.83 |
| Test accuracy | 100% (24 questions) |
| Corpus size | 22 files, 93 chunks |

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
data/corpus/*.txt, *.md
       |
       v
+-------------+
|  Chunking   |  Split docs into 512-token pieces (128 overlap)
|  (tiktoken) |
+------+------+
       |
       v
+-------------+
|  Embedding  |  Convert each chunk to a 384-dim vector
|  (e5-small) |  Uses prefix: "passage: {text}"
+------+------+
       |
       v
+-------------+
|  ChromaDB   |  Store vectors + original text
|  (upsert)   |  Now searchable by similarity
+-------------+
```

### Key Point
You ingest ONCE, query MANY times. That's why it's separate from the API.

### Command
```powershell
python scripts/ingest.py --corpus data/corpus/dynamiqs/docs
```

---

## Router: How We Decide Direct vs RAG

### The Problem
Every RAG system retrieves documents for EVERY query, even when the LLM already knows the answer. This wastes time.

### The Solution
A "meta-model" (router) that decides BEFORE calling retrieval:
- **direct**: LLM answers without context (fast)
- **rag**: Retrieve documents first, then answer (accurate)

### The Approach: Similarity-Based Routing

We use TWO methods combined:

```
Question arrives
       |
       v
+------------------+
|  KEYWORD CHECK   |  <- Rule-based (manual keywords.txt)
+--------+---------+
         |
    +----+----+
    |         |
 found    not found
    |         |
    v         v
  "rag"   +------------------+
          | SIMILARITY CHECK |  <- Embed query, compare to corpus
          +--------+---------+
                   |
             +-----+-----+
             |           |
          >= 0.84     < 0.84
             |           |
             v           v
           "rag"     "direct"
```

| Method | How it works | When used |
|--------|--------------|-----------|
| Keywords | If question contains domain terms -> rag | First check, always wins |
| Similarity | Embed query, find closest chunk in corpus | Only if no keyword match |

### Key Insight: Domain-Specific Knowledge

The router works by detecting whether a question is about the **domain** (what's in your corpus) or **general knowledge** (what the LLM already knows).

- **Domain questions** (about dynamiqs, your specific docs) -> similarity score 0.84-0.93 -> RAG
- **General questions** (capitals, history, math) -> similarity score 0.77-0.83 -> direct

The threshold (0.84) cleanly separates these two categories.

---

## Failed Approaches (What We Tried Before)

We tried several classification approaches before settling on similarity-based routing. Here's why each failed:

### v1: Token Overlap (50% threshold)
- **Logic**: Compare LLM answer to ground truth, if >=50% token overlap -> "direct"
- **Problem**: Too strict. 98% of questions routed to RAG, only 2% direct.
- **Why**: Exact token matching is too rigid. LLM can give correct answers with different wording.

### v2: Semantic Similarity (0.7 threshold)
- **Logic**: Use embedding similarity instead of token overlap
- **Problem**: Too lenient. 100% routed to direct, 0% to RAG.
- **Why**: Threshold was wrong, and semantic similarity between any two texts is often high.

### v3: Unsupervised Clustering (K-means)
- **Logic**: Cluster question embeddings, let algorithm find natural groups
- **Problem**: No ground truth to validate. Got 39% direct / 61% rag but couldn't verify correctness.
- **Why**: Without labels, can't evaluate if the clustering is meaningful.

### v4: LLM-as-Judge
- **Logic**: Ask the small LLM to verify if its own answer is correct
- **Problem**: Small model (qwen2.5:0.5b) unreliable as judge. Marked 95% as correct when only ~25% were.
- **Why**: Small models don't have good self-evaluation capabilities.

### The Real Problem

**We were training on NQ (real-world questions) but testing on domain-specific corpus.**

Natural Questions = random Wikipedia facts.
Our corpus = dynamiqs documentation.

These distributions don't match! A router trained on "Who is the president?" won't help with "How to simulate a Kerr oscillator?"

### The Solution

Stop trying to classify questions universally. Instead:
1. Pick a domain (dynamiqs documentation)
2. If question is similar to corpus -> RAG (domain question)
3. If question is NOT similar -> direct (general knowledge)

This is simpler and actually works.

---

## Current Corpus: dynamiqs

We use dynamiqs documentation as our knowledge base.

### Setup
```powershell
# Clone only docs folder (sparse checkout)
git clone --filter=blob:none --sparse https://github.com/dynamiqs/dynamiqs.git data/corpus/dynamiqs
cd data/corpus/dynamiqs
git sparse-checkout set docs

# Ingest into ChromaDB
python scripts/ingest.py --corpus data/corpus/dynamiqs/docs
```

### Stats
- 22 markdown files
- 93 chunks after splitting
- Topics: quantum simulation, Lindblad equations, solvers, JAX integration

### Test Results
With threshold 0.84:
- Domain questions (dynamiqs): similarity 0.84-0.93 -> routed to RAG
- General questions (capitals, math): similarity 0.77-0.83 -> routed to direct
- **Accuracy: 100%** on test set of 24 questions

---

## Testing the Router

### Test File
`data/test_questions.jsonl` - 24 questions (12 domain, 12 general)

### Run Test
```powershell
python scripts/test_routing.py --threshold 0.84
```

### Example Output
```
OK [domain] score=0.927 -> rag (expected rag)
   How do I simulate a lossy system with a jump operator...
OK [general] score=0.776 -> direct (expected direct)
   What is the capital of France?
```

---

## File Structure

```
src/
+-- config.py       # All settings (ports, paths, thresholds)
+-- router.py       # Keyword check + similarity-based routing
+-- retriever.py    # Embedding + ChromaDB + chunking
+-- generator.py    # LLM calls (direct + RAG prompts)
+-- api.py          # FastAPI /ask endpoint

scripts/
+-- ingest.py       # CLI: ingest documents into ChromaDB
+-- test_routing.py # CLI: test similarity-based routing

data/
+-- corpus/
|   +-- dynamiqs/docs/  # dynamiqs documentation (22 files)
+-- keywords.txt        # Domain keywords (one per line)
+-- test_questions.jsonl # Test set (24 questions)
```

---

## Quick Reference Commands

```powershell
# Start services
docker compose -f docker/docker-compose.yml up -d
docker exec -it ollama ollama pull qwen2.5:0.5b-instruct

# Ingest documents (dynamiqs docs)
python scripts/ingest.py --corpus data/corpus/dynamiqs/docs

# Test routing accuracy
python scripts/test_routing.py --threshold 0.84

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
| Router | Similarity threshold (0.84) | No training needed, domain-agnostic logic |
| Chunking | 512 tokens, 128 overlap | Fits model context, preserves continuity |

---

## Key Takeaways

1. **Don't over-engineer the router.** A simple similarity check works better than complex classifiers.

2. **Match your training data to your corpus.** Training on NQ and testing on domain docs is a distribution mismatch.

3. **Domain-specific routing is the key.** The question isn't "does the LLM know this?" but "is this question about our domain?"

4. **Small LLMs can't self-evaluate.** Don't use them as judges for their own answers.

5. **Threshold tuning is empirical.** Test with real domain vs general questions to find the right cutoff.

---

*Last updated: February 2026*
