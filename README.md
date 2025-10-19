# RAGRouter

RAGRouter is a lightweight, CPU-friendly framework that routes user queries between direct LLM answering and retrieval-augmented generation (RAG) to reduce latency without losing accuracy.

---

## Overview

Modern RAG systems answer every query with retrieval, even when the model already knows the answer. RAGRouter introduces a compact classifier that decides, per query, whether to:

* Answer directly using the local small-language model (SLM), or
* Invoke retrieval through a vector database (Chroma) and re-ranked context passages.

This routing mechanism cuts end-to-end latency and compute cost while preserving response quality.

---

## Architecture

```
User Query
   │
   ▼
[ Router (LogReg/XGBoost) ]
   ├──► Direct → Qwen2.5-0.5B → Answer
   └──► RAG → Chroma → Re-Rank → Qwen2.5-0.5B → Answer + Sources
```

* Inference model: `qwen2.5:0.5b-instruct` (CPU via Ollama)
* Vector store: ChromaDB (CPU)
* Embeddings: `intfloat/e5-small` (sentence-transformers)
* Router: logistic regression (scikit-learn)
* Interface: FastAPI backend + Streamlit web UI
* Deploys: fully on CPU via Docker Compose

---

## Key Features

* Latency-aware: automatically bypasses retrieval for simple queries.
* Faithful RAG: includes only retrieved context when required.
* Lightweight: runs locally on CPU; no GPU dependency.
* Modular: API-first design (FastAPI service, Streamlit demo).
* Dockerized: ollama + chroma + API stack.

---

## Quick Start (CPU)

```bash
# clone
git clone https://github.com/melaz2888/RAGRouter.git
cd RAGRouter

# create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell

pip install -r requirements.txt

# start backend (Ollama + Chroma)
docker compose -f docker/docker-compose.yml up -d
docker exec -it ollama ollama pull qwen2.5:0.5b-instruct

# launch API
uvicorn src.service_api:app --port 8008 --reload

# optional: run Streamlit demo
python -m streamlit run ui/app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## API Usage

**Endpoint**

```
POST /ask
{
  "question": "What is RMSE?"
}
```

**Response**

```json
{
  "route": "direct",
  "answer": "RMSE (Root Mean Square Error) ...",
  "passages": [],
  "timing_ms": 1420
}
```

---

## Model and Data

The released router is pre-trained on a balanced mix of:

* FiQA (financial QA)
* HotpotQA (multi-hop reasoning)
* Energy-domain documents (internal)

Labels {direct, needs-RAG} were generated via heuristics and manual validation. Inference and evaluation are entirely CPU-based.

---

## Roadmap

* Core routing pipeline (done)
* Lightweight re-ranking (MMR)
* Evaluation dashboard (faithfulness, latency, bypass percentage)
* Optional GPU build for larger LLMs
* Enterprise plugin API (planned)

---

## License

Apache 2.0

---

## Contributing

Contributions are welcome via pull requests for:

* Vector index optimizations
* Additional small LLM backends
* Evaluation utilities and test datasets

---

## Citation

If you use RAGRouter in your research or product, please cite:

```
@software{RAGRouter2025,
  author = {El Azhar, Mohammed},
  title  = {RAGRouter: Latency-aware Query Routing for RAG Systems},
  year   = {2025},
  url    = {https://github.com/melaz2888/RAGRouter}
}
```
