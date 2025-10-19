# RAG Router (CPU-only) — Skeleton

Objectif: Router `Direct` vs `RAG` pour réduire la latence tout en préservant la qualité.
Stack CPU: Ollama (Phi-3-mini ou Qwen2.5-1.5B quantisés), ChromaDB, sentence-transformers, scikit-learn, FastAPI, Streamlit.

## Prérequis
- Git, Python 3.11, pip
- Docker Desktop (ou Docker Engine)
- VS Code (recommandé: extensions Python, Docker)

## Démarrage rapide (Day 1 — MVP fonctionnel)
1) Cloner ce repo (ou extraire le zip) puis créer l’environnement Python:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Lancer l’infra (Ollama + Chroma) en CPU:
```bash
docker compose -f docker/docker-compose.yml up -d
# Récupérer un SLM léger (une seule fois) dans le conteneur Ollama:
docker exec -it ollama ollama pull phi3:mini-128k-instruct-q4_K_M
```

3) Tester l’API locale (réponse Direct sans retrieval):
```bash
# Terminal 1
uvicorn src.service_api:app --port 8008 --reload
# Terminal 2
python - << 'PY'
import requests;print(requests.post("http://localhost:8008/ask",json={"question":"What is RMSE?"}).json())
PY
```

4) Lancer la démo Streamlit:
```bash
streamlit run ui/app.py
```

5) (Optionnel) Pousser sur GitHub:
```bash
git init
git add .
git commit -m "RAG Router CPU skeleton (Day 1 MVP)"
git branch -M main
# Crée un repo sur GitHub puis:
git remote add origin <URL_DU_REPO>
git push -u origin main
```

## Prochaines étapes
- Ajouter vos documents dans `data/corpus/` puis indexer (à implémenter dans `src/ingest_docs.py`).
- Compléter `src/features.py`, `src/label_offline.py`, `src/train_router.py` pour entraîner le routeur.
- Implémenter le retrieval dans `src/retriever.py` et le RAG dans `src/answer_rag.py`.
- Mesurer avec `src/eval_metrics.py` et un script d’agrégat dans `scripts/eval_report.sh`.

Voir `README.md` pour les commandes clés et la structure.
