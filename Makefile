PY=python
ACT=source .venv/bin/activate || .venv\Scripts\activate

format:
\tblack . && isort . || true

serve:
\tuvicorn src.service_api:app --port 8008 --reload

ui:
\tstreamlit run ui/app.py

up:
\tdocker compose -f docker/docker-compose.yml up -d

pull-model:
\tdocker exec -it ollama ollama pull phi3:mini-128k-instruct-q4_K_M

test-api:
\t$(PY) - << 'PY'\nimport requests;print(requests.post(\"http://localhost:8008/ask\",json={\"question\":\"What is RMSE?\"}).json())\nPY
