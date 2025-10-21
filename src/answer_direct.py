import time, requests
from .config import OLLAMA_HOST, OLLAMA_MODEL
from .rag_types import DirectAnswer

def answer_direct(question: str, temperature: float = 0.2) -> DirectAnswer:
    t0 = time.time()
    prompt = f"""Answer concisely and factually.
Q: {question}
A:"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        # "options": {"temperature": temperature, "num_ctx": 4096},
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    ans = r.json().get("response", "").strip()
    return DirectAnswer(answer=ans, timing_ms=int((time.time() - t0) * 1000))
