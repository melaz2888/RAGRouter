from pathlib import Path

# Ports
OLLAMA_HOST = "http://localhost:11434"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Models
OLLAMA_MODEL = "qwen2.5:0.5b-instruct"  # tinyllama:1.1b / llama3.2:1b
OLLAMA_NUM_PREDICT = 96
OLLAMA_NUM_CTX = 512

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
QA_DIR = DATA_DIR / "qa"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Retrieval
CHROMA_COLLECTION = "kb_main"
TOPK_DEFAULT = 6
K_CTX_DEFAULT = 3
