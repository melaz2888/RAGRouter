"""
RAGRouter v2 Configuration

All settings, paths, and constants in one place.
Other modules import from here — no hardcoded values elsewhere.
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
MODELS_DIR = ROOT / "models"
KEYWORDS_FILE = DATA_DIR / "keywords.txt"
LABELS_FILE = DATA_DIR / "labels.jsonl"

# Model artifacts
ROUTER_MODEL_PATH = MODELS_DIR / "router.joblib"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf.joblib"

# =============================================================================
# OLLAMA (LLM)
# =============================================================================

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:0.5b-instruct"
OLLAMA_TIMEOUT = 120  # seconds

# Generation limits
MAX_TOKENS_DIRECT = 128
MAX_TOKENS_RAG = 256

# =============================================================================
# CHROMADB (Vector Store)
# =============================================================================

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "kb_main"

# =============================================================================
# EMBEDDING MODEL
# =============================================================================

EMBEDDING_MODEL = "intfloat/e5-small"

# E5 models require these prefixes
EMBEDDING_PREFIX_QUERY = "query: "
EMBEDDING_PREFIX_PASSAGE = "passage: "

# =============================================================================
# CHUNKING
# =============================================================================

CHUNK_SIZE = 512       # tokens per chunk
CHUNK_OVERLAP = 128    # token overlap between chunks

# =============================================================================
# RETRIEVAL
# =============================================================================

RETRIEVAL_TOP_K = 5    # number of passages to retrieve
CONTEXT_TOP_K = 3      # number of passages to include in prompt

# =============================================================================
# ROUTER
# =============================================================================

# Default route when classifier is not available
DEFAULT_ROUTE = "rag"

# Classifier settings
ROUTER_TRAIN_TEST_SPLIT = 0.2
ROUTER_RANDOM_SEED = 42

# =============================================================================
# API
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8008
