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
KEYWORDS_FILE = DATA_DIR / "keywords.txt"

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

# Similarity-based routing threshold
# If question similarity to corpus >= threshold -> rag (domain question)
# If question similarity to corpus < threshold -> direct (general question)
ROUTER_SIMILARITY_THRESHOLD = 0.84

# =============================================================================
# API
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8008
