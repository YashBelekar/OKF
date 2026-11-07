"""Central configuration.

Every path can be relocated by setting the OKF_HOME environment variable,
which is also how the test-suite isolates itself from real data.
"""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(os.environ.get("OKF_HOME", CODE_ROOT)).resolve()

RAW_DOCUMENTS_DIR = PROJECT_ROOT / "raw_documents"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "okf_assistant.db"

# --- Ollama / LLM -------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_TEMPERATURE = 0.1
OLLAMA_MAX_TOKENS = 512
OLLAMA_CONTEXT_TOKENS = 4096
OLLAMA_TIMEOUT_SECONDS = 300

# --- Embeddings ---------------------------------------------------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Cosine similarity below this is treated as "not relevant" (MiniLM scale).
MIN_SEMANTIC_SCORE = 0.25

# --- Chunking -----------------------------------------------------------
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150

# --- Retrieval ----------------------------------------------------------
RETRIEVAL_LIMIT = 5
RRF_K = 60

# --- Concept extraction -------------------------------------------------
MAX_CHUNKS_FOR_CONCEPTS = 40       # per document, keeps CPU-only runs sane
CONCEPT_CHUNK_CHARS = 1500

# --- Uploads ------------------------------------------------------------
MAX_UPLOAD_SIZE_MB = 100

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".csv",
    ".xlsx", ".pptx", ".json", ".html", ".htm",
}
