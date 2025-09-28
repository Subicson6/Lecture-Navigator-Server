# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Database ---
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
MONGODB_DB: str = os.getenv("MONGODB_DB", "vidseek_db")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "transcripts")

# Vector DB switch: 'mongodb' (default) or 'pinecone'
VECTOR_DB: str = os.getenv("VECTOR_DB", "mongodb").lower()

# Pinecone settings (used when VECTOR_DB=pincone)
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "lecturenav-index")
# Optional: serverless spec for auto-index creation
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")  # aws|gcp|azure
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

# --- AI Models ---
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
# Preferred default Gemini model
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")
FORCE_LOCAL_EMBEDDINGS: bool = os.getenv("FORCE_LOCAL_EMBEDDINGS", "false").lower() in {"1", "true"}

# Default embedding model aligned to test.py base concept
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")


def _infer_embedding_dim(model_name: str) -> int:
    name = (model_name or "").lower()
    if "bge-small" in name:
        return 384
    if "bge-base" in name:
        return 768
    if "bge-large" in name:
        return 1024
    # Sensible default if unknown
    return 768


# Allow explicit override via env; otherwise infer from EMBEDDING_MODEL_NAME
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", str(_infer_embedding_dim(EMBEDDING_MODEL_NAME))))

# --- Latency/quality knobs ---
# Keep retrieval snappy for UI (<2s target)
MAX_K: int = int(os.getenv("MAX_K", "3"))
ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "false").lower() in {"1", "true"}
FAST_MODE: bool = os.getenv("FAST_MODE", "false").lower() in {"1", "true"}
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "8.0"))

# --- External transcript service (youtube-transcript.io) ---
YOUTUBE_TRANSCRIPT_API_URL: str = os.getenv(
    "YOUTUBE_TRANSCRIPT_API_URL",
    "https://www.youtube-transcript.io/api/transcripts",
)

# Raw value from env; may be either:
# - The full header value like: "Basic <token>"
# - Or just the token value: "<token>"
YOUTUBE_TRANSCRIPT_API_KEY: str = os.getenv("YOUTUBE_TRANSCRIPT_API_KEY", "Basic 68d55ed4172ff538f9e26db8")


def _normalize_basic_auth(value: str) -> str:
    """Return a proper Authorization header value with 'Basic <token>'.

    Accepts either 'Basic <token>' or just '<token>' from env, trims whitespace,
    and normalizes the prefix to exactly 'Basic '. Returns empty string if input is empty.
    """
    v = (value or "").strip()
    if not v:
        return ""
    lower = v.lower()
    if lower.startswith("basic "):
        token = v.split(None, 1)[1].strip()
        return f"Basic {token}"
    if lower.startswith("basic"):
        token = v[5:].strip()
        return f"Basic {token}" if token else ""
    return f"Basic {v}"


# Normalized Authorization header value used by HTTP requests
YOUTUBE_TRANSCRIPT_AUTH_HEADER: str = _normalize_basic_auth(YOUTUBE_TRANSCRIPT_API_KEY)