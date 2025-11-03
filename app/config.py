import os
from dotenv import load_dotenv

# ----------------------------------------------------
# Load environment variables from .env file
# ----------------------------------------------------
load_dotenv()

# ----------------------------------------------------
# Core Models Configuration
# ----------------------------------------------------
# Local embedding model for creating vectors
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Cross-encoder model for re-ranking search results
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ----------------------------------------------------
# Database Configuration
# ----------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "lecture_navigator")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "video_embeddings")

# ----------------------------------------------------
# LLM & API Keys
# ----------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# The specific LLM to use via OpenRouter or Google
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")

# ----------------------------------------------------
# Search & Retrieval Configuration
# ----------------------------------------------------
MAX_K = int(os.getenv("MAX_K", "10"))
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "True").lower() in ("true", "1", "t")
FAST_MODE = os.getenv("FAST_MODE", "False").lower() in ("true", "1", "t")

# ----------------------------------------------------
# External Services
# ----------------------------------------------------
# YouTube Transcript API Configuration
YOUTUBE_TRANSCRIPT_API_KEY = os.getenv("YOUTUBE_TRANSCRIPT_API_KEY")
YOUTUBE_TRANSCRIPT_API_URL = os.getenv(
    "YOUTUBE_TRANSCRIPT_API_URL", "https://www.youtube-transcript.io/api/transcripts"
)

# Automatically prepare the Authorization header if API key exists
YOUTUBE_TRANSCRIPT_AUTH_HEADER = (
    f"Basic {YOUTUBE_TRANSCRIPT_API_KEY}"
    if YOUTUBE_TRANSCRIPT_API_KEY
    else None
)

# ----------------------------------------------------
# OpenRouter Analytics (optional)
# ----------------------------------------------------
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "VidSeek")

# ----------------------------------------------------
# LLM Network Settings (Advanced)
# ----------------------------------------------------
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "5.0"))
LLM_READ_TIMEOUT = float(os.getenv("LLM_READ_TIMEOUT", "30.0"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))

# ----------------------------------------------------
# Optional: Early Validation (helps during debugging)
# ----------------------------------------------------
def _validate_config():
    if not MONGODB_URI:
        print("⚠️ Warning: MONGODB_URI not found. Database operations may fail.")
    if not OPENROUTER_API_KEY and not GOOGLE_API_KEY:
        print("⚠️ Warning: No LLM API key found. Some query answering may fail.")
    if not YOUTUBE_TRANSCRIPT_API_KEY:
        print("⚠️ Warning: YouTube transcript API key missing. Transcript fetching may fail.")
    else:
        print("✅ YouTube transcript API key loaded successfully.")

# Run basic validation on import
_validate_config()
