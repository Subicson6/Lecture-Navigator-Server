# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Database ---
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
MONGODB_DB: str = os.getenv("MONGODB_DB", "vidseek_db")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "transcripts")

# --- AI Models ---
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-latest")
FORCE_LOCAL_EMBEDDINGS: bool = os.getenv("FORCE_LOCAL_EMBEDDINGS", "false").lower() in {"1", "true"}

# Local fallback model if no API keys are present
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")