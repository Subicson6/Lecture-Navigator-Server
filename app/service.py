import uuid
import logging
import os
from pathlib import Path
from typing import List

from fastapi import UploadFile
from pymongo import MongoClient

from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from sentence_transformers import CrossEncoder

from . import config, db
from .models import TimestampSearchResult


def initialize_models() -> None:
    """Configure LlamaIndex global Settings with LLM + embedding + node parser."""
    # Embedding model (HuggingFace local or CPU/GPU backed)
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_NAME)
    logging.info(f"Embedding model configured: {config.EMBEDDING_MODEL_NAME}")

    # LLM (Gemini via google-generativeai)
    if config.GOOGLE_API_KEY:
        Settings.llm = Gemini(model=config.LLM_MODEL_NAME, api_key=config.GOOGLE_API_KEY)
        logging.info(f"Gemini LLM configured: {config.LLM_MODEL_NAME}")
    else:
        Settings.llm = None
        logging.warning("GOOGLE_API_KEY not set. LLM features will be disabled.")

    # Default chunking
    Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=50)]

    # Cross-encoder for re-ranking (lazy init in getter)
    logging.info("Service models initialized.")


_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder | None:
    global _cross_encoder
    if _cross_encoder is None:
        try:
            model_name = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            _cross_encoder = CrossEncoder(model_name)
            logging.info(f"Cross-encoder loaded: {model_name}")
        except Exception as e:
            logging.warning(f"Failed to load cross-encoder: {e}")
            _cross_encoder = None
    return _cross_encoder


def process_and_store_video(youtube_url: str) -> str:
    """Ingest a YouTube video into MongoDB Atlas vector store via LlamaIndex."""
    # Extract raw YouTube ID
    if "watch?v=" in youtube_url:
        raw_id = youtube_url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        raw_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format.")

    video_id = str(uuid.uuid4())

    # 1) Reader
    reader = YoutubeTranscriptReader()
    docs = reader.load_data(video_ids=[raw_id], languages=["en"])  # list[Document]
    for d in docs:
        meta = {**(d.metadata or {})}
        # normalize timestamps if present from the transcript reader
        start = meta.get("start") or meta.get("t_start") or 0.0
        duration = meta.get("duration")
        end = meta.get("end") or (start + (duration or 0.0))
        meta.update({
            "video_id": video_id,
            "raw_video_id": raw_id,
            "source": "youtube",
            "t_start": float(start) if start is not None else 0.0,
            "t_end": float(end) if end is not None else 0.0,
            "title": meta.get("title") or f"YouTube Video {raw_id}",
        })
        d.metadata = meta

    # 2) Vector store (MongoDB Atlas Vector Search)
    if not config.MONGODB_URI:
        raise ConnectionError("MONGODB_URI not set.")
    client = MongoClient(config.MONGODB_URI)
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=config.MONGODB_DB,
        collection_name=config.MONGODB_COLLECTION,
        index_name="hybrid_index",
        text_key="text",
        embedding_key="embedding",
    )

    # 3) Storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4) ServiceContext is represented via global Settings (embed_model + transformations)

    # 5) Build index (handles chunking, embedding, storing)
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=Settings.transformations,
    )

    return video_id


async def process_and_store_srt(file: UploadFile) -> str:
    """Ingest an uploaded .srt file and index it similarly to YouTube ingestion."""
    if not file.filename or not file.filename.lower().endswith(".srt"):
        raise ValueError("Only .srt files are supported")

    video_id = str(uuid.uuid4())

    # Save to data/uploads
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    local_path = uploads_dir / f"{video_id}.srt"
    content = await file.read()
    with open(local_path, "wb") as f:
        f.write(content)

    # Read via LlamaIndex
    reader = SimpleDirectoryReader(input_files=[str(local_path)])
    docs = reader.load_data()
    for d in docs:
        meta = {**(d.metadata or {})}
        meta.update({
            "video_id": video_id,
            "source": "srt",
            "title": meta.get("file_name") or file.filename,
            # Timestamps might not be preserved; default to 0s
            "t_start": float(meta.get("t_start", 0.0)),
            "t_end": float(meta.get("t_end", 0.0)),
        })
        d.metadata = meta

    # Vector store
    if not config.MONGODB_URI:
        raise ConnectionError("MONGODB_URI not set.")
    client = MongoClient(config.MONGODB_URI)
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=config.MONGODB_DB,
        collection_name=config.MONGODB_COLLECTION,
        index_name="hybrid_index",
        text_key="text",
        embedding_key="embedding",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=Settings.transformations,
    )

    return video_id


def perform_hybrid_search(video_id: str, query: str, k: int) -> List[TimestampSearchResult]:
    """True hybrid search using Atlas $search compound (text + knn) with re-ranking."""
    collection = db.get_collection()
    if not collection or not Settings.embed_model:
        raise ConnectionError("DB or embedding model not initialized.")

    query_vector = Settings.embed_model.get_query_embedding(query)

    pipeline = [
        {
            "$search": {
                "index": "hybrid_index",
                "compound": {
                    "should": [
                        {"text": {"query": query, "path": ["text", "metadata.title"]}},
                        {"knnBeta": {"vector": query_vector, "path": "embedding", "k": max(k * 20, 50)}},
                    ],
                    "filter": [
                        {"equals": {"path": "metadata.video_id", "value": video_id}}
                    ],
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "searchScore"},
            }
        },
        {"$limit": max(k * 20, 50)}
    ]
    raw = list(collection.aggregate(pipeline))

    # Build candidates
    candidates: List[dict] = []
    for r in raw:
        m = r.get("metadata", {}) or {}
        text = r.get("text", "") or ""
        candidates.append({
            "title": m.get("title"),
            "t_start": float(m.get("t_start", 0.0)),
            "t_end": float(m.get("t_end", 0.0)),
            "snippet": text[:280],
            "score": float(r.get("score", 0.0)),
            "_full_text": text,  # keep for re-ranker scoring
        })

    # Cross-encoder re-ranking
    ce = _get_cross_encoder()
    if ce and candidates:
        pairs = [(query, c["_full_text"]) for c in candidates]
        try:
            ce_scores = ce.predict(pairs)
            for c, s in zip(candidates, ce_scores):
                c["score"] = float(s)
        except Exception as e:
            logging.warning(f"Cross-encoder scoring failed, falling back to searchScore: {e}")

    # Sort and trim top-k
    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    topk = candidates[:k]

    return [TimestampSearchResult(
        t_start=c["t_start"],
        t_end=c["t_end"],
        title=c.get("title"),
        snippet=c["snippet"],
        score=c["score"],
    ) for c in topk]


def synthesize_answer_with_gemini(query: str, results: List[TimestampSearchResult]) -> str:
    """Use LlamaIndex LLM (Gemini) to synthesize an answer from retrieved sources."""
    if not Settings.llm:
        return "LLM not configured. Unable to provide a summary."

    context = "\n---\n".join(
        [f"Timestamp: {r.t_start:.0f}s\nContent: {r.snippet}" for r in results]
    )

    prompt = f"""
    User Query: "{query}"

    Based on the following video transcript excerpts, provide a concise, direct answer to the user's query.
    Do not add any preamble like "Based on the transcript...".
    If the context is insufficient to answer the query, state that you cannot find the answer in the provided text.

    Context:
    {context}

    Answer:
    """

    try:
        response = Settings.llm.complete(prompt)
        return getattr(response, "text", str(response)).strip()
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return "There was an error generating the answer."


def is_prompt_injection(query: str) -> bool:
    """Very simple keyword-based prompt injection guard."""
    suspicious = [
        "ignore previous instructions",
        "disregard previous instructions",
        "system prompt",
        "act as",
        "developer instructions",
        "change your rules",
        "override",
    ]
    q = (query or "").lower()
    return any(kw in q for kw in suspicious)