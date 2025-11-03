import uuid
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover - fallback if urllib3 location changes
    try:
        # Older import path
        from urllib3.util import Retry  # type: ignore
    except Exception:  # If unavailable, we will rely on manual retries
        Retry = None  # type: ignore

from fastapi import UploadFile
from pymongo import MongoClient

from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.schema import Document
# Removed: from llama_index.core.node_parser import SentenceSplitter # No longer needed for default transformations
# NOTE: Avoid llama_index YoutubeTranscriptReader due to incompatibilities with
# youtube-transcript-api versions in some environments. We fetch transcripts
# directly via youtube-transcript-api with a compatibility shim below.
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

from . import config, db
from .models import TimestampSearchResult, QASnippet


def initialize_models() -> None:
    """Configure embeddings and parsing. LLM is handled via direct providers (OpenRouter/Gemini)."""
    # Embedding model (HuggingFace local or CPU/GPU backed)
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_NAME)
    logging.info(f"Embedding model configured: {config.EMBEDDING_MODEL_NAME}")

    # We do not initialize a LlamaIndex LLM. Answer synthesis uses direct OpenRouter/Gemini clients.
    Settings.llm = None
    if not (config.OPENROUTER_API_KEY or config.GOOGLE_API_KEY):
        logging.warning("No OPENROUTER_API_KEY or GOOGLE_API_KEY set. LLM features will be disabled.")

    # Default chunking
    # --- MODIFICATION START (1/3) ---
    # Disable default LlamaIndex SentenceSplitter, as we're using custom time-based chunking.
    Settings.transformations = []
    # --- MODIFICATION END (1/3) ---

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


# --- MODIFICATION START (2/3) ---
# New function for time-based chunking
def chunk_by_time(
        segments: List[dict],
        video_id: str,
        raw_id: str,
        title: str,
        source: str = "youtube",
        window_seconds: float = 30.0,
        overlap_seconds: float = 5.0,
) -> List[Document]:
    """
    Chunks transcript segments into fixed-size time windows with overlap.

    Args:
        segments: The list of raw transcript segments from the API.
        video_id: The unique ID for the video.
        raw_id: The original YouTube video ID.
        title: The video title.
        source: The source of the video (e.g., 'youtube').
        window_seconds: The duration of each chunk in seconds.
        overlap_seconds: The duration of the overlap between consecutive chunks.

    Returns:
        A list of LlamaIndex Document objects representing the time-based chunks.
    """
    if not segments:
        return []

    # Calculate the effective video duration based on the last segment
    video_duration = 0.0
    if segments:
        last_seg = segments[-1]
        video_duration = float(last_seg.get("end", last_seg.get("start", 0.0) + last_seg.get("duration", 0.0)))

    time_chunks: List[Document] = []
    current_time = 0.0
    step = window_seconds - overlap_seconds

    # Ensure step is positive to avoid infinite loop if window_seconds <= overlap_seconds
    if step <= 0:
        logging.warning(
            f"Window size ({window_seconds}s) is not greater than overlap ({overlap_seconds}s). Setting step to window_seconds.")
        step = window_seconds  # Fallback to no overlap if misconfigured

    while current_time < video_duration + overlap_seconds:  # Loop slightly beyond to catch last few seconds
        window_start = max(0.0, current_time)  # Ensure window_start is not negative
        window_end = current_time + window_seconds

        # Find all segments that overlap with the current time window
        texts_in_window = []
        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start + seg.get("duration", 0.0)))

            # Check for any overlap between the segment and the window
            if seg_start < window_end and seg_end > window_start:
                texts_in_window.append(seg.get("text", "").strip())

        # If there's content, create a document for this time chunk
        if texts_in_window:
            chunk_text = " ".join(texts_in_window).strip()

            if chunk_text:  # Only create a chunk if there's actual text
                # Create the metadata for this specific time window
                meta = {
                    "video_id": video_id,
                    "raw_video_id": raw_id,
                    "source": source,
                    "t_start": window_start,
                    "t_end": window_end,
                    "title": title,
                }
                time_chunks.append(Document(text=chunk_text, metadata=meta))

        # Move to the next window start time
        current_time += step

    return time_chunks


# --- MODIFICATION END (2/3) ---


def process_and_store_video(youtube_url: str) -> str:
    """Ingest a YouTube video into MongoDB Atlas vector store via LlamaIndex with timing logs."""
    import time

    total_start = time.perf_counter()

    # ----------------------------
    # 1️⃣ Extract raw YouTube ID
    # ----------------------------
    id_start = time.perf_counter()
    if "watch?v=" in youtube_url:
        raw_id = youtube_url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        raw_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format.")
    id_end = time.perf_counter()
    logging.info(f"[TIMING] Extract YouTube ID: {id_end - id_start:.2f}s")

    # ----------------------------
    # 2️⃣ Idempotent ingest check
    # ----------------------------
    vc = db.get_videos_collection()
    video_id = None
    # -------------------
    # MODIFICATION 1 of 4
    # -------------------
    # Changed `if vc:` to `if vc is not None:`
    if vc is not None:
        try:
            found = vc.find_one({"raw_id": raw_id}, {"video_id": 1})
            if found and found.get("video_id"):
                logging.info("Video already ingested. Reusing video_id.")
                return str(found["video_id"])
        except Exception:
            pass
    video_id = str(uuid.uuid4())

    # ----------------------------
    # 3️⃣ Fetch transcript segments
    # ----------------------------
    t_start = time.perf_counter()
    segments: List[dict]
    try:
        if config.YOUTUBE_TRANSCRIPT_AUTH_HEADER:
            try:
                segments = _fetch_transcript_via_api(raw_id)
            except Exception as e:
                logging.warning(f"youtube-transcript.io failed ({e}); falling back to youtube_transcript_api")
                segments = _fetch_youtube_transcript(raw_id, languages=["en"])
        else:
            segments = _fetch_youtube_transcript(raw_id, languages=["en"])
    except Exception as e:
        logging.error(f"Transcript fetch failed: {e}")
        segments = []
    t_end = time.perf_counter()
    logging.info(f"[TIMING] Transcript fetch: {t_end - t_start:.2f}s")

    # --- MODIFICATION START (3/3) ---
    # ----------------------------
    # 4️⃣ Build Time-Based Chunks
    # ----------------------------
    d_start = time.perf_counter()

    # Attempt to fetch video title from YouTube oEmbed if not already found.
    # This ensures a meaningful title for metadata.
    video_title = f"YouTube Video {raw_id}"
    try:
        oembed = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={raw_id}", "format": "json"},
            timeout=2.0,
        )
        if oembed.ok:
            j = oembed.json()
            video_title = j.get("title", video_title)
    except Exception as e:
        logging.warning(f"Could not fetch YouTube oEmbed title for {raw_id}: {e}")

    # Use the new time-based chunking function
    docs = chunk_by_time(
        segments=segments,
        video_id=video_id,
        raw_id=raw_id,
        title=video_title,  # Pass the fetched title
        window_seconds=30.0,  # You can make these configurable via config.py
        overlap_seconds=5.0,  # You can make these configurable via config.py
    )

    d_end = time.perf_counter()
    logging.info(f"[TIMING] Time-based chunking: {d_end - d_start:.2f}s (chunks={len(docs)})")

    # --- MODIFICATION END (3/3) ---

    # ----------------------------
    # 5️⃣ Vector store embedding
    # ----------------------------
    v_start = time.perf_counter()
    if not config.MONGODB_URI:
        raise ConnectionError("MONGODB_URI not set.")
    client = MongoClient(config.MONGODB_URI)
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=config.MONGODB_DB,
        collection_name=config.MONGODB_COLLECTION,
        vector_index_name="hybrid_index",
        text_key="text",
        embedding_key="embedding",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        docs,  # Now 'docs' contains the time-based chunks
        storage_context=storage_context,
        transformations=Settings.transformations,  # This is now an empty list, so no further splitting
    )
    v_end = time.perf_counter()
    logging.info(f"[TIMING] Embedding & MongoDB store: {v_end - v_start:.2f}s")

    # ----------------------------
    # 6️⃣ Persist metadata
    # ----------------------------
    m_start = time.perf_counter()
    # -------------------
    # MODIFICATION 2 of 4
    # -------------------
    # Changed `if vc:` to `if vc is not None:`
    if vc is not None:
        try:
            # Re-fetch title/thumbnail if not already done during chunking to ensure consistency
            # or if title could not be fetched during chunking.
            final_title, thumbnail = video_title, None
            try:
                oembed = requests.get(
                    "https://www.youtube.com/oembed",
                    params={"url": f"https://www.youtube.com/watch?v={raw_id}", "format": "json"},
                    timeout=2.0,
                )
                if oembed.ok:
                    j = oembed.json()
                    final_title = j.get("title", final_title)
                    thumbnail = j.get("thumbnail_url")
            except Exception:
                pass

            vc.update_one(
                {"raw_id": raw_id},
                {
                    "$set": {
                        "raw_id": raw_id,
                        "video_id": video_id,
                        "source": "youtube",
                        "url": f"https://www.youtube.com/watch?v={raw_id}",
                        "title": final_title,
                        "thumbnail_url": thumbnail,
                    },
                    "$setOnInsert": {"created_at": __import__("datetime").datetime.utcnow()},
                },
                upsert=True,
            )
        except Exception:
            pass
    m_end = time.perf_counter()
    logging.info(f"[TIMING] Persist metadata: {m_end - m_start:.2f}s")

    total_end = time.perf_counter()
    logging.info(f"[TIMING] Total ingestion for video_id={video_id}: {total_end - total_start:.2f}s\n")

    return video_id


def transcribe_youtube(youtube_url: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Return video_id, transcript_text, and segments for a YouTube URL using the API-first approach.

    This does not persist anything by itself. Use process_and_store_video to index.
    """
    # Extract raw YouTube ID
    if "watch?v=" in youtube_url:
        raw_id = youtube_url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        raw_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format.")

    # Prefer API
    if not config.YOUTUBE_TRANSCRIPT_AUTH_HEADER:
        # Fallback to local library
        segments = _fetch_youtube_transcript(raw_id, languages=["en"])  # type: ignore
    else:
        segments = _fetch_transcript_via_api(raw_id)

    # Compose text
    transcript_text = " ".join([(s.get("text") or "").strip() for s in segments if s.get("text")])
    video_id = str(uuid.uuid4())
    return video_id, transcript_text, segments


def purge_video_embeddings(video_id: str) -> int:
    """Delete all indexed segments for the given video_id from the Mongo collection."""
    collection = db.get_collection()
    if collection is None:
        raise ConnectionError("DB not initialized.")
    res = collection.delete_many({"metadata.video_id": video_id})
    return int(getattr(res, "deleted_count", 0))


def _index_segments_for_video(video_id: str, raw_id: str, source: str, segments: List[dict]) -> None:
    """Index provided segments under an existing video_id (used for re-embedding)."""
    # --- MODIFICATION START (Added title for chunking) ---
    # Attempt to fetch video title from YouTube oEmbed for proper chunking metadata.
    video_title = f"YouTube Video {raw_id}"
    if source == "youtube":
        try:
            oembed = requests.get(
                "https://www.youtube.com/oembed",
                params={"url": f"https://www.youtube.com/watch?v={raw_id}", "format": "json"},
                timeout=2.0,
            )
            if oembed.ok:
                j = oembed.json()
                video_title = j.get("title", video_title)
        except Exception as e:
            logging.warning(f"Could not fetch YouTube oEmbed title for {raw_id} in _index_segments_for_video: {e}")
    # --- MODIFICATION END ---

    # --- MODIFICATION START (Use time-based chunking here too) ---
    docs = chunk_by_time(
        segments=segments,
        video_id=video_id,
        raw_id=raw_id,
        title=video_title,
        source=source,
        window_seconds=30.0,
        overlap_seconds=5.0,
    )
    # --- MODIFICATION END ---

    if not docs:
        return

    if not config.MONGODB_URI:
        raise ConnectionError("MONGODB_URI not set.")
    client = MongoClient(config.MONGODB_URI)
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=config.MONGODB_DB,
        collection_name=config.MONGODB_COLLECTION,
        vector_index_name="hybrid_index",
        text_key="text",
        embedding_key="embedding",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=Settings.transformations,  # Now empty
    )


def reembed_video_by_id(video_id: str) -> str:
    """Re-embed an already known video using current embedding model and index settings.

    Steps:
    - Look up raw_id and source from videos collection.
    - Fetch transcript segments.
    - Purge existing vectors for this video_id.
    - Index segments under the same video_id.
    """
    vc = db.get_videos_collection()
    if vc is None:
        raise ConnectionError("DB not initialized.")
    vdoc = vc.find_one({"video_id": video_id}, {"raw_id": 1, "source": 1})
    if not vdoc or not vdoc.get("raw_id"):
        raise ValueError("Unknown video_id or missing raw_id")
    raw_id = str(vdoc["raw_id"])
    source = (vdoc.get("source") or "youtube").lower()

    # Fetch segments
    if source == "youtube":
        if config.YOUTUBE_TRANSCRIPT_AUTH_HEADER:
            try:
                segments = _fetch_transcript_via_api(raw_id)
            except Exception as e:
                logging.warning(f"youtube-transcript.io failed during reembed ({e}); falling back")
                segments = _fetch_youtube_transcript(raw_id, languages=["en"])  # type: ignore
        else:
            segments = _fetch_youtube_transcript(raw_id, languages=["en"])  # type: ignore
    else:
        # For non-YouTube, we currently do not support fetch by id
        raise ValueError("Re-embed is currently supported only for YouTube sources")

    # Purge and re-index
    purge_video_embeddings(video_id)
    _index_segments_for_video(video_id, raw_id, source, segments)
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
    raw_docs = reader.load_data()

    # Extract segments-like structure from raw_docs for time-based chunking
    srt_segments: List[dict] = []
    video_title = file.filename  # Use filename as title for SRT
    total_duration = 0.0

    if raw_docs:
        # Assuming SimpleDirectoryReader for SRT produces one document per logical block
        # with metadata.start/end if available, or we'll infer.
        for i, d in enumerate(raw_docs):
            text = (d.text or "").strip()
            if not text:
                continue
            # Attempt to extract start/end from metadata if available (e.g., from LlamaIndex's SRT reader)
            # Otherwise, use a dummy duration if not present for basic processing
            start_time = float(d.metadata.get("start", 0.0))
            end_time = float(d.metadata.get("end", start_time + 5.0))  # Default to 5s if end not found
            duration = end_time - start_time
            if duration <= 0:  # Ensure positive duration
                duration = 5.0
                end_time = start_time + duration

            srt_segments.append({
                "text": text,
                "start": start_time,
                "end": end_time,
                "duration": duration,
            })
            total_duration = max(total_duration, end_time)  # Track max duration

    # --- MODIFICATION START (Use time-based chunking for SRT files too) ---
    docs = chunk_by_time(
        segments=srt_segments,  # Use the parsed segments
        video_id=video_id,
        raw_id=file.filename,  # Use filename as raw_id for SRT
        title=video_title,
        source="srt",
        window_seconds=30.0,
        overlap_seconds=5.0,
    )
    # --- MODIFICATION END ---

    # Vector store
    if not config.MONGODB_URI:
        raise ConnectionError("MONGODB_URI not set.")
    client = MongoClient(config.MONGODB_URI)
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=config.MONGODB_DB,
        collection_name=config.MONGODB_COLLECTION,
        vector_index_name="hybrid_index",
        text_key="text",
        embedding_key="embedding",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        docs,  # Now 'docs' contains the time-based chunks
        storage_context=storage_context,
        transformations=Settings.transformations,  # This is now an empty list
    )

    # Also persist metadata for SRT files
    vc = db.get_videos_collection()
    if vc is not None:
        try:
            vc.update_one(
                {"raw_id": file.filename},  # Use filename as raw_id for SRT files
                {
                    "$set": {
                        "raw_id": file.filename,
                        "video_id": video_id,
                        "source": "srt",
                        "url": None,  # No external URL for uploaded SRT
                        "title": video_title,
                        "thumbnail_url": None,
                        "duration": total_duration,  # Store calculated duration
                    },
                    "$setOnInsert": {"created_at": __import__("datetime").datetime.utcnow()},
                },
                upsert=True,
            )
        except Exception:
            pass

    return video_id


def perform_hybrid_search(video_id: str, query: str, k: int, window_seconds: float | None = None) -> List[
    TimestampSearchResult]:
    """Hybrid search using MongoDB Atlas search + embeddings, with optional re-rank.

    If window_seconds is provided, enrich the returned snippets by fetching ~window_seconds of
    transcript around each top hit and set t_end = t_start + window_seconds for display.
    """
    if Settings.embed_model is None:
        raise ConnectionError("Embedding model not initialized.")

    # Enforce low k to keep latency <2s
    try:
        k = max(1, min(int(k), int(config.MAX_K)))
    except Exception:
        k = config.MAX_K

    # Fetch raw_id and source for URL building (if available)
    raw_id: str | None = None
    video_source: str = "unknown"
    try:
        vc = db.get_videos_collection()
        if vc is not None:
            vdoc = vc.find_one({"video_id": video_id}, {"raw_id": 1, "source": 1})
            # -------------------
            # MODIFICATION 3 of 4
            # -------------------
            # Changed `if vdoc:` to `if vdoc is not None:`
            if vdoc is not None:
                rid = vdoc.get("raw_id")
                raw_id = str(rid) if rid is not None else None
                video_source = str(vdoc.get("source", "unknown"))
    except Exception:
        pass

    # Mongo path only
    collection = db.get_collection()
    if collection is None:
        raise ConnectionError("DB not initialized.")

    query_vector = Settings.embed_model.get_query_embedding(query)
    pipeline = [
        {
            "$search": {
                "index": "hybrid_index",
                "knnBeta": {"vector": query_vector, "path": "embedding", "k": max(k * 20, 50)}
            }
        },
        {"$match": {"metadata.video_id": video_id}},
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
    try:
        raw = list(collection.aggregate(pipeline))
    except Exception as e:
        # Fallback for clusters that disallow knnBeta composition: use text-only and post-filter by video_id
        logging.warning(f"Primary hybrid pipeline failed ({e}); falling back to text-only search.")
        pipeline = [
            {"$search": {"index": "hybrid_index", "text": {"query": query, "path": ["text", "metadata.title"]}}},
            {"$match": {"metadata.video_id": video_id}},
            {"$project": {"_id": 0, "text": 1, "metadata": 1, "score": {"$meta": "searchScore"}}},
            {"$limit": max(k * 20, 50)}
        ]
        raw = list(collection.aggregate(pipeline))
    candidates: List[dict] = []
    for r in raw:
        m = r.get("metadata", {}) or {}
        text = r.get("text", "") or ""
        # Robust snippet: prefer text, but if empty, try minimal placeholder
        snippet = (text or "").strip() or f"Segment at ~{float(m.get('t_start', 0.0)):.0f}s"

        # Construct deep link URL when raw_id available and source is youtube
        ts = float(m.get("t_start", 0.0))
        te = float(m.get("t_end", 0.0))
        # Ensure non-zero window in responses, using the actual chunk end if available
        if te <= ts:
            # If t_end is not meaningful, infer from window_seconds or default
            te = ts + (window_seconds if window_seconds is not None else 30.0)

        url_val = None
        if raw_id and video_source == "youtube":
            url_val = f"https://www.youtube.com/watch?v={raw_id}&t={int(round(ts))}s"

        candidates.append({
            "title": m.get("title"),
            "t_start": ts,
            "t_end": te,
            "snippet": snippet[:280],
            "score": float(r.get("score", 0.0)),
            "url": url_val,
            "_full_text": text,
        })

    # Cross-encoder re-ranking
    ce = _get_cross_encoder() if config.ENABLE_RERANK else None
    if ce and candidates:
        pairs = [(query, c["_full_text"]) for c in candidates]
        try:
            ce_scores = ce.predict(pairs)
            for c, s in zip(candidates, ce_scores):
                c["score"] = float(s)
        except Exception as e:
            logging.warning(f"Cross-encoder scoring failed, falling back to base scores: {e}")

    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    topk = candidates[:k]

    # Optionally replace snippets with ~window_seconds window text and normalize
    def _normalize_text(s: str) -> str:
        s = (s or "").replace("\n", " ")
        while "  " in s:
            s = s.replace("  ", " ")
        return s.strip()

    if window_seconds and window_seconds > 0:
        for c in topk:
            try:
                # _fetch_window_text_mongo already considers a window around t_start
                expanded = _fetch_window_text_mongo(video_id, c["t_start"], window_seconds=float(window_seconds))
                if expanded.strip():
                    c["snippet"] = _normalize_text(expanded)[:800]
                # The t_end for display should reflect the actual chunk's end or expanded window
                # With time-based chunking, c["t_end"] already holds the window end.
                # If you specifically want to show a fixed-size window around the start for ALL snippets,
                # you can uncomment the line below. Otherwise, the chunk's own t_end is often preferred.
                # c["t_end"] = float(c["t_start"]) + float(window_seconds)
            except Exception:
                pass

    return [TimestampSearchResult(
        t_start=c["t_start"],
        t_end=c["t_end"],
        title=c.get("title"),
        snippet=c["snippet"],
        score=c["score"],
        url=c.get("url"),
    ) for c in topk]


def _llm_complete(prompt: str) -> str:
    """Call OpenRouter (preferred) or Gemini API directly and return text."""
    # 1) OpenRouter (OpenAI-compatible chat API)
    if config.OPENROUTER_API_KEY:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            # OpenRouter expects model slugs like "google/gemini-1.5-flash"
            model = os.getenv(
                "OPENROUTER_MODEL",
                "google/" + (config.LLM_MODEL_NAME or "gemini-1.5-flash").replace("models/", ""),
            )

            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            # Optional metadata headers recommended by OpenRouter
            if getattr(config, "OPENROUTER_SITE_URL", ""):
                headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
            if getattr(config, "OPENROUTER_APP_NAME", ""):
                headers["X-Title"] = config.OPENROUTER_APP_NAME

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You summarize transcript excerpts into a single, concise answer."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }

            # Build a session with retry-on-transient failures
            session = requests.Session()
            adapter: HTTPAdapter
            if Retry is not None and int(getattr(config, "LLM_MAX_RETRIES", 0)) > 0:
                retry = Retry(
                    total=int(config.LLM_MAX_RETRIES),
                    read=int(config.LLM_MAX_RETRIES),
                    connect=int(config.LLM_MAX_RETRIES),
                    backoff_factor=float(config.LLM_RETRY_BACKOFF),
                    status_forcelist=[408, 429, 500, 502, 503, 504],
                    allowed_methods=frozenset(["POST"]),
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(max_retries=retry)
            else:
                adapter = HTTPAdapter()
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            # Manual retry loop to also cover read timeouts
            attempts = int(getattr(config, "LLM_MAX_RETRIES", 0)) + 1
            last_err: Exception | None = None
            for i in range(max(1, attempts)):
                try:
                    resp = session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=(float(config.LLM_CONNECT_TIMEOUT), float(config.LLM_READ_TIMEOUT)),
                    )
                    if 200 <= resp.status_code < 300:
                        data = resp.json()
                        return (
                            (data.get("choices") or [{}])[0]
                            .get("message", {})
                            .get("content", "")
                            .strip()
                        )
                    # Retry on specific status codes
                    if resp.status_code in {408, 429, 500, 502, 503, 504} and i < attempts - 1:
                        sleep_s = float(config.LLM_RETRY_BACKOFF) * (2 ** i)
                        time.sleep(min(sleep_s, 5.0))
                        continue
                    # Non-retriable HTTP error
                    resp.raise_for_status()
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as te:
                    last_err = te
                    if i < attempts - 1:
                        sleep_s = float(config.LLM_RETRY_BACKOFF) * (2 ** i)
                        time.sleep(min(sleep_s, 5.0))
                        continue
                    raise
                except requests.exceptions.RequestException as re:
                    last_err = re
                    # For request exceptions that are not retriable, break
                    raise
            # If loop exits without return, raise last error if any
            if last_err:
                raise last_err
            # Fallback guard
            raise RuntimeError("OpenRouter call failed for unknown reasons")
        except Exception as e:
            logging.error(f"OpenRouter call failed: {e}")

    # 2) Google Gemini via google-generativeai
    if config.GOOGLE_API_KEY:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=config.GOOGLE_API_KEY)
            model_name = (config.LLM_MODEL_NAME or "gemini-1.5-flash").replace("models/", "")
            model = genai.GenerativeModel(model_name)
            # Use fine-grained read timeout; SDK does not support separate connect/read, so pass overall
            overall_timeout = float(getattr(config, "LLM_READ_TIMEOUT", config.LLM_TIMEOUT_SECONDS))
            resp = model.generate_content(prompt, request_options={"timeout": overall_timeout})
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            # Fallback: concatenate parts
            parts = getattr(resp, "candidates", None) or []
            if parts:
                return str(parts[0]).strip()
        except Exception as e:
            logging.error(f"Gemini call failed: {e}")

    return "LLM not configured. Unable to provide a summary."


def _fetch_window_text_mongo(video_id: str, center_s: float, window_seconds: float = 30.0,
                             max_chars: int = 1500) -> str:
    """Fetch ~window_seconds of transcript text around a center timestamp for a given video.

    Assumes MongoDB Atlas vector store; collects overlapping transcript segments in the window.
    """
    try:
        collection = db.get_collection()
        if collection is None:
            return ""
        half = max(0.0, float(window_seconds) / 2.0)
        start = max(0.0, float(center_s) - half)
        end = float(center_s) + half
        cursor = collection.find(
            {
                "metadata.video_id": video_id,
                # overlap any segment intersecting the window
                "metadata.t_start": {"$lte": end},
                "metadata.t_end": {"$gte": start},
            },
            {"text": 1, "metadata.t_start": 1, "_id": 0},
        ).sort("metadata.t_start", 1)
        parts: list[str] = []
        total = 0
        for doc in cursor:
            t = (doc.get("text") or "").strip()
            if not t:
                continue
            parts.append(t)
            total += len(t)
            if total >= max_chars:
                break
        return " ".join(parts)[:max_chars]
    except Exception:
        return ""


def synthesize_answer(query: str, results: List[TimestampSearchResult], video_id: str | None = None,
                      window_seconds: float = 30.0) -> str:
    """Use OpenRouter or Gemini API to synthesize an answer from retrieved sources.

    If possible, expand each result into ~window_seconds of transcript around its timestamp
    to provide more context to the LLM.
    """
    # If FAST_MODE is on but snippets are empty, fall back to LLM
    if config.FAST_MODE:
        joined = " ".join([r.snippet for r in results if (r.snippet or "").strip()])
        if joined.strip():
            return joined[:800]

    # Build context; expand to ~30s windows when possible; if still empty, use a placeholder
    def _safe(sn: str) -> str:
        s = (sn or "").strip()
        return s if s else "[no snippet available]"

    context_blocks: list[str] = []
    for r in results:
        expanded = ""
        if video_id:
            expanded = _fetch_window_text_mongo(video_id, r.t_start, window_seconds=float(window_seconds))
        use_text = expanded.strip() if expanded.strip() else _safe(r.snippet)
        context_blocks.append(f"Timestamp: {r.t_start:.0f}s\nContent: {use_text}")
    context = "\n---\n".join(context_blocks)

    prompt = f"""
    User Query: "{query}"

    Based on the following video transcript excerpts, provide a concise, direct answer to the user's query.
    Do not add any preamble like "Based on the transcript...".
    If the context is insufficient to answer the query, say: "I cannot find the answer in the provided text."

    Context:
    {context}

    Answer:
    """

    try:
        return _llm_complete(prompt)
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
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


# --- MVP helpers ---
def _extract_raw_youtube_id(youtube_url: str) -> str:
    if "watch?v=" in youtube_url:
        return youtube_url.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format.")


def _timestamp_url(raw_id: str, seconds: float) -> str:
    # Use watch?v format to ensure &t= works across clients
    return f"https://www.youtube.com/watch?v={raw_id}&t={int(round(seconds))}s"


def qa_from_url(youtube_url: str, query: str, k: int, window_seconds: float = 30.0) -> tuple[str, List[QASnippet]]:
    """Ingest the URL (idempotent enough for MVP), run hybrid search, build timestamped links, and summarize."""
    raw_id = _extract_raw_youtube_id(youtube_url)

    # Ingest (for MVP we always ingest; consider caching in future)
    video_id = process_and_store_video(youtube_url)

    # Search (expand snippets to window_seconds for display)
    results = perform_hybrid_search(video_id, query, k, window_seconds=window_seconds)

    # Build links
    enriched: List[QASnippet] = []
    # Fetch source from DB to correctly build URLs
    video_source: str = "youtube"  # Assume youtube if raw_id exists
    try:
        vc = db.get_videos_collection()
        if vc is not None:
            vdoc = vc.find_one({"video_id": video_id}, {"source": 1})
            if vdoc:
                video_source = str(vdoc.get("source", "youtube"))
    except Exception:
        pass

    for r in results:
        url = r.url
        if not url and raw_id and video_source == "youtube":
            url = _timestamp_url(raw_id, r.t_start)
        enriched.append(QASnippet(
            t_start=r.t_start,
            t_end=r.t_end,
            title=r.title,
            snippet=r.snippet,
            score=r.score,
            url=url or "",
        ))

    # Summarize with expanded windows
    answer = synthesize_answer(query, results, video_id=video_id, window_seconds=window_seconds)
    return answer, enriched


def qa_from_video_id(video_id: str, query: str, k: int, window_seconds: float = 30.0) -> tuple[str, List[QASnippet]]:
    """Run retrieval on an already ingested video (by video_id) and return timestamped links + summary."""
    # Lookup raw_id and source (for deep links) if available
    raw_id: str | None = None
    video_source: str = "unknown"
    try:
        vc = db.get_videos_collection()
        if vc is not None:
            vdoc = vc.find_one({"video_id": video_id}, {"raw_id": 1, "source": 1})
            # -------------------
            # MODIFICATION 4 of 4
            # -------------------
            # Changed `if vdoc:` to `if vdoc is not None:`
            if vdoc is not None:
                rid = vdoc.get("raw_id")
                raw_id = str(rid) if rid is not None else None
                video_source = str(vdoc.get("source", "unknown"))
    except Exception:
        pass

    # Search (expand snippets to window_seconds for display)
    results = perform_hybrid_search(video_id, query, k, window_seconds=window_seconds)

    # Build links
    enriched: List[QASnippet] = []
    for r in results:
        url = r.url
        if not url and raw_id and video_source == "youtube":  # Only create YouTube URLs if source is YouTube
            url = _timestamp_url(raw_id, r.t_start)
        enriched.append(QASnippet(
            t_start=r.t_start,
            t_end=r.t_end,
            title=r.title,
            snippet=r.snippet,
            score=r.score,
            url=url or "",
        ))

    # Summarize with expanded windows
    answer = synthesize_answer(query, results, video_id=video_id, window_seconds=window_seconds)
    return answer, enriched


# ----------------------------------------------------
# Internal helpers for YouTube transcript compatibility
# ----------------------------------------------------
def _fetch_youtube_transcript(raw_id: str, languages: List[str]) -> List[dict]:
    """Fetch transcript entries for a YouTube video with broad library compatibility.

    Returns list of entries like: {"text": str, "start": float, "duration": float}
    """
    try:
        # Import at call-time to avoid import errors during app startup if package missing
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "youtube-transcript-api is required to fetch YouTube transcripts."
        ) from e

    # Try common APIs across versions
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        # Typical API in 0.6.x
        return YouTubeTranscriptApi.get_transcript(raw_id, languages=languages)  # type: ignore

    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        # Alternative flow: pick matching language if available, else best-effort
        try:
            tl = YouTubeTranscriptApi.list_transcripts(raw_id)  # type: ignore
            # First try direct language match
            try:
                tr = tl.find_transcript(languages)
                return tr.fetch()
            except Exception:
                pass
            # Try translating first available to requested language
            for tr in tl:
                try:
                    return tr.translate(languages[0]).fetch()
                except Exception:
                    continue
            # Finally, fetch the first available as-is
            for tr in tl:
                try:
                    return tr.fetch()
                except Exception:
                    continue
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve transcripts for video {raw_id}: {e}") from e

    # If we reach here, the installed package is incompatible with expected APIs
    raise RuntimeError(
        "Installed youtube-transcript-api does not expose get_transcript or list_transcripts. "
        "Please pin youtube-transcript-api to a compatible version (e.g., 0.6.1/0.6.2)."
    )


def _fetch_transcript_via_api(raw_id: str) -> List[dict]:
    """Fetch transcript from youtube-transcript.io using the raw video ID.

    Expected response shape (simplified):
    [
      {
        "tracks": [
          {
            "transcript": [
              {"text": str, "start": float, "duration": float, ...}, ...
            ]
          }
        ]
      }
    ]
    """
    if not config.YOUTUBE_TRANSCRIPT_API_URL:
        raise RuntimeError("YOUTUBE_TRANSCRIPT_API_URL not configured")
    if not config.YOUTUBE_TRANSCRIPT_AUTH_HEADER:
        raise RuntimeError("YOUTUBE_TRANSCRIPT_API_KEY not configured")

    headers = {
        "Authorization": config.YOUTUBE_TRANSCRIPT_AUTH_HEADER,
        "Content-Type": "application/json",
    }
    payload = {"ids": [raw_id]}

    try:
        resp = requests.post(config.YOUTUBE_TRANSCRIPT_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise RuntimeError(f"Transcript API HTTP error: {http_err} - {getattr(resp, 'text', '')}") from http_err
    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"Transcript API request error: {req_err}") from req_err

    data = resp.json()
    if not (isinstance(data, list) and data and isinstance(data[0], dict)):
        raise RuntimeError("Unexpected transcript API response format")

    tracks = data[0].get("tracks") or []
    if not (isinstance(tracks, list) and tracks):
        raise RuntimeError("Transcript API response missing tracks")

    transcript = tracks[0].get("transcript") or []
    if not isinstance(transcript, list):
        raise RuntimeError("Transcript API response missing transcript list")

    # Normalize fields and compute end
    norm: List[dict] = []
    for seg in transcript:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        duration = seg.get("duration")
        end = float(seg.get("end") or (start + float(duration or 0.0)))
        norm.append({"text": text, "start": start, "end": end, "duration": float(duration or (end - start))})

    return norm