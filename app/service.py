import uuid
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import requests

from fastapi import UploadFile
from pymongo import MongoClient

from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
# NOTE: Avoid llama_index YoutubeTranscriptReader due to incompatibilities with
# youtube-transcript-api versions in some environments. We fetch transcripts
# directly via youtube-transcript-api with a compatibility shim below.
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.vector_stores.pinecone import PineconeVectorStore  # type: ignore
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
    
    # Idempotent ingest: reuse existing video if present
    vc = db.get_videos_collection()
    video_id = None
    if vc is not None:
        try:
            found = vc.find_one({"raw_id": raw_id}, {"video_id": 1})
            if found and found.get("video_id"):
                return str(found["video_id"])  # already ingested
        except Exception:
            pass

    video_id = str(uuid.uuid4())

    # 1) Fetch transcript segments using youtube-transcript.io when available, else fallback
    segments: List[dict]
    if config.YOUTUBE_TRANSCRIPT_AUTH_HEADER:
        try:
            segments = _fetch_transcript_via_api(raw_id)
        except Exception as e:
            logging.warning(f"youtube-transcript.io failed ({e}); falling back to youtube_transcript_api")
            segments = _fetch_youtube_transcript(raw_id, languages=["en"])  # list[dict]
    else:
        segments = _fetch_youtube_transcript(raw_id, languages=["en"])  # list[dict]

    # Build Documents, 1 per segment, preserving timestamps for later linking
    docs: list[Document] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        duration = seg.get("duration")
        end = float(seg.get("end") or (start + float(duration or 0.0)))
        meta = {
            "video_id": video_id,
            "raw_video_id": raw_id,
            "source": "youtube",
            "t_start": start,
            "t_end": end,
            "title": f"YouTube Video {raw_id}",
        }
        docs.append(Document(text=text, metadata=meta))

    # 2) Vector store: MongoDB Atlas or Pinecone
    if config.VECTOR_DB == "pinecone":
        try:
            from pinecone import Pinecone as PineconeClient  # type: ignore
            pc = PineconeClient(api_key=config.PINECONE_API_KEY)
            index_name = config.PINECONE_INDEX_NAME
            # Create index if not exists (serverless)
            existing = {i["name"] for i in pc.list_indexes()}
            if index_name not in existing:
                pc.create_index(
                    name=index_name,
                    dimension=384,  # must match embedding size
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": config.PINECONE_CLOUD,
                            "region": config.PINECONE_REGION,
                        }
                    },
                )
            vector_store = PineconeVectorStore(
                pc.Index(index_name),
                namespace=str(video_id),  # isolate per ingested video
                text_key="text",
                metadata_keys=["video_id", "raw_video_id", "source", "t_start", "t_end", "title"],
            )
        except Exception as e:
            raise ConnectionError(f"Failed to init Pinecone: {e}")
    else:
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

    # Persist ingest mapping for reuse
    if vc is not None:
        try:
            # Optionally fetch basic metadata from YouTube oEmbed for title/thumbnail
            title = None
            thumbnail = None
            try:
                oembed = requests.get(
                    "https://www.youtube.com/oembed",
                    params={"url": f"https://www.youtube.com/watch?v={raw_id}", "format": "json"},
                    timeout=2.0,
                )
                if oembed.ok:
                    j = oembed.json()
                    title = j.get("title")
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
                        "title": title,
                        "thumbnail_url": thumbnail,
                    },
                    "$setOnInsert": {"created_at": __import__("datetime").datetime.utcnow()},
                },
                upsert=True,
            )
        except Exception:
            pass

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
    """Hybrid search: MongoDB Atlas pipeline or Pinecone similarity with optional re-rank."""
    if Settings.embed_model is None:
        raise ConnectionError("Embedding model not initialized.")

    # Enforce low k to keep latency <2s
    try:
        k = max(1, min(int(k), int(config.MAX_K)))
    except Exception:
        k = config.MAX_K

    # Fetch raw_id for URL building (if available)
    raw_id: str | None = None
    try:
        vc = db.get_videos_collection()
        if vc is not None:
            vdoc = vc.find_one({"video_id": video_id}, {"raw_id": 1})
            if vdoc:
                rid = vdoc.get("raw_id")
                raw_id = str(rid) if rid is not None else None
    except Exception:
        pass

    # Mongo path (existing behavior)
    if config.VECTOR_DB != "pinecone":
        collection = db.get_collection()
        if collection is None:
            raise ConnectionError("DB not initialized.")

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
        candidates: List[dict] = []
        for r in raw:
            m = r.get("metadata", {}) or {}
            text = r.get("text", "") or ""
            # Construct deep link URL when raw_id available
            ts = float(m.get("t_start", 0.0))
            url_val = f"https://www.youtube.com/watch?v={raw_id}&t={int(round(ts))}s" if raw_id else None
            candidates.append({
                "title": m.get("title"),
                "t_start": ts,
                "t_end": float(m.get("t_end", 0.0)),
                "snippet": text[:280],
                "score": float(r.get("score", 0.0)),
                "url": url_val,
                "_full_text": text,
            })
    else:
        # Pinecone similarity search using LlamaIndex API
        try:
            from pinecone import Pinecone as PineconeClient  # type: ignore
            pc = PineconeClient(api_key=config.PINECONE_API_KEY)
            index = pc.Index(config.PINECONE_INDEX_NAME)
            query_vec = Settings.embed_model.get_query_embedding(query)
            res = index.query(
                vector=query_vec,
                top_k=max(k * 20, 50),
                include_metadata=True,
                namespace=str(video_id),
            )
            candidates = []
            for m in res.get("matches", []) or []:
                md = m.get("metadata", {}) or {}
                text = md.get("text", "")
                ts = float(md.get("t_start", 0.0))
                url_val = f"https://www.youtube.com/watch?v={raw_id}&t={int(round(ts))}s" if raw_id else None
                candidates.append({
                    "title": md.get("title"),
                    "t_start": ts,
                    "t_end": float(md.get("t_end", 0.0)),
                    "snippet": (text or "")[:280],
                    "score": float(m.get("score", 0.0)),
                    "url": url_val,
                    "_full_text": text or "",
                })
        except Exception as e:
            raise ConnectionError(f"Pinecone query failed: {e}")

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
            model = os.getenv("OPENROUTER_MODEL", config.LLM_MODEL_NAME or "openrouter/google/gemini-2.0-flash-exp")
            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
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
                "timeout": config.LLM_TIMEOUT_SECONDS,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            return (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception as e:
            logging.error(f"OpenRouter call failed: {e}")

    # 2) Google Gemini via google-generativeai
    if config.GOOGLE_API_KEY:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=config.GOOGLE_API_KEY)
            model_name = (config.LLM_MODEL_NAME or "gemini-1.5-flash").replace("models/", "")
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt, request_options={"timeout": config.LLM_TIMEOUT_SECONDS})
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


def synthesize_answer(query: str, results: List[TimestampSearchResult]) -> str:
    """Use OpenRouter or Gemini API directly to synthesize an answer from retrieved sources."""
    if config.FAST_MODE:
        # Very quick extractive answer: join top snippets, avoid external LLM
        joined = " ".join([r.snippet for r in results])
        return joined[:800]
    context = "\n---\n".join(
        [f"Timestamp: {r.t_start:.0f}s\nContent: {r.snippet}" for r in results]
    )

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
    raise ValueError("Invalid YouTube URL format.")


def _timestamp_url(raw_id: str, seconds: float) -> str:
    # Use watch?v format to ensure &t= works across clients
    return f"https://www.youtube.com/watch?v={raw_id}&t={int(round(seconds))}s"


def qa_from_url(youtube_url: str, query: str, k: int) -> tuple[str, List[QASnippet]]:
    """Ingest the URL (idempotent enough for MVP), run hybrid search, build timestamped links, and summarize."""
    raw_id = _extract_raw_youtube_id(youtube_url)

    # Ingest (for MVP we always ingest; consider caching in future)
    video_id = process_and_store_video(youtube_url)

    # Search
    results = perform_hybrid_search(video_id, query, k)

    # Build links
    enriched: List[QASnippet] = []
    for r in results:
        enriched.append(QASnippet(
            t_start=r.t_start,
            t_end=r.t_end,
            title=r.title,
            snippet=r.snippet,
            score=r.score,
            url=_timestamp_url(raw_id, r.t_start),
        ))

    # Summarize
    answer = synthesize_answer(query, results)
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