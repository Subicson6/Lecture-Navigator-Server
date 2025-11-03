
# app/main.py
import sys
import uuid
import logging
import time # Ensure time is imported
from contextvars import ContextVar

# ... (compatibility shim) ...
try:
    import collections  # noqa: F401
    import collections.abc as _abc  # noqa: F401
    for _name in ("Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(_abc, _name))
except Exception:
    pass

# --- MODIFICATION START (1/3): Import Response ---
from fastapi import FastAPI, HTTPException, status, Request, UploadFile, File, Response
# --- MODIFICATION END (1/3) ---
from fastapi.routing import APIRouter
from loguru import logger
from . import service, db
from .models import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    QARequest,
    QAResponse,
    QAByIdRequest,
    TranscribeRequest,
    TranscribeResponse,
    VideosResponse,
)


# ... (Loguru setup) ...
REQUEST_ID_CTX_KEY = "request_id"
REQUEST_ID_CTX_VAR: ContextVar[str | None] = ContextVar(REQUEST_ID_CTX_KEY, default=None)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.bind().opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def add_request_id_filter(record):
    record["extra"].setdefault("request_id", REQUEST_ID_CTX_VAR.get())
    return True


logging.basicConfig(handlers=[InterceptHandler()], level=0)
for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
    logging.getLogger(name).handlers = [InterceptHandler()]

logger.remove()
logger.add(sys.stdout, level="INFO", enqueue=True, backtrace=True, diagnose=False, serialize=True, filter=add_request_id_filter)


app = FastAPI(title="VidSeek API", version="0.2.0")
api = APIRouter(prefix="/api")


@app.on_event("startup")
def startup_event():
    """Initialize services on application startup."""
    db.init_db()
    service.initialize_models()


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())
    token = REQUEST_ID_CTX_VAR.set(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        REQUEST_ID_CTX_VAR.reset(token)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/transcribe_youtube", response_model=TranscribeResponse)
@api.post("/transcribe_youtube", response_model=TranscribeResponse)
def transcribe_youtube(payload: TranscribeRequest):
    """Fetch transcript via youtube-transcript.io (fallback to local lib) and return segments + full text."""
    try:
        video_id, transcript_text, segs = service.transcribe_youtube(payload.youtube_url)
        segments = [
            {
                "text": s.get("text", ""),
                "t_start": float(s.get("start", 0.0) or 0.0),
                "t_end": float(s.get("end", 0.0) or 0.0),
            }
            for s in segs
        ]
        return TranscribeResponse(video_id=video_id, transcript_text=transcript_text, segments=segments)
    except ValueError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Transcription failed (Value Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except ConnectionError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Transcription failed (Connection Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except Exception:
        logger.exception("Transcription failed (Unexpected Error)")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during transcription.")


# Video catalog for frontend main menu
@app.get("/videos", response_model=VideosResponse)
@api.get("/videos", response_model=VideosResponse)
def list_videos():
    try:
        vc = db.get_videos_collection()
        if vc is None:
            return VideosResponse(items=[])
        docs = list(vc.find({}, {"_id": 0}).sort("created_at", -1).limit(100))
        # Normalize created_at to ISO string
        for d in docs:
            ca = d.get("created_at")
            if ca and not isinstance(ca, str):
                try:
                    d["created_at"] = ca.isoformat()
                except Exception:
                    d["created_at"] = str(ca)
        return VideosResponse(items=docs)  # type: ignore[arg-type]
    except Exception:
        logger.exception("Failed to list videos")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list videos")


@app.post("/ingest_video", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
@api.post("/ingest_video", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_video(payload: IngestRequest):
    """
    Ingests a YouTube video by URL, processes its transcript, and stores it.
    """
    try:
        video_id = service.process_and_store_video(payload.youtube_url)
        return IngestResponse(video_id=video_id)
    except ValueError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Ingestion failed (Value Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except ConnectionError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Ingestion failed (Connection Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except Exception: # Catch broader exceptions last
        logger.exception("Ingestion failed (Unexpected Error)")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during ingestion.")


@app.post("/search_timestamps", response_model=SearchResponse)
@api.post("/search_timestamps", response_model=SearchResponse)
def search_in_video(payload: SearchRequest):
    """
    Searches within a previously ingested video for a query.
    """
    # --- MODIFICATION START: Add timing ---
    search_start = time.perf_counter()
    # --- MODIFICATION END ---
    try:
        results = service.perform_hybrid_search(payload.video_id, payload.query, payload.k, window_seconds=float(getattr(payload, "window_seconds", 30.0) or 30.0))
        if not results:
            search_end = time.perf_counter() # Log time even if no results
            logger.info(f"⏱️ Query found no results in {search_end - search_start:.2f}s (video_id={payload.video_id})")
            return SearchResponse(answer="Could not find any relevant information in the video for your query.",
                                  results=[])

        # simple safety guard before LLM
        if service.is_prompt_injection(payload.query):
            search_end = time.perf_counter() # Log time for prompt injection block
            logger.warning(f"⏱️ Query blocked (prompt injection) in {search_end - search_start:.2f}s (video_id={payload.video_id})")
            return SearchResponse(
                answer="Your query appears to contain prompt-injection content. Please rephrase and try again.",
                results=results,
            )

        # Enrich answer with configurable context windows using video_id for better LLM synthesis
        answer = service.synthesize_answer(payload.query, results, video_id=payload.video_id, window_seconds=float(getattr(payload, "window_seconds", 30.0) or 30.0))
        # If any result lacks a URL (non-YouTube source), best-effort enrich via videos mapping
        try:
            if any(getattr(r, "url", None) is None for r in results):
                vc = db.get_videos_collection()
                if vc is not None:
                    vdoc = vc.find_one({"video_id": payload.video_id}, {"raw_id": 1, "source": 1})
                    rid = (vdoc or {}).get("raw_id")
                    source = (vdoc or {}).get("source")
                    if rid and (source == "youtube"):
                        for r in results:
                            if not getattr(r, "url", None):
                                ts = int(round(r.t_start))
                                r.url = f"https://www.youtube.com/watch?v={rid}&t={ts}s"
        except Exception:
            logger.warning("Failed to enrich result URLs", exc_info=True) # Log enrichment errors
            pass
        # --- MODIFICATION START: Add timing log for success ---
        search_end = time.perf_counter()
        logger.info(f"⏱️ Query answered successfully in {search_end - search_start:.2f}s (video_id={payload.video_id})")
        # --- MODIFICATION END ---

        return SearchResponse(answer=answer, results=results)
    # --- MODIFICATION START: Add timing logs for errors ---
    except ConnectionError as e:
        search_end = time.perf_counter()
        error_message = str(e) # Ensure simple string
        logger.error(f"⏱️ Search FAILED (Connection Error) in {search_end - search_start:.2f}s: {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except ValueError as e: # Add specific handling for ValueError if needed
        search_end = time.perf_counter()
        error_message = str(e) # Ensure simple string
        logger.error(f"⏱️ Search FAILED (Value Error) in {search_end - search_start:.2f}s: {error_message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except Exception: # Catch broader exceptions last
        search_end = time.perf_counter()
        logger.exception(f"⏱️ Search FAILED (Unexpected Error) in {search_end - search_start:.2f}s")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during search.")
    # --- MODIFICATION END ---


@app.post("/ingest_file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
@api.post("/ingest_file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest an uploaded .srt file using the same pipeline as YouTube ingestion."""
    try:
        video_id = await service.process_and_store_srt(file)
        return IngestResponse(video_id=video_id)
    except ValueError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Ingestion (file) failed (Value Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except ConnectionError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Ingestion (file) failed (Connection Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except Exception:
        logger.exception("Ingestion (file) failed (Unexpected Error)")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during ingestion.")


@app.post("/qa_youtube", response_model=QAResponse)
@api.post("/qa_youtube", response_model=QAResponse)
def qa_youtube(payload: QARequest):
    """MVP: Provide answer and timestamped YouTube links directly from a URL + question."""
    # --- MODIFICATION START: Add timing ---
    qa_start = time.perf_counter()
    # --- MODIFICATION END ---
    try:
        # cap k aggressively in the handler too for latency safety
        k = max(1, min(int(payload.k or 3), int(getattr(service.config, 'MAX_K', 5)))) # Use getattr for config
        answer, results = service.qa_from_url(payload.youtube_url, payload.query, k, window_seconds=30.0)
        # --- MODIFICATION START: Add timing log for success ---
        qa_end = time.perf_counter()
        logger.info(f"⏱️ QA from URL successful in {qa_end - qa_start:.2f}s")
        # --- MODIFICATION END ---
        return QAResponse(answer=answer, results=results)
    # --- MODIFICATION START: Explicitly convert error to string and add timing ---
    except ValueError as e:
        qa_end = time.perf_counter()
        error_message = str(e) # Convert error to string BEFORE logging/raising
        logger.error(f"⏱️ QA from URL FAILED (Value Error) in {qa_end - qa_start:.2f}s: {error_message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except ConnectionError as e:
        qa_end = time.perf_counter()
        error_message = str(e) # Convert error to string
        logger.error(f"⏱️ QA from URL FAILED (Connection Error) in {qa_end - qa_start:.2f}s: {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except Exception: # Catch broader exceptions last
        qa_end = time.perf_counter()
        logger.exception(f"⏱️ QA from URL FAILED (Unexpected Error) in {qa_end - qa_start:.2f}s")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during QA.")
    # --- MODIFICATION END ---


@app.post("/qa_video", response_model=QAResponse)
@api.post("/qa_video", response_model=QAResponse)
def qa_video(payload: QAByIdRequest):
    """Chat over an already-ingested video by video_id (no re-ingest)."""
    # --- MODIFICATION START: Add timing ---
    qa_start = time.perf_counter()
    # --- MODIFICATION END ---
    try:
        k = max(1, min(int(payload.k or 3), int(getattr(service.config, 'MAX_K', 5)))) # Use getattr
        answer, results = service.qa_from_video_id(payload.video_id, payload.query, k, window_seconds=float(getattr(payload, "window_seconds", 30.0) or 30.0))
        # --- MODIFICATION START: Add timing log for success ---
        qa_end = time.perf_counter()
        logger.info(f"⏱️ QA by Video ID successful in {qa_end - qa_start:.2f}s (video_id={payload.video_id})")
        # --- MODIFICATION END ---
        return QAResponse(answer=answer, results=results)
    # --- MODIFICATION START: Explicitly convert error to string and add timing ---
    except ValueError as e: # Handle ValueError here too
        qa_end = time.perf_counter()
        error_message = str(e) # Convert error to string
        logger.error(f"⏱️ QA by Video ID FAILED (Value Error) in {qa_end - qa_start:.2f}s: {error_message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except ConnectionError as e:
        qa_end = time.perf_counter()
        error_message = str(e) # Convert error to string
        logger.error(f"⏱️ QA by Video ID FAILED (Connection Error) in {qa_end - qa_start:.2f}s: {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except Exception: # Catch broader exceptions last
        qa_end = time.perf_counter()
        logger.exception(f"⏱️ QA by Video ID FAILED (Unexpected Error) in {qa_end - qa_start:.2f}s")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during QA.")
    # --- MODIFICATION END ---


# --- MODIFICATION START (2/3): Add DELETE endpoint ---
@app.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
@api.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: str):
    """Deletes all data associated with a video_id (embeddings and metadata)."""
    try:
        # 1. Delete embeddings
        deleted_embeddings = service.purge_video_embeddings(video_id)
        logger.info(f"🗑️ Deleted {deleted_embeddings} embedding chunks for video_id={video_id}")

        # 2. Delete metadata from the 'videos' collection
        vc = db.get_videos_collection()
        deleted_metadata = 0
        if vc is not None:
            res = vc.delete_one({"video_id": video_id})
            deleted_metadata = res.deleted_count
            if deleted_metadata > 0:
                 logger.info(f"🗑️ Deleted metadata entry for video_id={video_id}")
            else:
                 logger.warning(f"⚠️ No metadata entry found to delete for video_id={video_id}")
        else:
            logger.error("❌ Could not get videos collection to delete metadata.")
            # Decide if you want to raise an error here or just log it
            # If embeddings were deleted, it might still be considered a partial success

        # Return 204 No Content on successful deletion
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except ConnectionError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"❌ Failed to delete video {video_id} due to connection error: {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except Exception: # Catch broader exceptions last
        logger.exception(f"❌ Unexpected error while deleting video {video_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to delete video data.")
# --- MODIFICATION END (2/3) ---


# --- Admin Endpoints (Optional: Keep or adjust security) ---
@app.post("/admin/purge_video/{video_id}")
@api.post("/admin/purge_video/{video_id}")
def purge_video(video_id: str):
    """Delete all embeddings/documents for a video_id to prepare for re-ingestion with new dims."""
    try:
        deleted = service.purge_video_embeddings(video_id)
        return {"video_id": video_id, "deleted": deleted}
    except ConnectionError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Purge failed (Connection Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except Exception:
        logger.exception("Purge failed (Unexpected Error)")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to purge video embeddings.")


@app.post("/admin/reembed_video/{video_id}", response_model=IngestResponse)
@api.post("/admin/reembed_video/{video_id}", response_model=IngestResponse)
def reembed_video(video_id: str):
    """Re-embed an existing video using current embedding model/dims; keeps the same video_id."""
    try:
        vid = service.reembed_video_by_id(video_id)
        return IngestResponse(video_id=vid)
    except ValueError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Re-embed failed (Value Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except ConnectionError as e:
        error_message = str(e) # Ensure simple string
        logger.error(f"Re-embed failed (Connection Error): {error_message}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_message)
    except Exception:
        logger.exception("Re-embed failed (Unexpected Error)")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to re-embed video.")

# Mount /api router for frontend proxy compatibility
# --- MODIFICATION START (3/3): Ensure api router includes the new endpoint ---
app.include_router(api)
# --- MODIFICATION END (3/3) ---

