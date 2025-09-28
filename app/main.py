# app/main.py
import sys
import uuid
import logging
from contextvars import ContextVar

# Temporary Python 3.12 compatibility shim for packages that still import
# MutableMapping/Mapping/Sequence from 'collections' instead of 'collections.abc'.
# This prevents ImportError during startup if an older 'bson' package is
# inadvertently installed. Proper fix: uninstall the standalone 'bson' package
# and rely on pymongo's bundled bson (see notes in README or issue tracker).
try:
    import collections  # noqa: F401
    import collections.abc as _abc  # noqa: F401
    for _name in ("Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(_abc, _name))
except Exception:
    # Best-effort shim; do not fail app startup because of this block
    pass

from fastapi import FastAPI, HTTPException, status, Request, UploadFile, File
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


# ---- Loguru setup (JSON logs with request_id) ----
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


# Configure root/uvicorn loggers to go through Loguru
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception:
        logger.exception("Transcription failed")
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during ingestion.")


@app.post("/search_timestamps", response_model=SearchResponse)
@api.post("/search_timestamps", response_model=SearchResponse)
def search_in_video(payload: SearchRequest):
    """
    Searches within a previously ingested video for a query.
    """
    try:
        results = service.perform_hybrid_search(payload.video_id, payload.query, payload.k, window_seconds=float(getattr(payload, "window_seconds", 30.0) or 30.0))
        if not results:
            return SearchResponse(answer="Could not find any relevant information in the video for your query.",
                                  results=[])

        # simple safety guard before LLM
        if service.is_prompt_injection(payload.query):
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
            pass
        return SearchResponse(answer=answer, results=results)
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during search.")


@app.post("/ingest_file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
@api.post("/ingest_file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest an uploaded .srt file using the same pipeline as YouTube ingestion."""
    try:
        video_id = await service.process_and_store_srt(file)
        return IngestResponse(video_id=video_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception:
        logger.exception("Ingestion (file) failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during ingestion.")


@app.post("/qa_youtube", response_model=QAResponse)
@api.post("/qa_youtube", response_model=QAResponse)
def qa_youtube(payload: QARequest):
    """MVP: Provide answer and timestamped YouTube links directly from a URL + question."""
    try:
        # cap k aggressively in the handler too for latency safety
        k = max(1, min(int(payload.k or 3), int(service.config.MAX_K)))
        answer, results = service.qa_from_url(payload.youtube_url, payload.query, k, window_seconds=30.0)
        return QAResponse(answer=answer, results=results)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception:
        logger.exception("QA from URL failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during QA.")


@app.post("/qa_video", response_model=QAResponse)
@api.post("/qa_video", response_model=QAResponse)
def qa_video(payload: QAByIdRequest):
    """Chat over an already-ingested video by video_id (no re-ingest)."""
    try:
        k = max(1, min(int(payload.k or 3), int(service.config.MAX_K)))
        answer, results = service.qa_from_video_id(payload.video_id, payload.query, k, window_seconds=float(getattr(payload, "window_seconds", 30.0) or 30.0))
        return QAResponse(answer=answer, results=results)
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception:
        logger.exception("QA by video_id failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during QA.")


@app.post("/admin/purge_video/{video_id}")
@api.post("/admin/purge_video/{video_id}")
def purge_video(video_id: str):
    """Delete all embeddings/documents for a video_id to prepare for re-ingestion with new dims."""
    try:
        deleted = service.purge_video_embeddings(video_id)
        return {"video_id": video_id, "deleted": deleted}
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception:
        logger.exception("Purge failed")
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception:
        logger.exception("Re-embed failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to re-embed video.")

# Mount /api router for frontend proxy compatibility
app.include_router(api)