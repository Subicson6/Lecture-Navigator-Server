# app/main.py
import sys
import uuid
import logging
from contextvars import ContextVar
from fastapi import FastAPI, HTTPException, status, Request, UploadFile, File
from loguru import logger
from . import service, db
from .models import IngestRequest, IngestResponse, SearchRequest, SearchResponse


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


@app.post("/ingest_video", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
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
def search_in_video(payload: SearchRequest):
    """
    Searches within a previously ingested video for a query.
    """
    try:
        results = service.perform_hybrid_search(payload.video_id, payload.query, payload.k)
        if not results:
            return SearchResponse(answer="Could not find any relevant information in the video for your query.",
                                  results=[])

        # simple safety guard before LLM
        if service.is_prompt_injection(payload.query):
            return SearchResponse(
                answer="Your query appears to contain prompt-injection content. Please rephrase and try again.",
                results=results,
            )

        answer = service.synthesize_answer_with_gemini(payload.query, results)
        return SearchResponse(answer=answer, results=results)
    except ConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during search.")


@app.post("/ingest_file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
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