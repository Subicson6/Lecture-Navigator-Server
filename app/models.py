# app/models.py
from pydantic import BaseModel
from typing import List, Optional

class IngestRequest(BaseModel):
    youtube_url: str

class IngestResponse(BaseModel):
    video_id: str

class TimestampSearchResult(BaseModel):
    t_start: float
    t_end: float
    title: str | None = None
    snippet: str
    score: float
    # Deep link to the timestamp (e.g., YouTube watch URL with &t= param)
    url: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[TimestampSearchResult]
    answer: str

class SearchRequest(BaseModel):
    video_id: str
    query: str
    k: int = 3
    # Optional expansion window for LLM context per hit (seconds)
    window_seconds: float = 30.0


# --- Transcription (API-aligned) ---
class TranscriptSegment(BaseModel):
    text: str
    t_start: float
    t_end: float


class TranscribeRequest(BaseModel):
    youtube_url: str


class TranscribeResponse(BaseModel):
    video_id: str
    transcript_text: str
    segments: List[TranscriptSegment]


# --- MVP: URL-based QA with timestamped links ---
class QARequest(BaseModel):
    youtube_url: str
    query: str
    k: int = 4


class QASnippet(BaseModel):
    t_start: float
    t_end: float
    title: str | None = None
    snippet: str
    score: float
    url: str


class QAResponse(BaseModel):
    answer: str
    results: List[QASnippet]


# --- Video-id based QA ---
class QAByIdRequest(BaseModel):
    video_id: str
    query: str
    k: int = 3
    window_seconds: float = 30.0


# --- Video catalog for frontend main menu ---
class VideoMeta(BaseModel):
    video_id: str
    raw_id: str
    source: str
    url: str
    title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: Optional[str] = None  # ISO timestamp


class VideosResponse(BaseModel):
    items: List[VideoMeta]