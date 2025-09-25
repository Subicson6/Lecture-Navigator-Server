# app/models.py
from pydantic import BaseModel
from typing import List

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

class SearchResponse(BaseModel):
    results: List[TimestampSearchResult]
    answer: str

class SearchRequest(BaseModel):
    video_id: str
    query: str
    k: int = 4


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