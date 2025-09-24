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