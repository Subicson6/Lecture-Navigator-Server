
# Lecture Navigator Server (VidSeek API)

Lecture Navigator Server (VidSeek API)


FastAPI service for ingesting YouTube video transcripts and enabling semantic search with a Retrieval-Augmented Generation (RAG) pipeline. The app uses LlamaIndex for ingestion (chunking + embedding), MongoDB Atlas Vector Search for storage/retrieval, and Gemini as the LLM for answer synthesis. Structured JSON logs are emitted via Loguru with a per-request request_id.

## Features
- Ingest YouTube transcripts via LlamaIndex YoutubeTranscriptReader
- Smart chunking with SentenceSplitter
- Embeddings via HuggingFace (default: all-MiniLM-L6-v2)
- Storage and retrieval via MongoDB Atlas Vector Search
- Optional answer synthesis using Gemini (via LlamaIndex llm)
- Structured JSON logging with request_id (Loguru + middleware)

## Project Structure

```
app/
  config.py        # Environment/config values
  db.py            # MongoDB connection and optional index creation
  main.py          # FastAPI app, routes, Loguru setup
  models.py        # Pydantic request/response models
  service.py       # LlamaIndex pipeline (ingest/search/synthesize)
  __init__.py
pyproject.toml     # Poetry dependencies
README.md
```

## Requirements
- Python 3.10–3.12
- MongoDB Atlas (or MongoDB 7.0+ with vector search) and connection string
- Optional: Google API key for Gemini

## Setup (Poetry)

1) Install Poetry (if not installed)
   https://python-poetry.org/docs/#installation

2) From the project root:
```
poetry install
```

3) Create a .env file (or set env vars in PyCharm Run/Debug Config):

```
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>/?retryWrites=true&w=majority
MONGODB_DB=vidseek_db
MONGODB_COLLECTION=transcripts

# Optional (for LLM synthesis)
GOOGLE_API_KEY=your_google_api_key
LLM_MODEL_NAME=gemini-1.5-flash-latest

# Embeddings
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
FORCE_LOCAL_EMBEDDINGS=false

# Optional (not required by default)
OPENROUTER_API_KEY=
```

## Run the server

From the project root:
```
poetry run uvicorn app.main:app --reload
```

By default the API will be available at:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

Each HTTP request is assigned a request_id and emitted in JSON logs (stdout). The same request_id is returned in the `X-Request-ID` header.

## How the pipeline works

1) Ingestion (POST /ingest_video)
   - Extract YouTube video id from URL
   - YoutubeTranscriptReader loads transcript as Documents
   - VectorStoreIndex.from_documents with StorageContext(vector store = MongoDB Atlas Vector Search)
     - SentenceSplitter chunks text
     - HuggingFaceEmbedding creates embeddings
     - Documents are stored with embeddings in MongoDB

2) Retrieval (POST /search)
   - Encode user query with the same embedding model
   - Use $vectorSearch on MongoDB Atlas to retrieve top-k chunks for a given video_id

3) Synthesis (optional)
   - If GOOGLE_API_KEY is set, LlamaIndex Gemini llm synthesizes a concise answer from retrieved chunks

## API

### POST /ingest_video
Request:
```
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEOID"
}
```
Response:
```
{
  "video_id": "<uuid>"
}
```

### POST /search
Request:
```
{
  "video_id": "<uuid>",
  "query": "What is vector search?",
  "k": 5
}
```
Response (example):
```
{
  "answer": "Vector search finds semantically similar chunks ...",
  "sources": [
    {
      "video_id": "<uuid>",
      "text": "...",
      "t_start": 30.0,
      "t_end": 60.0,
      "score": 0.89
    }
  ]
}
```

## Logging
- Loguru is configured to emit JSON to stdout with `request_id` bound for each request
- Uvicorn and FastAPI logs are intercepted and routed through Loguru

## MongoDB Atlas indices
- app/db.py includes a helper to create Atlas Search indices (vector and keyword). This is called on startup. If running against non-Atlas or without permissions, index creation may warn and continue.

## Troubleshooting
- 401 or 403 from Gemini: verify GOOGLE_API_KEY and project permissions
- Vector search errors: ensure cluster supports Atlas Search; confirm index name `vector_index`
- Empty results: verify ingestion completed and `video_id` matches the ingested response
- Missing logs: ensure stdout is visible in your run configuration

## Notes
- The default embedding model is small and CPU-friendly. You can switch to another HuggingFace model by setting EMBEDDING_MODEL_NAME.
- OPENROUTER_API_KEY is present for future routing but not required in this build.

<<<<<<< HEAD
=======
| Method | Endpoint                            | Description                                                                    |
| :----- | :---------------------------------- | :----------------------------------------------------------------------------- |
| `POST` | `/api/ingest_video`                 | Ingests a YouTube video from a URL, processes it, and stores the embeddings.     |
| `POST` | `/api/ingest_file`                  | Ingests a local `.srt` transcript file.                                        |
| `POST` | `/api/search_timestamps`            | Performs a hybrid search for a query within a previously ingested video.         |
| `POST` | `/api/qa_youtube`                   | A one-shot endpoint to get a Q\&A response directly from a YouTube URL.         |
| `POST` | `/api/qa_video`                     | Asks a question about an already-ingested video using its unique `video_id`.       |
| `GET`  | `/api/videos`                       | Lists the metadata of up to 100 recently ingested videos.                        |
| `POST` | `/api/transcribe_youtube`           | Fetches and returns the transcript for a YouTube video without indexing it.      |
| `POST` | `/api/admin/purge_video/{video_id}` | **Admin**: Deletes all data associated with a specific `video_id`.             |
| `POST` | `/api/admin/reembed_video/{video_id}` | **Admin**: Re-generates embeddings for a video using current model settings. |
| `GET`  | `/health`                           | A simple health check endpoint.                                                |

