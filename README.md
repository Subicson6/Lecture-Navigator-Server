You got it\! I've updated the heading and incorporated a flowchart to visually represent the core data ingestion and querying process.

Here's the revised `README.md` content:

-----

# Lecture Navigator Server (VidSeek API)

[](https://www.python.org/)
[](https://fastapi.tiangolo.com/)
[](https://www.mongodb.com/atlas)
[](https://www.google.com/search?q=%5Bhttps://www.pinecone.io/%5D\(https://www.pinecone.io/\))
[](https://www.google.com/search?q=%5Bhttps://huggingface.co/%5D\(https://huggingface.co/\))

The VidSeek API is a powerful backend service designed to index and search through video content, primarily from YouTube. It automatically fetches video transcripts, processes them into vector embeddings, and allows for intelligent, semantic question-answering. Users can ask questions about a video and receive concise answers synthesized by a Large Language Model (LLM), complete with direct, timestamped links to the relevant moments in the video.

## ✨ Key Features

  * **YouTube Video Ingestion**: Process and index any YouTube video via its URL.
  * **SRT File Uploads**: Ingest local video transcripts by uploading `.srt` files.
  * **Hybrid Search**: Combines traditional text search with modern vector search for highly relevant results within video transcripts.
  * **LLM-Powered Q\&A**: Utilizes powerful LLMs like Google's Gemini or models from OpenRouter to provide direct, synthesized answers to user queries.
  * **Timestamped Results**: Every answer is backed by source snippets that link directly to the specific timestamp in the YouTube video.
  * **Dual Vector Database Support**: Flexible backend configuration allows you to choose between **MongoDB Atlas Vector Search** and **Pinecone** for storing and querying embeddings.
  * **High-Quality Embeddings**: Leverages state-of-the-art sentence-transformer models from Hugging Face (`BAAI/bge-small-en-v1.5`, etc.) for generating embeddings.
  * **Fast and Modern API**: Built with FastAPI for a high-performance, asynchronous, and well-documented API.

## 🌊 System Flowchart

The following diagram illustrates the primary data ingestion and query process within the Lecture Navigator Server.

```mermaid
graph TD
    A[Start] --> B(User Provides Video URL or SRT)
    B --> C{Ingest Video API Call}

    C --> D{Fetch Transcript}
    D -- YouTube URL --> E[Use youtube-transcript API]
    D -- SRT File --> F[Process Uploaded SRT]

    E --> G[Clean & Chunk Transcript]
    F --> G

    G --> H[Generate Embeddings]
    H --> I{Store in Vector Database}
    I -- Configured as --> J[MongoDB Atlas Vector Search]
    I -- Configured as --> K[Pinecone]

    J --> L(Ingestion Complete)
    K --> L

    L --> M{User Submits Query}
    M --> N[Generate Query Embedding]
    N --> O{Search Vector Database}
    O --> P[Retrieve Relevant Chunks]
    P --> Q[Re-rank Chunks (Cross-encoder)]
    Q --> R{Send Chunks & Query to LLM}
    R -- e.g., Gemini, OpenRouter --> S[LLM Generates Answer]
    S --> T[Format Answer with Timestamps]
    T --> U(Return Answer to User)
```

> **Explanation:**
>
> 1.  **Ingestion:** A video URL or SRT file is provided. The system fetches/processes the transcript, chunks it into smaller, manageable pieces, and generates vector embeddings for each chunk. These embeddings are then stored in the configured vector database (MongoDB Atlas or Pinecone).
> 2.  **Querying:** When a user asks a question, the query is also converted into an embedding. This embedding is used to search the vector database for the most semantically similar text chunks from the ingested videos. These chunks are then re-ranked for optimal relevance and sent to a Large Language Model (LLM) along with the original question.
> 3.  **Answering:** The LLM synthesizes an answer based on the provided relevant chunks. The API then formats this answer, including direct timestamps to the source material within the video, and returns it to the user.

## 🛠️ Tech Stack

  * **Backend**: Python, FastAPI
  * **Databases**: MongoDB, Pinecone
  * **AI / ML**:
      * LlamaIndex
      * Sentence-Transformers (Hugging Face)
      * Google Generative AI (Gemini)
      * OpenRouter
  * **Logging**: Loguru

## 🚀 Getting Started

### Prerequisites

  * Python 3.9+
  * A MongoDB Atlas account with a cluster URL or a Pinecone account.
  * API keys for Google AI (Gemini) and/or OpenRouter.
  * An API key for the `youtube-transcript.io` service (optional but recommended).

### 1\. Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/Subicson6/Lecture-Navigator-Server.git
cd Lecture-Navigator-Server
```

Next, it is highly recommended to create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required Python packages. Create a `requirements.txt` file with the following content and then run `pip install -r requirements.txt`.

```txt
# requirements.txt
fastapi
uvicorn[standard]
python-dotenv
pymongo
loguru
llama-index
sentence-transformers
torch
google-generativeai
requests
urllib3
```

### 2\. Configuration

The server is configured using environment variables. Create a file named `.env` in the root of the project by copying the example below.

```env
# .env

# --- Database ---
# Choose your vector DB: "mongodb" or "pinecone"
VECTOR_DB=pinecone

# MongoDB Settings (if VECTOR_DB=mongodb)
MONGODB_URI="mongodb+srv://<user>:<password>@<your-cluster-url>/"
MONGODB_DB=vidseek_db
MONGODB_COLLECTION=transcripts

# Pinecone Settings (if VECTOR_DB=pinecone)
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_INDEX_NAME=lecturenav-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1


# --- AI Models ---
# Google Generative AI / Gemini API key
GOOGLE_API_KEY="your-google-ai-api-key"

# OpenRouter API key (alternative to Google)
OPENROUTER_API_KEY="your-openrouter-api-key"
OPENROUTER_MODEL=google/gemini-1.5-flash

# Embedding model from Hugging Face
EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
# Cross-encoder for re-ranking search results
CROSS_ENCODER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2


# --- External transcript service (youtube-transcript.io) ---
YOUTUBE_TRANSCRIPT_API_KEY="your-youtube-transcript-api-key"
```

### 3\. Running the Server

Once the dependencies are installed and your `.env` file is configured, run the application using Uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

## 📖 API Documentation

The API provides several endpoints for ingesting and querying video content.

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