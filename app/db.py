# app/db.py
import logging
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from . import config

mongo_client = None
db = None
collection = None
videos_collection = None

def init_db():
    """Initializes the MongoDB client and collection."""
    global mongo_client, db, collection, videos_collection
    if config.MONGODB_URI:
        try:
            mongo_client = MongoClient(config.MONGODB_URI)
            db = mongo_client[config.MONGODB_DB]
            collection = db[config.MONGODB_COLLECTION]
            videos_collection = db["videos"]
            logging.info("MongoDB connection successful.")
            ensure_search_indexes()
            ensure_videos_indexes()
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            mongo_client = db = collection = videos_collection = None
    else:
        logging.warning("MONGODB_URI not set. Database functionality will be disabled.")

def ensure_search_indexes():
    """Creates Atlas Search indexes if they don't exist."""
    # Avoid truthiness checks on PyMongo objects (they don't implement bool())
    if db is None or collection is None:
        return
    try:
        # 1) Skip if already exists
        try:
            res = db.command({
                "listSearchIndexes": config.MONGODB_COLLECTION,
                "name": "hybrid_index",
            })
            first_batch = (res or {}).get("cursor", {}).get("firstBatch", [])
            if first_batch:
                logging.info("Atlas Search index 'hybrid_index' already exists. Skipping creation.")
                return
        except Exception:
            # If listing fails (older server), continue and attempt create
            pass

        # 2) Create index with recommended mapping
        # NOTE: dims should match your embedding model. BGE-small = 384; adjust if needed.
        from .config import EMBEDDING_DIM
        vector_dims = EMBEDDING_DIM
        definition = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text": {"type": "string", "analyzer": "lucene.standard"},
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": vector_dims,
                        "similarity": "cosine",
                    },
                    "metadata": {
                        "type": "document",
                        "fields": {
                            "title": {"type": "string", "analyzer": "lucene.standard"},
                            "video_id": {"type": "string", "analyzer": "lucene.keyword"},
                        },
                    },
                },
            },
        }

        db.command({
            "createSearchIndexes": config.MONGODB_COLLECTION,
            "indexes": [
                {
                    "name": "hybrid_index",
                    "definition": definition,
                }
            ],
        })
        logging.info("Atlas hybrid Search index created.")
    except OperationFailure as e:
        # Handle already-exists or non-Atlas gracefully
        msg = str(e)
        if "IndexAlreadyExists" in msg or e.code == 68:
            logging.info("Atlas Search index 'hybrid_index' already exists. Skipping creation.")
        else:
            logging.warning(f"Search index creation skipped or failed (possibly non-Atlas or permissions): {e}")
    except Exception as e:
        logging.warning(f"Search index creation skipped or failed (possibly non-Atlas or permissions): {e}")

def get_collection():
    return collection


def ensure_videos_indexes():
    """Ensure caches/videos collection has required indexes."""
    if db is None:
        return
    try:
        vc = db["videos"]
        # unique mapping from raw_id to video_id
        vc.create_index("raw_id", unique=True)
        # quick lookup by video_id (used to build deep links without re-embedding)
        vc.create_index("video_id")
        vc.create_index("created_at")
    except Exception as e:
        logging.warning(f"Failed creating videos indexes: {e}")


def get_videos_collection() -> MongoClient | None:  # type: ignore[override]
    return videos_collection