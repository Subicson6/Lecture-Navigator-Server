# app/db.py
import logging
from pymongo import MongoClient
from . import config

mongo_client = None
db = None
collection = None

def init_db():
    """Initializes the MongoDB client and collection."""
    global mongo_client, db, collection
    if config.MONGODB_URI:
        try:
            mongo_client = MongoClient(config.MONGODB_URI)
            db = mongo_client[config.MONGODB_DB]
            collection = db[config.MONGODB_COLLECTION]
            logging.info("MongoDB connection successful.")
            ensure_search_indexes()
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            mongo_client = db = collection = None
    else:
        logging.warning("MONGODB_URI not set. Database functionality will be disabled.")

def ensure_search_indexes():
    """Creates Atlas Search indexes if they don't exist."""
    if not db or not collection:
        return
    try:
        # NOTE: Vector dimensions are hardcoded for the fallback model.
        # A more robust solution would pass this in from the embedding provider.
        vector_dims = 384  # all-MiniLM-L6-v2 dimensions

        # Create a single hybrid index that supports both vector and keyword search
        db.command({
            "createSearchIndexes": config.MONGODB_COLLECTION,
            "indexes": [
                {
                    "name": "hybrid_index",
                    "definition": {
                        # Enable dynamic mappings so metadata fields (e.g., metadata.video_id) are searchable/filterable
                        "mappings": {"dynamic": True},
                        # Configure the vector field for KNN
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": vector_dims,
                                "similarity": "cosine",
                            }
                        ],
                    },
                }
            ],
        })
        logging.info("Atlas hybrid Search index checked/created.")
    except Exception as e:
        logging.warning(f"Search index creation skipped or failed (this is expected on non-Atlas DBs): {e}")

def get_collection():
    return collection