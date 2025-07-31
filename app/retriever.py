import os
import pinecone
from dotenv import load_dotenv

load_dotenv()

_embedder = None
_index = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small model to fit in 512MB RAM
            _embedder = SentenceTransformer("all-MiniLM-L4-v2")  # ~60MB
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load embedder: {e}")
    return _embedder

def get_index():
    global _index
    if _index is None:
        try:
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            index_name = os.getenv("PINECONE_INDEX_NAME")
            if not index_name:
                raise ValueError("❌ Pinecone index name not set in environment variables.")
            _index = pinecone.Index(index_name)
            _index.describe_index_stats()
        except Exception as e:
            raise RuntimeError(f"❌ Pinecone index not ready or does not exist: {e}")
    return _index

def store_chunks_in_pinecone(chunks, file_id):
    try:
        embedder = get_embedder()
        index = get_index()
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return

    vectors = []
    for i, chunk in enumerate(chunks):
        try:
            vec = embedder.encode(chunk).tolist()
            vectors.append({
                "id": f"{file_id}-{i}",
                "values": vec,
                "metadata": {"text": chunk}
            })
        except Exception as e:
            print(f"⚠️ Skipping chunk {i} due to encoding error: {e}")

    if vectors:
        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            print(f"❌ Upsert failed: {e}")

def query_chunks_from_pinecone(query, top_k=3):
    try:
        embedder = get_embedder()
        index = get_index()
        query_vec = embedder.encode(query).tolist()
        results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results.get("matches", [])]
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []
