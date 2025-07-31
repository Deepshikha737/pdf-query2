import os
import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialize SentenceTransformer once
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone once
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

def get_index():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("❌ Pinecone index name not set in environment variables.")

    try:
        index = pinecone.Index(index_name)
        index.describe_index_stats()
        return index
    except Exception as e:
        raise RuntimeError(f"❌ Pinecone index not ready or does not exist: {e}")

def get_embedder():
    return _embedder

def store_chunks_in_pinecone(chunks, file_id):
    embedder = get_embedder()
    index = get_index()

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
        index.upsert(vectors=vectors)

def query_chunks_from_pinecone(query, top_k=3):
    embedder = get_embedder()
    index = get_index()

    try:
        query_vec = embedder.encode(query).tolist()
        results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results.get("matches", [])]
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []
