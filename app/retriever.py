import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load embedder once globally to avoid repeated heavy loads
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        raise ValueError("❌ Pinecone API key or Index name not set in environment variables.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    try:
        index.describe_index_stats()
    except Exception as e:
        raise RuntimeError(f"❌ Pinecone index not ready or does not exist: {e}")

    return index

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

def query_chunks_from_pinecone(query, top_k=3):  # Reduce top_k to save memory
    embedder = get_embedder()
    index = get_index()

    try:
        query_vec = embedder.encode(query).tolist()
        results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results.get("matches", [])]
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []
