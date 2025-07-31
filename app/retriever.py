import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def get_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        raise ValueError("❌ Pinecone API key or Index name not set in environment variables.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Optional: ensure index is ready
    try:
        index.describe_index_stats()
    except Exception as e:
        raise RuntimeError(f"❌ Pinecone index not ready or does not exist: {e}")

    return index

def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def store_chunks_in_pinecone(chunks, file_id):
    embedder = get_embedder()
    index = get_index()
    vectors = [
        {
            "id": f"{file_id}-{i}",
            "values": embedder.encode(chunk).tolist(),
            "metadata": {"text": chunk}
        }
        for i, chunk in enumerate(chunks)
    ]
    index.upsert(vectors=vectors)

def query_chunks_from_pinecone(query, top_k=5):
    embedder = get_embedder()
    index = get_index()
    query_vec = embedder.encode(query).tolist()
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]
