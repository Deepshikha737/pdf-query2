import os
import pinecone
from dotenv import load_dotenv

load_dotenv()

_embedder = None
_index = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # ~80MB
  # or a smaller model if needed
    return _embedder

def get_index():
    global _index
    if _index is None:
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("❌ Pinecone index name not set in environment variables.")

        try:
            _index = pinecone.Index(index_name)
            _index.describe_index_stats()
        except Exception as e:
            raise RuntimeError(f"❌ Pinecone index not ready or does not exist: {e}")
    return _index

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
