import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

_embedder = None
_index = None
_pc_client = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            # Use small model to fit in 512MB RAM
            _embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # 384-dim
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load embedder: {e}")
    return _embedder

def get_index():
    global _index, _pc_client
    if _index is None:
        try:
            index_name = os.getenv("PINECONE_INDEX_NAME")
            if not index_name:
                raise ValueError("❌ Pinecone index name not set in environment variables.")
            
            # Initialize Pinecone client
            _pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

            if index_name not in _pc_client.list_indexes().names():
                _pc_client.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",  # or "gcp" if using GCP
                        region=os.getenv("PINECONE_REGION", "us-west-2")
                    )
                )

            _index = _pc_client.Index(index_name)
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
