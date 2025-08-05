import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

def get_embedder():
    try:
        return SentenceTransformer("paraphrase-MiniLM-L3-v2",device="cpu")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load embedder: {e}")

def get_index():
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("❌ Pinecone index name not set.")
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_REGION", "us-west-2")
                )
            )
        return pc.Index(index_name)
    except Exception as e:
        raise RuntimeError(f"❌ Pinecone index error: {e}")

def store_chunks_in_pinecone(chunks, file_id):
    try:
        embedder = get_embedder()
        index = get_index()

        for i, chunk in enumerate(chunks):
            vec = embedder.encode(chunk, convert_to_numpy=True).tolist()
            index.upsert(vectors=[{
                "id": f"{file_id}-{i}",
                "values": vec,
                "metadata": {"text": chunk}
            }])
    except Exception as e:
        print(f"❌ Store error: {e}")

def query_chunks_from_pinecone(query, top_k=3):
    try:
        embedder = get_embedder()
        index = get_index()
        query_vec = embedder.encode(query, convert_to_numpy=True).tolist()
        results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results.get("matches", [])]
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []
