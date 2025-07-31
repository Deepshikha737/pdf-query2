# main.py

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from app.parser import extract_text_from_pdf
from app.chunker import chunk_text
from app.retriever import store_chunks_in_pinecone, query_chunks_from_pinecone
from app.groq_llm import query_groq_llm

import uuid
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.post("/run")
async def run_query(file: UploadFile = File(...), question: str = Form(...)):
    try:
        print("📥 Step 1: Reading uploaded file")
        file_bytes = await file.read()

        print("📄 Step 2: Extracting text from PDF")
        raw_text = extract_text_from_pdf(file_bytes)
        print(f"✅ Extracted {len(raw_text)} characters of text")

        print("🧩 Step 3: Chunking text")
        chunks = chunk_text(raw_text)
        print(f"✅ Chunked into {len(chunks)} pieces")

        print("📦 Step 4: Storing chunks in Pinecone")
        file_id = str(uuid.uuid4())
        store_chunks_in_pinecone(chunks, file_id)
        print(f"✅ Stored chunks with file_id: {file_id}")

        print("🔍 Step 5: Querying Pinecone")
        top_chunks = query_chunks_from_pinecone(question)
        print(f"✅ Retrieved {len(top_chunks)} top chunks")

        print("🤖 Step 6: Querying Groq LLM")
        answer = query_groq_llm(" ".join(top_chunks), question)
        print("✅ Received answer from Groq")

        return {
            "question": question,
            "context_used": top_chunks,
            "answer": answer
        }

    except Exception as e:
        print(f"🔥 Error in /run: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "LLM PDF QA API is running. Go to /docs for Swagger UI."}
