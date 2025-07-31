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
        file_bytes = await file.read()
        raw_text = extract_text_from_pdf(file_bytes)

        chunks = chunk_text(raw_text)
        file_id = str(uuid.uuid4())

        store_chunks_in_pinecone(chunks, file_id)
        top_chunks = query_chunks_from_pinecone(question)
        answer = query_groq_llm(" ".join(top_chunks), question)

        return {
            "question": question,
            "context_used": top_chunks,
            "answer": answer
        }

    except Exception as e:
        # Log error to console
        print(f"🔥 Error in /run: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "LLM PDF QA API is running. Go to /docs for Swagger UI."}
