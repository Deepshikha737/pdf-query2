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
        if not raw_text.strip():
            return JSONResponse(content={"error": "No extractable text found in PDF."}, status_code=400)

        chunks = chunk_text(raw_text)
        if not chunks:
            return JSONResponse(content={"error": "Failed to generate any chunks from text."}, status_code=400)

        file_id = str(uuid.uuid4())
        store_chunks_in_pinecone(chunks, file_id)

        top_chunks = query_chunks_from_pinecone(question)
        if not top_chunks:
            return JSONResponse(content={"error": "No relevant context found."}, status_code=400)

        # Join top 2 chunks to reduce context size for LLM
        context = " ".join(top_chunks[:2])

        answer = query_groq_llm(context, question)

        return {
            "question": question,
            "context_used": top_chunks[:2],
            "answer": answer
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "LLM PDF QA API is running. Go to /docs for Swagger UI."}
