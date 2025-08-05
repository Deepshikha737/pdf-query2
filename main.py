from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from app.parser import extract_text_from_pdf
from app.chunker import chunk_text
from app.retriever import store_chunks_in_pinecone, query_chunks_from_pinecone
from app.groq_llm import query_groq_llm  # You can swap this with openrouter_llm later

import uuid
import logging
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

MAX_PDF_WORDS = 800  # 🔹 Limit PDF size for memory savings
MAX_CONTEXT_CHUNKS = 2  # 🔹 Keep only 2 chunks in LLM context

def truncate_text(text: str, max_words=MAX_PDF_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words])

@app.post("/run")
async def run_query(file: UploadFile = File(...), question: str = Form(...)):
    try:
        logging.info("📥 Received file and question: %s", question)

        file_bytes = await file.read()
        raw_text = extract_text_from_pdf(file_bytes)
        logging.info("📝 Extracted %d characters", len(raw_text))

        if not raw_text.strip():
            return JSONResponse(content={"error": "No extractable text found in PDF."}, status_code=400)

        truncated_text = truncate_text(raw_text)
        chunks = chunk_text(truncated_text)
        logging.info("✂️ Created %d truncated chunks", len(chunks))

        if not chunks:
            return JSONResponse(content={"error": "No chunks created from the text."}, status_code=400)

        file_id = str(uuid.uuid4())
        store_chunks_in_pinecone(chunks, file_id)
        logging.info("📦 Stored %d chunks with file ID %s", len(chunks), file_id)

        top_chunks = query_chunks_from_pinecone(question)
        logging.info("🔍 Retrieved %d matching chunks", len(top_chunks))

        if not top_chunks:
            return JSONResponse(content={"error": "No relevant context found."}, status_code=400)

        context = " ".join(top_chunks[:MAX_CONTEXT_CHUNKS])
        answer = query_groq_llm(context, question)

        return {
            "question": question,
            "context_used": top_chunks[:MAX_CONTEXT_CHUNKS],
            "answer": answer
        }

    except Exception as e:
        logging.exception("❌ Error during /run endpoint:")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "✅ LLM PDF QA API is running. Visit /docs to test."}
