# app/groq_llm.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def query_groq_llm(context, question):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ LLM error: GROQ_API_KEY is not set in environment variables"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",  # or "llama3-70b-8192"
        "messages": [
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"GROQ LLM error: {str(e)}"
