# app/groq_llm.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def query_groq_llm(context, question):
    prompt = f"""You are an intelligent assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",  # or "llama3-70b-8192" if preferred
        "messages": [
            {"role": "user", "content": prompt}
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
