import os
import requests
from dotenv import load_dotenv

load_dotenv()

def truncate_context(context, max_words=800) -> str:
    """Limit the context to avoid long payloads and memory spikes."""
    words = context.split()
    return " ".join(words[:max_words])

def query_groq_llm(context: str, question: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "❌ OpenRouter API key is missing in environment variables."

    context = truncate_context(context)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # 🔹 Replace with your public domain on production
        "X-Title": "pdf-qa-app"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",  # 🔹 Lighter model than GPT-4 etc.
        "messages": [
            {"role": "system", "content": "You are an intelligent assistant."},
            {
                "role": "user",
                "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        "temperature": 0.3,   # 🔹 Lower temp = fewer hallucinations, smaller outputs
        "max_tokens": 150     # 🔹 Limits memory usage
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10  # 🔹 Add timeout to prevent render freeze/crash
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return "⏱️ OpenRouter LLM timeout error. Try again with a shorter context."
    except Exception as e:
        return f"❌ OpenRouter LLM error: {str(e)}"
