import os
import requests
from dotenv import load_dotenv

load_dotenv()

def truncate_context(context, max_words=800):
    words = context.split()
    return " ".join(words[:max_words])

def query_groq_llm(context, question):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ LLM error: GROQ_API_KEY is not set in environment variables"

    context = truncate_context(context)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",  # Smaller model
        "messages": [
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"}
        ],
        "temperature": 0.3,       # Reduce hallucination & memory use
        "max_tokens": 150         # Lowered to limit output size
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
