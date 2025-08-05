import os
import requests
from dotenv import load_dotenv

load_dotenv()

def truncate_context(context, max_words=800):
    words = context.split()
    return " ".join(words[:max_words])

def query_groq_llm(context, question):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "OpenRouter LLM error: OPENROUTER_API_KEY is not set in environment variables"

    context = truncate_context(context)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # Replace with your domain in production
        "X-Title": "pdf-qa-app"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",  # You can also try llama3, gemma, etc.
        "messages": [
            {"role": "system", "content": "You are an intelligent assistant."},
            {"role": "user", "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"}
        ],
        "temperature": 0.3,
        "max_tokens": 150
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"OpenRouter LLM error: {str(e)}"
