"""services/openai_service.py

Replaced OpenAI direct calls with OpenRouter API (async) so the rest of the
codebase can keep calling `generate_text(context)` unchanged.
"""

from fastapi import HTTPException
import httpx
from app.config import OPENROUTER_API_KEY


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def generate_text(context: str, model: str = "deepseek/deepseek-chat", max_tokens: int = 150):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENROUTER_API_KEY in configuration")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourapp.com",
        "X-Title": "Project Task AI Agent",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context},
        ],
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # extract response text safely
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            # fallback: try other common shapes
            if isinstance(data, dict) and "output" in data:
                return str(data["output"]).strip()
            raise HTTPException(status_code=500, detail=f"Unexpected OpenRouter response shape: {data}")

    except httpx.HTTPStatusError as e:
        detail = f"OpenRouter API error: {e.response.status_code} {e.response.text}"
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
