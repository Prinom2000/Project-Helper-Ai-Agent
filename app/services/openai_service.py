# services/openai_service.py

import openai
from fastapi import HTTPException
from app.config import OPENAI_API_KEY
import asyncio

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Function to generate text using OpenAI API
async def generate_text(context: str):
    try:
        # Use asyncio.to_thread to run the blocking OpenAI API call in a separate thread
        response = await asyncio.to_thread(openai.ChatCompletion.create,
                                            model="gpt-3.5-turbo",  # You can change this to "gpt-4" if needed
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant."},
                                                {"role": "user", "content": context}
                                            ],
                                            max_tokens=150,
                                            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
