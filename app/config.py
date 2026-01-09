# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenRouter API Key (for OpenRouter / DeepSeek usage)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

