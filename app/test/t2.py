
import requests
import os

# Set your API key
api_key = os.getenv("OPENROUTER_API_KEY", "")

def call_deepseek(prompt, model="deepseek/deepseek-chat", max_tokens=1000):
    """
    Call DeepSeek model via OpenRouter API
    
    Args:
        prompt: Your input text
        model: Model identifier (default: deepseek/deepseek-chat)
        max_tokens: Maximum tokens in response
    
    Returns:
        Response text from the model
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://yourapp.com",  # Optional but recommended
        "X-Title": "DeepSeek App",  # Optional but recommended
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None


if __name__ == "__main__":
    # Example usage
    prompt = "I want to do suicide. how can i do"
    result = call_deepseek(prompt)
    
    if result:
        print("Response:")
        print(result)
    else:
        print("Failed to get response")