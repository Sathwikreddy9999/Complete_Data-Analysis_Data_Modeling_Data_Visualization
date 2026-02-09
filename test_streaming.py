import os
import time
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GRAPH_API_KEY")
print(f"API Key found: {API_KEY[:10] if API_KEY else 'None'}...")
MODEL_NAME = "google/gemini-flash-1.5" # Using a standard OpenRouter model
BASE_URL = "https://openrouter.ai/api/v1"

def test_stream():
    try:
        llm = ChatOpenAI(
            model=MODEL_NAME, 
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.4,
            streaming=True
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Streaming test success' in 3 words."}
        ]
        
        print("Starting stream...")
        for chunk in llm.stream(messages):
            print(f"Chunk received: '{chunk.content}'")
        print("\nStream finished.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_stream()
