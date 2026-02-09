from langchain_openai import ChatOpenAI
import os

api_key = "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"
model = "google/gemini-2.0-flash-exp:free" # Trying the known working one first, then the user's request

try:
    print(f"Testing {model} with OpenRouter...")
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1
    )
    response = llm.invoke("Hello, are you working?")
    print(f"Success! Response: {response.content}")
except Exception as e:
    print(f"Error: {e}")

model2 = "google/gemini-3-flash-preview"
try:
    print(f"\nTesting {model2} with OpenRouter...")
    llm = ChatOpenAI(
        model=model2,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1
    )
    response = llm.invoke("Hello, are you working?")
    print(f"Success! Response: {response.content}")
except Exception as e:
    print(f"Error with {model2}: {e}")
