from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

api_key = "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"
os.environ["NVIDIA_API_KEY"] = api_key

try:
    print("Attempting to list models...")
    models = ChatNVIDIA.get_available_models()
    print(f"Found {len(models)} models.")
    for m in models:
        print(f"- {m.id}")
        
    print("\nTesting basic invoke with default model...")
    llm = ChatNVIDIA()
    print(llm.invoke("Hello").content)
    
except Exception as e:
    print(f"Error: {e}")
