import pandas as pd
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import io
import os

# Mock Setup
api_key = "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"
model_name = "meta/llama-3.1-70b-instruct"

print("1. Loading Data...")
df = pd.read_csv("test_data.csv")
print(f"Loaded {len(df)} rows.")

print("2. Generating Stats...")
buffer = io.StringIO()
df.info(buf=buffer)
info_text = buffer.getvalue()
desc_text = df.describe().to_markdown()
head_text = df.head().to_markdown()
context = f"Dataset Info:\n{info_text}\n\nStats:\n{desc_text}\n\nFirst Rows:\n{head_text}"

print("3. Calling NVIDIA API (Auto-Summary)...")
try:
    llm = ChatNVIDIA(model=model_name, nvidia_api_key=api_key)
    prompt = f"You are a Data Scientist. Analyze the following dataset summary and provide a comprehensive executive summary of the data. Use bullet points.\n\n{context}"
    
    response = llm.invoke(prompt)
    print("\n--- Response Received ---")
    print(response.content)
    print("--- End Response ---")
    print("\nSUCCESS: NVIDIA API integration is working.")
except Exception as e:
    print(f"\nFAILURE: {e}")
