import requests
import pandas as pd
import json

# Mimic the function in data_agent.py
def get_ai_insights(corr_matrix, api_key):
    print("Sending data to NVIDIA Llama 3 API...")
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    context = f"""
    You are a professional Data Analyst. Analyze the following correlation matrix and provide 3-5 key business insights.
    Focus on strong correlations (positive or negative). Be concise and professional.
    
    Correlation Matrix Data:
    {corr_matrix.to_string()}
    """
    
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [{"role": "user", "content": context}],
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
        return body['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating API insights: {e}"

def test_api():
    # Setup dummy data like the app
    df = pd.read_csv("dummy.csv")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    
    print("Correlation Matrix:")
    print(corr_matrix)
    
    key = "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"
    
    insights = get_ai_insights(corr_matrix, key)
    print("\n--- AI Insights ---")
    print(insights)

if __name__ == "__main__":
    test_api()
