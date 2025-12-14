from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import pandas as pd
import os

# Mock Setup
api_key = "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"
model_name = "meta/llama-3.1-70b-instruct"
os.environ["NVIDIA_API_KEY"] = api_key

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

@tool
def get_dataset_info() -> str:
    """Returns basic analysis of the dataset structure."""
    return "Dataset Info: 3 rows, 2 cols."

@tool
def get_summary_statistics() -> str:
    """Returns the summary statistics of the dataframe."""
    return df.describe().to_markdown()

print("1. Initializing React Agent (Llama 70b)...")
try:
    llm = ChatNVIDIA(model=model_name, temperature=0.1)
    tools = [get_dataset_info, get_summary_statistics]
    
    # Standard React Prompt
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    print("2. Invoking...")
    response = agent_executor.invoke({"input": "What are the summary statistics of the data?"})
    
    print("\n--- Response ---")
    print(response['output'])
    print("\nSUCCESS: React Agent works.")
    
except Exception as e:
    print(f"\nFAILURE: {e}")
