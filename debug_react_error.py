from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import pandas as pd
import os

# Setup
api_key = "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"
model_name = "meta/llama-3.1-70b-instruct"
os.environ["NVIDIA_API_KEY"] = api_key

# Tools
@tool
def get_dataset_info() -> str:
    """Returns info."""
    return "Dataset Info: 3 rows, 2 cols."

tools = [get_dataset_info]

print("1. Initializing...")
llm = ChatNVIDIA(model=model_name, temperature=0.1)

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
try:
    response = agent_executor.invoke({"input": "What is the info of the dataset?"})
    print(response['output'])
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
