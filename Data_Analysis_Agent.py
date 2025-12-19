import streamlit as st
import pandas as pd
import os
import io
import sys
import streamlit.components.v1 as components
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_json_chat_agent, create_react_agent
from langchain_core.prompts import PromptTemplate
from style_utils import apply_apple_style

# --- TOOLS for ReAct Agents ---

@tool
def run_pandas_code(code: str) -> str:
    """
    Executes Python pandas code on the active dataframe. 
    Context: 
    - 'dfs' is a dictionary of loaded dataframes (keys are filenames like 'df_orders', 'df_users').
    - 'df' is the first dataframe loaded (for convenience, usually the main table).
    Returns the printed output of the code.
    """
    if "dfs" not in st.session_state or not st.session_state.dfs: return "No data."
    dfs = st.session_state.dfs
    df = next(iter(dfs.values()))
    
    # Capture stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    local_vars = {"dfs": dfs, "df": df, "pd": pd}
    
    try:
        # Exec logic
        exec(code, {}, local_vars)
        output = redirected_output.getvalue()
        if len(output) > 2500:
            output = output[:2500] + "\n... (Output truncated)"
            
        if not output.strip():
            return "Code executed successfully but returned no output. Use print() to see results."
        return output
    except Exception as e:
        return f"Execution Error: {e}"
    finally:
        sys.stdout = old_stdout

@tool
def generate_interactive_html(user_request: str) -> str:
    """
    Generates an interactive HTML visualization (Chart.js) snippet.
    Use ONLY for requests to "plot", "graph", "chart", "visualize".
    """
    if "dfs" not in st.session_state or not st.session_state.dfs: return "No data."
    if "api_key" not in st.session_state: return "No API key found."
    
    dfs = st.session_state.dfs
    api_key = st.session_state.api_key
    
    try:
        # Prepare Data Context (Optimized for Token Limit)
        context_str = ""
        for name, d in dfs.items():
            # Summarize columns and types
            dtypes = d.dtypes.to_string()
            # shape
            shape = d.shape
            # Small head
            head = d.head(3).to_markdown()
            context_str += f"\n--- TABLE: {name} (Shape: {shape}) ---\nColumns & Types:\n{dtypes}\nSample Data (Top 3):\n{head}\n"
        
        # Init LLM
        if api_key.startswith("sk-or-"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="meta-llama/llama-3.1-70b-instruct", api_key=api_key, base_url="https://openrouter.ai/api/v1", temperature=0.2)
        else:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key)

        system_prompt = """You are a Frontend Data Visualization Expert.
        Goal: Generate a SINGLE, self-contained HTML snippet to visualize the data based on the user request.
        
        CRITICAL REQUIREMENTS:
        1. Use **TailwindCSS** and **Chart.js** via CDN.
        2. **BACKGROUND MUST BE WHITE (#ffffff)**.
        3. Embed the data from the provided sample directly into the JS.
        4. Return ONLY the raw HTML code.
        """
        user_prompt = f"""
        User Request: "{user_request}"
        Data Context:
        {context_str}
        Generate the HTML code now.
        """
        
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        html_code = response.content.strip()
        
        # Cleanup
        if html_code.startswith("```html"): html_code = html_code[7:]
        elif html_code.startswith("```"): html_code = html_code[3:]
        if html_code.endswith("```"): html_code = html_code[:-3]
        
        # Save for rendering
        if "analysis_plots" not in st.session_state:
            st.session_state.analysis_plots = []
        st.session_state.analysis_plots.append(html_code)
        
        return "Visualization generated and displayed."
        
    except Exception as e:
        return f"HTML Generation failed: {e}"

# --- HELPER: Schema for Code Gen ---
def get_schema_context(dfs):
    context = ""
    for name, df in dfs.items():
        dtypes = df.dtypes.to_string()
        cols = list(df.columns)
        if len(cols) > 200:
             cols = cols[:200]
             cols_str = ", ".join(cols) + f" ... (+{len(df.columns)-200} more)"
        else:
             cols_str = ", ".join(cols)
        
        context += f"Table: {name}\nColumns: {cols_str}\nDtypes:\n{dtypes}\n\n"
    return context

def main():
    st.set_page_config(page_title="Data Analysis Agent", page_icon=None, layout="wide")
    apply_apple_style()
    st.title("Data Analysis Agent")

    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Reset on load if needed or keep history
    if "analysis_plots" not in st.session_state:
        st.session_state.analysis_plots = []
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        default_key = "YOUR_API_KEY_HERE"
        # Secure API Key Loading
        api_key = os.getenv("OPENROUTER_API_KEY", default_key)
        st.session_state.api_key = api_key
        
        st.header("Mode Selection")
        # 5 Options as requested
        agent_mode = st.radio(
            "Select Agent Mode:",
            ["SQL Code", "Python Code", "R Code", "Ask Questions", "Generate Graphs"]
        )
        
        # Checkbox removed in favor of Action Button
        
        st.header("Data Source")
        uploaded_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
        
        if uploaded_files:
            if "dfs" not in st.session_state: st.session_state.dfs = {}
            new_files = False
            for f in uploaded_files:
                safe_name = "df_" + os.path.splitext(f.name)[0].replace(" ", "_").lower()
                if safe_name not in st.session_state.dfs:
                    try:
                        df = pd.read_csv(f)
                        st.session_state.dfs[safe_name] = df
                        new_files = True
                    except Exception as e:
                        st.error(f"Error loading {f.name}: {e}")
            if new_files:
                st.success(f"Loaded {len(st.session_state.dfs)} datasets.")
        else:
            st.session_state.dfs = {}

    # --- Chat Interface ---
    # Clear history if mode changes? Optional. For now let's keep a shared history or clear it.
    # To keep it simple, we just show messages.
    if "last_mode" not in st.session_state or st.session_state.last_mode != agent_mode:
        st.session_state.messages = [] # Clear history on mode switch for clarity
        st.session_state.last_mode = agent_mode
        st.session_state.messages.append({"role": "assistant", "content": f"Switched to **{agent_mode}** mode. How can I help?"})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Render Plots (Top level)
    if st.session_state.analysis_plots:
        with st.chat_message("assistant"):
            for html in st.session_state.analysis_plots:
                components.html(html, height=500, scrolling=True)
        st.session_state.analysis_plots = []

    if prompt := st.chat_input("Input..."):
        st.session_state.last_prompt = prompt # Store for "Download" action
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if "dfs" not in st.session_state or not st.session_state.dfs:
            with st.chat_message("assistant"):
                st.warning("Please upload a CSV file first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner(f"Processing in {agent_mode} mode..."):
                    try:
                        # Init LLM
                        if api_key.startswith("sk-or-"):
                            from langchain_openai import ChatOpenAI
                            llm = ChatOpenAI(model="meta-llama/llama-3.1-70b-instruct", api_key=api_key, base_url="https://openrouter.ai/api/v1")
                        else:
                            from langchain_nvidia_ai_endpoints import ChatNVIDIA
                            llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key)
                        
                        # INTERACTIONS
                        if agent_mode == "Ask Questions":
                            # INTERACTIVE MODE - Calculations Focus
                            tools = [run_pandas_code]
                            template = '''You are a Data Analyst. Answer the user's question by calculating values.
                            TOOLS: {tools}
                            AVAILABLE DATAFRAMES: {df_names}
                            
                            RULES:
                            1. Use `run_pandas_code` to calculate the answer.
                            2. PRINT result.
                            3. Answer in text.
                            
                            Question: {input}
                            Thought:{agent_scratchpad}
                            Action: the action to take, should be one of [{tool_names}]
                            Action Input: the input to the action
                            '''
                            df_names = list(st.session_state.dfs.keys())
                            
                            prompt_template = PromptTemplate.from_template(template).partial(df_names=str(df_names))
                            agent = create_react_agent(llm, tools, prompt_template)
                            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)
                            response = agent_executor.invoke({"input": prompt})
                            output = response['output']
                            
                            # Check for and Offer Download
                            if os.path.exists("analysis_output.csv"):
                                st.success("Analysis Result Generated!")
                                with open("analysis_output.csv", "rb") as f:
                                    st.download_button(
                                        label="Download Result (CSV)",
                                        data=f,
                                        file_name="analysis_result.csv",
                                        mime="text/csv"
                                    )
                                # Clean up handled by overwrite next time or explicit delete if desired
                            
                        elif agent_mode == "Generate Graphs":
                            # INTERACTIVE MODE - Visualization Focus
                            # Clear previous plots to ensure we only show the new one
                            st.session_state.analysis_plots = []
                            
                            tools = [generate_interactive_html, run_pandas_code] 
                            template = '''You are a Data Visualization Specialist.
                            TOOLS: {tools}
                            AVAILABLE DATAFRAMES: {df_names}
                            
                            RULES:
                            1. The user wants **EXACTLY ONE** HTML visualization. NOT a dashboard.
                            2. **Analyze** the request. If it implies multiple insights, pick the **SINGLE MOST CRITICAL** one to visualize.
                            3. Call `generate_interactive_html` **ONCE** and then **STOP**.
                            4. Final Answer: "Graph generated."
                            
                            Question: {input}
                            Thought:{agent_scratchpad}
                            Action: the action to take, should be one of [{tool_names}]
                            Action Input: the input to the action
                            '''
                            df_names = list(st.session_state.dfs.keys())
                            prompt_template = PromptTemplate.from_template(template).partial(df_names=str(df_names))
                            agent = create_react_agent(llm, tools, prompt_template)
                            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=6)
                            response = agent_executor.invoke({"input": prompt})
                            output = response['output']
                            
                        else:
                            # CODE GENERATION MODE (SQL, Python, R)
                            schema_context = get_schema_context(st.session_state.dfs)
                            
                            if agent_mode == "SQL Code":
                                lang = "SQL"
                                instruction = "Generate a valid SQL query for the provided schema tables."
                            elif agent_mode == "Python Code":
                                lang = "Python"
                                instruction = "Generate valid Python pandas code assuming dataframes are loaded as named."
                            else: # R Code
                                lang = "R"
                                instruction = "Generate valid R code (tidyverse) assuming dataframes are loaded."
                                
                            system_prompt = f"""You are a generic Data Coding Assistant.
                            Task: {instruction}
                            Language: {lang}
                            
                            Schema:
                            {schema_context}
                            
                            Return ONLY the code block and a brief explanation.
                            """
                            
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ]
                            response = llm.invoke(messages)
                            output = response.content

                        st.markdown(output)
                        st.session_state.messages.append({"role": "assistant", "content": output})
                        
                        # Show plots if generated
                        if st.session_state.analysis_plots:
                             for html in st.session_state.analysis_plots:
                                components.html(html, height=500, scrolling=True)
                             st.session_state.analysis_plots = []
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- Post-Chat Actions ---
    # Show button for ALL modes that involve data manipulation (excluding pure Viz for now unless requested)
    if agent_mode in ["Ask Questions", "SQL Code", "Python Code", "R Code"] and "last_prompt" in st.session_state:
        st.divider()
        if st.button("Perform Operation and Download CSV", type="primary"):
            with st.spinner("Generating CSV (Direct Execution)..."):
                try:
                    # Re-Init resources
                    if api_key.startswith("sk-or-"):
                         from langchain_openai import ChatOpenAI
                         llm_code = ChatOpenAI(model="meta-llama/llama-3.1-70b-instruct", api_key=api_key, base_url="https://openrouter.ai/api/v1", temperature=0)
                    else:
                         from langchain_nvidia_ai_endpoints import ChatNVIDIA
                         llm_code = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key, temperature=0)
                    
                    # Direct Code Gen Prompt
                    schema_info = get_schema_context(st.session_state.dfs)
                    prompt_text = st.session_state.last_prompt
                    
                    system_prompt = f"""You are a Python Data Analyst.
                    Task: Write Python pandas code to answer the user request and SAVE the result to a CSV file.
                    
                    Dataframes (Loaded in 'dfs' dict):
                    - Keys: {list(st.session_state.dfs.keys())}
                    - Schema: 
                    {schema_info}
                    
                    Requirements:
                    1. Use `dfs['key']` to access dataframes.
                    2. Perform the operation requested by the user.
                    3. Save the final result dataframe to 'analysis_output.csv' using `to_csv('analysis_output.csv', index=False)`.
                    4. Return ONLY valid Python code. No markdown formatting, no explanations.
                    """
                    
                    from langchain_core.messages import HumanMessage, SystemMessage
                    response = llm_code.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt_text)])
                    code = response.content.strip()
                    
                    # Clean Code
                    if code.startswith("```python"): code = code[9:]
                    elif code.startswith("```"): code = code[3:]
                    if code.endswith("```"): code = code[:-3]
                    
                    # Validating Code Safety (Basic)
                    if "os.system" in code or "sys.modules" in code:
                        st.error("Unsafe code detected. Operation aborted.")
                    else:
                        # Prepare execution context
                        local_vars = {"dfs": st.session_state.dfs, "pd": pd}
                        exec(code, {}, local_vars)
                        
                        st.success("Result Generated Successfully!")
                        
                        if os.path.exists("analysis_output.csv"):
                            with open("analysis_output.csv", "rb") as f:
                                st.download_button(
                                    label="Download Result (CSV)",
                                    data=f,
                                    file_name="analysis_result.csv",
                                    mime="text/csv"
                                )
                except Exception as e:
                    st.error(f"Generation Error: {e}")

if __name__ == "__main__":
    main()
