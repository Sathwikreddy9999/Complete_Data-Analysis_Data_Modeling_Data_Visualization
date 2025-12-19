import streamlit as st
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from style_utils import apply_apple_style
import streamlit.components.v1 as components

# Initialize Session State
if "selected_agents" not in st.session_state:
    st.session_state.selected_agents = []
if "mcp_active" not in st.session_state:
    st.session_state.mcp_active = False
if "sql_connected" not in st.session_state:
    st.session_state.sql_connected = False
if "show_sql_form" not in st.session_state:
    st.session_state.show_sql_form = False

def toggle_agent(agent_name):
    if agent_name in st.session_state.selected_agents:
        st.session_state.selected_agents.remove(agent_name)
    else:
        st.session_state.selected_agents.append(agent_name)

def main():
    st.set_page_config(page_title="IAI Data Manager", page_icon="👔", layout="wide")
    apply_apple_style()
    st.title("👔 AI Data Manager")
    st.caption("Orchestrate your highly autonomous data team.")

    # Sidebar Config
    with st.sidebar:
        st.header("Configuration")
        default_key = "YOUR_API_KEY_HERE"
        # Secure API Key Loading
        api_key = os.getenv("OPENROUTER_API_KEY", default_key)
        
        st.divider()
        if not st.session_state.mcp_active:
             if st.button("🔌 Create MCP Server"):
                with st.spinner("Initializing MCP Server..."):
                    # Simulation of MCP Server Creation
                    st.session_state.mcp_active = True
                    st.success("MCP Server Created on Port 8000")
                    # st.success("Agents connected via MCP protocol.")
                    st.rerun()
        else:
             st.success("✅ MCP Server Active (Port 8000)")

        # Conditional SQL Connection - Shows ALWAYS if not connected
        if not st.session_state.sql_connected:
            # Show Connect Button if form not open
            if not st.session_state.show_sql_form:
                if st.button("🔗 Connect SQL Server"):
                    st.session_state.show_sql_form = True
                    st.rerun()
            
            # Show Credential Form
            if st.session_state.show_sql_form:
                st.markdown("### SQL Server Credentials")
                with st.form("sql_creds"):
                    host = st.text_input("Host", value="localhost")
                    col_u, col_p = st.columns(2)
                    with col_u:
                         username = st.text_input("Username")
                    with col_p:
                         password = st.text_input("Password", type="password")
                    
                    database = st.text_input("Database Name")
                    
                    submitted = st.form_submit_button("Connect", type="primary")
                    if submitted:
                        with st.spinner("Authenticating & Connecting via MCP..."):
                            # Simulation of Connection
                            st.session_state.sql_connected = True
                            # AUTO-INIT MCP if not active
                            if not st.session_state.mcp_active:
                                st.session_state.mcp_active = True
                                st.success("MCP Server Auto-Started.")
                                
                            st.session_state.show_sql_form = False
                            st.success(f"Connected to {database}@{host} via MCP!")
                            st.rerun()
            
            if st.button("Back", key="cancel_sql") if st.session_state.show_sql_form else False:
                 st.session_state.show_sql_form = False
                 st.rerun()
                     
        else:
            st.success("✅ SQL Server Active")

    # 1. Data Upload (Centralized)
    st.subheader("1. Central Data Hub")
    uploaded_files = st.file_uploader("Upload Dataset(s) (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True)
    
    if uploaded_files:
        # Initialize dictionary to store multiple dfs if not present or if re-uploading logic needs handling
        # For simplicity, we'll rebuild it or update it. 
        # Better to check if we already processed these exact files? 
        # Streamlit re-runs script on interaction, so we can just process active uploads.
        
        if "dfs" not in st.session_state:
            st.session_state.dfs = {}
            
        file_names = []
        for uploaded_file in uploaded_files:
            file_names.append(uploaded_file.name)
            if uploaded_file.name not in st.session_state.dfs:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_temp = pd.read_csv(uploaded_file)
                    else:
                        df_temp = pd.read_excel(uploaded_file)
                    st.session_state.dfs[uploaded_file.name] = df_temp
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {e}")
        
        # Dataset Selector
        if st.session_state.dfs:
            selected_file = st.selectbox("Select Active Dataset", list(st.session_state.dfs.keys()), key="mgr_active_dataset")
            
            # Set Global Active DF for agents
            st.session_state.df = st.session_state.dfs[selected_file]
            
            # Preview
            st.caption(f"Previewing: {selected_file} ({len(st.session_state.df)} rows)")
            st.dataframe(st.session_state.df.head())
            
            st.success(f"Active Data: {selected_file} shared across all agents.")
    
    # Check if df exists but no files currently in uploader (e.g. cleared but session remains?)
    # Streamlit clears uploaded_files if removed from UI. 
    # Valid state requires at least one df.

    st.divider()

    # 2. Workflow Orchestration
    st.subheader("2. Design Your Workflow")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_request = st.text_area("Describe your goal (AI Auto-Plan):", 
                                    placeholder="E.g., Clean the data, then run a regression model.",
                                    height=100)
    
    with col2:
        st.write("OR Select Manually below:")
        st.caption("Click buttons in order of execution.")

    # 3. Agent Toggles (The "Switch" Logic)
    st.subheader("3. Agent Squad")
    
    # Define Agents
    agents = ["Data_Cleaning_agent", "Data_Analysis_agent", "Data_Visulization_agent", "Data_Science_agent"]
    
    cols = st.columns(len(agents))
    
    for i, agent_name in enumerate(agents):
        with cols[i]:
            # determine visual state
            is_selected = agent_name in st.session_state.selected_agents
            order = st.session_state.selected_agents.index(agent_name) + 1 if is_selected else None
            
            label = f"{agent_name.replace('Data_', '').replace('_agent', '')}"
            if order:
                label = f"#{order} {label} ✅"
            else:
                label = f"{label} ⚪"
            
            # Button acts as toggle
            if st.button(label, key=f"btn_{agent_name}", use_container_width=True):
                toggle_agent(agent_name)
                st.rerun()
            
            # Contextual Options for Cleaning Agent
            if is_selected and agent_name == "Data_Cleaning_agent":
                st.multiselect(
                    "Advanced Cleaning",
                    ["Colors/Nominal", "Missing Values", "Near-Duplicates", "Mixed Formats", "Outliers"],
                    key="mgr_clean_opts",
                    help="Select cleaning steps to apply automatically."
                )

            # Contextual Options for Analysis Agent
            if is_selected and agent_name == "Data_Analysis_agent":
                st.selectbox(
                    "Analysis Mode",
                    ["Auto-Summary", "Ask Questions", "Generate Graphs", "SQL Code", "Python Code", "R Code"],
                    key="mgr_analysis_mode",
                    help="Select how the Analysis Agent should behave."
                )
                
                # Checkbox for CSV Generation (Only visible effectively for Ask Questions/Code modes implied)
                # Checkbox for CSV Generation (Visible for Ask Questions and Code Generation modes)
                if st.session_state.mgr_analysis_mode in ["Ask Questions", "SQL Code", "Python Code", "R Code"]:
                    st.checkbox("Generate Downloadable CSV", key="mgr_gen_csv", help="If checked, the agent will save the operation result to a CSV file.")

            # Contextual Options for Visualization Agent
            if is_selected and agent_name == "Data_Visulization_agent":
                st.selectbox(
                    "Business Domain",
                     ["Banking", "Utility", "Retail", "Healthcare", "Manufacturing", "Marketing", "Sales", "HR"],
                     key="mgr_viz_domain"
                )
                st.selectbox(
                     "Dashboard Style",
                     ["Power BI", "Tableau", "Looker", "QlikSense", "Google Data Studio"],
                     key="mgr_viz_style"
                )

            # Contextual Options for Data Science Agent
            if is_selected and agent_name == "Data_Science_agent":
                if "df" in st.session_state:
                    df = st.session_state.df
                    # Guess target (last col)
                    default_target_idx = len(df.columns) - 1
                    
                    st.selectbox(
                        "Target Variable (Y)",
                        df.columns,
                        index=default_target_idx,
                        key="mgr_ds_target",
                        help="Select the column you want to predict."
                    )
                    
                    # Default to all other columns? Or let user pick? 
                    # If we pre-select all, the input box might be huge. Let's leave empty implies "All/Auto".
                    st.multiselect(
                        "Predictor Variables (X)",
                        [c for c in df.columns], # Can select target too? Logic in tool handles it? Better exclude target in list? 
                        # Actually tool drops target from df first, so selecting target in X would fail "c in X.columns" check after drop.
                        # Ideally exclude target from options, but target changes. 
                        # Simplest is just list all, user knows not to pick target.
                        key="mgr_ds_features",
                        help="Leave empty to use all other columns."
                    )
                else:
                    st.caption("Upload data to see Model options.")

    st.divider()

    # Execution Block
    if st.button("🚀 Execute Workflow", type="primary", use_container_width=True):
        if not api_key:
            st.error("API Key required.")
            return

        final_plan = []
        
        # Priority: Manual Selection > AI Generation
        if st.session_state.selected_agents:
            final_plan = st.session_state.selected_agents
            st.info(f"Executing Manual Plan: {' -> '.join(final_plan)}")
        elif user_request:
            # AI Planner Fallback
            with st.spinner("AI Generating Plan..."):
                try:
                    llm = ChatOpenAI(
                        model="meta-llama/llama-3.1-70b-instruct",
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                        temperature=0.1
                    )
                    system_prompt = """
You are the **AI Data Manager**. Map the request to a sequence of: Data_Cleaning_agent, Data_Analysis_agent, Data_Visulization_agent, Data_Science_agent.
Return ONLY comma-separated names.
"""
                    messages = [("system", system_prompt), ("human", user_request)]
                    response = llm.invoke(messages)
                    final_plan = [a.strip() for a in response.content.strip().split(",")]
                    st.info(f"AI Selected Plan: {' -> '.join(final_plan)}")
                except Exception as e:
                    st.error(f"AI Planning Failed: {e}")
                    return
        else:
            st.warning("Please select agents or describe a task.")
            return

        # --- WORKFLOW VALIDATION (Strict Rules) ---
        # Rule 3 & 4: Viz and DS are terminal.
        # Rule 2: Analysis -> Next ONLY allowed if CSV Generation is checked.
        
        for i in range(len(final_plan) - 1):
             current = final_plan[i]
             next_agent = final_plan[i+1]
             
             # Rule 3 & 4: Terminal Agents
             if current == "Data_Visulization_agent":
                 st.error(f"❌ Operation Not Allowed: Data Visualization Agent is terminal. It cannot pass data to {next_agent}.")
                 return
             if current == "Data_Science_agent":
                 st.error(f"❌ Operation Not Allowed: Data Science Agent is terminal. It cannot pass data to {next_agent}.")
                 return
                 
             # Rule 2: Analysis -> Next
             if current == "Data_Analysis_agent":
                 # Check if CSV download is enabled
                 gen_csv = st.session_state.get("mgr_gen_csv", False)
                 # Also verify mode is compatible (Ask Questions / Code) - already implied by sidebar visibility but safest to check var
                 mode = st.session_state.get("mgr_analysis_mode", "Auto-Summary")
                 valid_modes = ["Ask Questions", "SQL Code", "Python Code", "R Code"]
                 
                 if not (gen_csv and mode in valid_modes):
                     st.error(f"❌ Operation Not Allowed: Data Analysis Agent can only pass data to {next_agent} if 'Generate Downloadable CSV' is ENABLED.")
                     return

        # Execution Simulation & Real Output Generation
        st.divider()
        st.subheader("📺 Output Window")
        
        progress_bar = st.progress(0)
        
        import io
        import json
        # Import Agent Capabilities (Lazy import to avoid circular dep issues if any, though scripts are independent)
        # Note: We rely on the files being in the same directory
        try:
            from Data_Cleaning_Agent import get_column_profile, apply_advanced_cleaning
            from Data_Visualization_agent import generate_design_prompt, generate_dashboard_html
            from Data_Analysis_Agent import run_pandas_code, generate_interactive_html, get_schema_context
            from langchain.agents import create_react_agent, AgentExecutor
            from langchain_core.prompts import PromptTemplate
            from Data_Science_Agent import run_automl, generate_plot
        except ImportError as e:
            st.error(f"Could not import agent modules: {e}")
            return

        for idx, agent in enumerate(final_plan):
            with st.status(f"Running {agent}...", expanded=True) as status:
                
                # --- DATA CLEANING ---
                if agent == "Data_Cleaning_agent":
                    st.write("**Action:** Profiling & Checking Data Quality")
                    if "df" in st.session_state:
                        # Check for advanced options from the UI
                        # We stored them in session state via key="mgr_clean_opts" in the sidebar loop
                        # But wait, keys inside loops need to be unique or handled carefully.
                        # The key above was 'mgr_clean_opts'. Streamlit syncs it to st.session_state['mgr_clean_opts']
                        
                        clean_opts = st.session_state.get("mgr_clean_opts", [])
                        
                        if clean_opts:
                            st.write(f"Applying: {', '.join(clean_opts)}")
                            clean_df, logs = apply_advanced_cleaning(st.session_state.df, clean_opts)
                            st.session_state.df = clean_df # UPDATE GLOBAL DATAFRAME for subsequent agents
                            
                            st.markdown("### 🧹 Advanced Cleaning Report")
                            for log in logs:
                                st.caption(f"• {log}")
                            
                            st.download_button(
                                label="Download Cleaned Data (CSV)",
                                data=clean_df.to_csv(index=False),
                                file_name="cleaned_data_manager.csv",
                                mime="text/csv"
                            )
                        else:
                            # Default profiling if no advanced options selected
                            profile = get_column_profile(st.session_state.df)
                            st.markdown("### 🧹 Data Cleaning Report (Profile)")
                            st.json(profile) 
                    else:
                        st.warning("No data loaded to clean.")
                
                # --- DATA ANALYSIS ---
                elif agent == "Data_Analysis_agent":
                    mode = st.session_state.get("mgr_analysis_mode", "Auto-Summary")
                    st.write(f"**Action:** {mode}")
                    
                    if "df" in st.session_state:
                         df = st.session_state.df
                         # Prep environment for Analysis Agent Tools
                         # Analysis Agent usually expects 'dfs' dict in session state
                         # Only create if missing (to preserve multi-file uploads)
                         if "dfs" not in st.session_state or not st.session_state.dfs:
                             st.session_state.dfs = {"main_data": df}
                         
                         if mode == "Auto-Summary":
                             desc = df.describe()
                             st.markdown("### 📊 Statistical Analysis")
                             st.text(desc.to_markdown()) 
                         
                         elif mode in ["Ask Questions", "Generate Graphs"]:
                             # Interactive ReAct Agent
                             prompt_text = user_request if user_request else (
                                 "Analyze the dataset and provide key insights." if mode == "Ask Questions" 
                                 else "Create a meaningful visualization for this dataset."
                             )
                             
                             try:
                                 # Init LLM
                                 llm_an = ChatOpenAI(model="meta-llama/llama-3.1-70b-instruct", api_key=api_key, base_url="https://openrouter.ai/api/v1")
                                 
                                 if mode == "Ask Questions":
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
                                     agent_an = create_react_agent(llm_an, tools, prompt_template)
                                     agent_exec = AgentExecutor(agent=agent_an, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)
                                     response = agent_exec.invoke({"input": prompt_text})
                                     st.markdown("### 💬 Analysis Results")
                                     st.markdown(response['output'])

                                 elif mode == "Generate Graphs":
                                      # Clear previous plots
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
                                      agent_an = create_react_agent(llm_an, tools, prompt_template)
                                      agent_exec = AgentExecutor(agent=agent_an, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=6)
                                      response = agent_exec.invoke({"input": prompt_text})
                                      st.markdown(f"**Agent Status:** {response['output']}")
                                      
                                      if "analysis_plots" in st.session_state and st.session_state.analysis_plots:
                                          st.markdown("### 📊 Generated Graph")
                                          for html in st.session_state.analysis_plots:
                                              components.html(html, height=500, scrolling=True)
                                          st.session_state.analysis_plots = [] # Clear
                             
                             except Exception as e:
                                 st.error(f"Analysis Agent Error: {e}")

                         elif mode in ["SQL Code", "Python Code", "R Code"]:
                            # Code Generation
                            prompt_text = user_request if user_request else "Generate code to analyze this data."
                            schema = get_schema_context(st.session_state.dfs)
                            
                            lang = mode.split(" ")[0]
                            system_prompt = f"You are a {lang} Coding Assistant. Return ONLY valid {lang} code for the schema below.\n\nSchema:\n{schema}"
                            
                            llm_an = ChatOpenAI(model="meta-llama/llama-3.1-70b-instruct", api_key=api_key, base_url="https://openrouter.ai/api/v1")
                            resp = llm_an.invoke([("system", system_prompt), ("user", prompt_text)])
                            
                            st.markdown(f"### 💻 {lang} Code")
                            st.code(resp.content, language=lang.lower())

                     
                         # --- COMMON CSV GENERATION LOGIC (Refactored Direct Execution) ---
                         if st.session_state.get("mgr_gen_csv", False) and mode in ["Ask Questions", "SQL Code", "Python Code", "R Code"]:
                             with st.spinner("Generating Downloadable CSV (Direct Execution)..."):
                                 try:
                                     # Re-Init specific Code Gen LLM
                                     llm_code = ChatOpenAI(model="meta-llama/llama-3.1-70b-instruct", api_key=api_key, base_url="https://openrouter.ai/api/v1", temperature=0)
                                     
                                     # Context
                                     schema_info = get_schema_context(st.session_state.dfs)
                                     
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
                                     
                                     if "os.system" in code or "sys.modules" in code:
                                         st.error("Unsafe code generated. Skipped execution.")
                                     else:
                                         # Execute
                                         local_vars = {"dfs": st.session_state.dfs, "pd": pd}
                                         exec(code, {}, local_vars)
                                         
                                         if os.path.exists("analysis_output.csv"):
                                             with open("analysis_output.csv", "rb") as f:
                                                 st.download_button(
                                                     label="💾 Download Analysis Result (CSV)",
                                                     data=f,
                                                     file_name="manager_analysis_result.csv",
                                                     mime="text/csv"
                                                 )
                                             st.success("CSV generated successfully.")
                                         
                                         # --- DATA PASSING (Rule 2 Support) ---
                                         try:
                                             if os.path.exists("analysis_output.csv"):
                                                 new_df = pd.read_csv("analysis_output.csv")
                                                 st.session_state.df = new_df
                                                 # Also update 'main_data' in dfs to keep sync
                                                 st.session_state.dfs['main_data'] = new_df
                                                 st.info("🔄 Output passed to next agent.")
                                         except Exception as e:
                                             st.warning(f"Could not pass data to next agent: {e}")
                                 except Exception as e:
                                     st.error(f"CSV Gen Failed: {e}")

                    else:
                         st.warning("No data loaded to analyze.")

                # --- DATA VISUALIZATION ---
                elif agent == "Data_Visulization_agent":
                    domain = st.session_state.get("mgr_viz_domain", "General Business")
                    style = st.session_state.get("mgr_viz_style", "Tableau")
                    
                    st.write(f"**Action:** Interactive Dashboard ({style} / {domain})")
                    if "df" in st.session_state:
                        df = st.session_state.df
                        # Prepare Context
                        df_head = df.head().to_csv(index=False)
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        df_info = buffer.getvalue()
                        
                        try:
                            # 1. Generate Design
                            design_prompt = generate_design_prompt(df_head, df_info, domain, style, api_key)
                            
                            # 2. Generate Code
                            html_code = generate_dashboard_html(df_head, df_info, domain, style, design_prompt, api_key)
                            
                            st.markdown("### 🎨 Visualization Output")
                            if "Error" in html_code:
                                st.error(html_code)
                            else:
                                components.html(html_code, height=600, scrolling=True)
                                st.caption("Interactive Dashboard (Tableau Style)")
                                
                        except Exception as e:
                            st.error(f"Viz Error: {e}")
                    else:
                        st.warning("No data loaded to visualize.")

                # --- DATA SCIENCE ---
                elif agent == "Data_Science_agent":
                    st.write("**Action:** Running AutoML Pipeline")
                    if "df" in st.session_state:
                        df = st.session_state.df
                        
                        # retrieve explicit selections
                        target = st.session_state.get("mgr_ds_target")
                        features = st.session_state.get("mgr_ds_features", [])
                        
                        # Fallback if logic failed or no selection yet (e.g. ran workflow without looking at sidebar?)
                        if not target:
                             target = df.columns[-1]
                        
                        st.write(f"Target Variable: `{target}`")
                        if features:
                            st.write(f"Predictors: {features}")
                        else:
                            st.write("Predictors: All available columns")
                        
                        # run_automl is a LangChain tool, we can call it directly via .func
                        # or invoke it.
                        try:
                            # Determine task type
                            is_numeric = pd.api.types.is_numeric_dtype(df[target])
                            task_type = "regression" if is_numeric and df[target].nunique() > 10 else "classification"
                            
                            # We need to temporarily set 'df' in st.session_state for the tool to pick it up?
                            # It is already there since manager uses st.session_state.df!
                            # And since we imported the function, it will look at st.session_state of THIS process.
                            # Perfect.
                            
                            # Pass feature_cols only if selected
                            result = run_automl.func(
                                target_col=target, 
                                task_type=task_type, 
                                feature_cols=features if features else None
                            )
                            st.markdown("### 🧪 AutoML Results")
                            st.markdown(result) # Text/Table Output
                            
                            # --- POST-UPDATE: GENERATE PLOTS ---
                            st.write("Generating Relationship Plots...")
                            
                            # 1. Correlation Matrix
                            generate_plot.func("heatmap")
                            
                            # 2. Target Distribution
                            generate_plot.func("histogram", x_col=target)
                            
                            if "plots" in st.session_state and st.session_state.plots:
                                st.markdown("### 📊 Data Science Visuals")
                                col_p1, col_p2 = st.columns(2)
                                
                                # We expect 2 plots typically, but let's handle dynamic
                                for i, fig in enumerate(st.session_state.plots):
                                    with col_p1 if i % 2 == 0 else col_p2:
                                        st.pyplot(fig)
                                
                                st.session_state.plots = [] # Clear after showing
                                
                        except Exception as e:
                            st.error(f"AutoML Failed: {e}")
                    else:
                         st.warning("No data for AutoML.")
                
                status.update(label=f"{agent} Complete!", state="complete", expanded=False)
            
            progress_bar.progress((idx + 1) / len(final_plan))
            
        st.success("Workflow Execution Completed!")

if __name__ == "__main__":
    main()
