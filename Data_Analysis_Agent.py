import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import sys
import re
import streamlit.components.v1 as components
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tabulate import tabulate
import xgboost as xgb
import lightgbm as lgb
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
        exec(code, local_vars)
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
    Generates a high-quality interactive HTML visualization (Chart.js/Plotly) snippet.
    Use ONLY for requests to "plot", "graph", "chart", "visualize".
    """
    if "dfs" not in st.session_state or not st.session_state.dfs: return "No data."
    if "api_key" not in st.session_state: return "No API key found."
    
    dfs = st.session_state.dfs
    api_key = st.session_state.api_key
    
    try:
        # 1. Prepare Rich Data Context
        context_str = ""
        for name, d in dfs.items():
            buffer = io.StringIO()
            d.info(buf=buffer)
            df_info = buffer.getvalue()
            head_csv = d.head(5).to_csv(index=False)
            context_str += f"\n--- TABLE: {name} (Shape: {d.shape}) ---\n"
            context_str += f"SCHEMA & INFO:\n{df_info}\n"
            context_str += f"SAMPLE DATA (CSV):\n{head_csv}\n"
        
        # 2. Init LLM
        # Pull key from session state (OpenRouter preferred for Gemini)
        graph_api_key = st.session_state.get("openrouter_key", "")
        if not graph_api_key:
            return "Error: OpenRouter API Key missing. Please configure in sidebar."
            
        if graph_api_key.startswith("sk-or-"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="google/gemini-3-flash-preview", api_key=graph_api_key, base_url="https://openrouter.ai/api/v1", temperature=0.2)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="google/gemini-3-flash-preview", api_key=graph_api_key)

        # 3. Sophisticated Designer-Coder Prompt
        system_prompt = """You are a Senior Design Architect & Frontend Visualization Expert.
        
        CRITICAL MANDATE: You MUST build EXCLUSIVELY ONE professional-grade interactive visualization. Do NOT generate multiple charts, grids, or dashboards.
        
        DESIGN PRINCIPLES:
        1. **Single Focus**: Choose the SINGLE most effective chart type (Bar, Line, etc.) to answer the user's specific question.
        2. **Aesthetics**: Use modern, business-professional color palettes.
        3. **Background**: The container background MUST be WHITE (#ffffff).
        
        TECHNICAL REQUIREMENTS:
        1. Use **TailwindCSS** for layout and **Chart.js** (or Plotly) via CDN.
        2. Embed necessary data from the provided sample directly as JS variables.
        3. The result must be a single, self-contained HTML snippet.
        4. Return ONLY the raw HTML code. Do NOT include explanations or multiple containers.
        """
        
        user_prompt = f"""
        User Question: "{user_request}"
        
        DATA CONTEXT:
        {context_str}
        
        STRICT INSTRUCTIONS:
        1. Analyze the Question and find the most relevant table/columns.
        2. Generate ONLY ONE high-quality interactive chart. If the question implies multiple views, select the most important one.
        3. MANDATORY: The output must contain ONLY one `<canvas>` or one Plotly `<div>`. 
        4. Ensure axes are labeled and the title is descriptive.
        
        Generate the SINGLE HTML code block now.
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
        st.session_state.analysis_plots.append({"html": html_code})
        
        return "Visualization generated and displayed."
        
    except Exception as e:
        return f"HTML Generation failed: {e}"




@tool
def run_linear_regression(target_col: str, feature_cols: list[str]) -> str:
    """
    Runs a Linear Regression model.
    Args:
        target_col (str): The column to predict.
        feature_cols (list[str]): List of predictor columns.
    """
    if "dfs" not in st.session_state or not st.session_state.dfs: return "No data."
    df = next(iter(st.session_state.dfs.values()))
    
    try:
        model_df = df[[target_col] + feature_cols].dropna()
        X = model_df[feature_cols]
        y = model_df[target_col]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return str(model.summary())
    except Exception as e:
        return f"Regression failed: {e}"

@tool
def run_manova(dependent_vars: list[str], independent_var: str) -> str:
    """
    Runs MANOVA (Multivariate Analysis of Variance).
    Args:
        dependent_vars (list[str]): List of numeric dependent variables.
        independent_var (str): The categorical grouping variable.
    """
    if "dfs" not in st.session_state or not st.session_state.dfs: return "No data."
    df = next(iter(st.session_state.dfs.values()))
    
    try:
        deps_str = " + ".join(dependent_vars)
        formula = f"{deps_str} ~ {independent_var}"
        manova = MANOVA.from_formula(formula, data=df)
        return str(manova.mv_test())
    except Exception as e:
        return f"MANOVA failed: {e}"

@tool
def run_automl(target_col: str, task_type: str = 'classification', feature_cols: list[str] = None) -> str:
    """
    Runs AutoML to find the best model (RF, XGB, LightGBM, Neural Networks, etc.).
    Args:
        target_col (str): The name of the target column.
        task_type (str): 'classification' or 'regression'.
        feature_cols (list[str]): Optional list of columns to use as predictors.
    """
    if "dfs" not in st.session_state or not st.session_state.dfs: return "No data."
    df = next(iter(st.session_state.dfs.values())).copy()
    
    if target_col not in df.columns:
        return f"Error: Target column '{target_col}' not found."
    
    try:
        # 1. Preprocessing
        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if feature_cols:
             valid_cols = [c for c in feature_cols if c in X.columns]
             if valid_cols: X = X[valid_cols]
             else: return "Error: Selected features not found."
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        # 2. Models (Including Neural Networks)
        models = {}
        if task_type.lower() == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
            models = {
                'RandomForest': (RandomForestClassifier(), {'model__n_estimators': [50, 100]}),
                'XGBoost': (xgb.XGBClassifier(eval_metric='logloss'), {'model__learning_rate': [0.01, 0.1]}),
                'Neural Network (MLP)': (MLPClassifier(max_iter=500), {'model__hidden_layer_sizes': [(50,), (100,)], 'model__alpha': [0.0001, 0.001]}),
                'LightGBM': (lgb.LGBMClassifier(verbose=-1), {'model__n_estimators': [50, 100]})
            }
            metric = 'accuracy'
        else:
            models = {
                'RandomForest': (RandomForestRegressor(), {'model__n_estimators': [50, 100]}),
                'XGBoost': (xgb.XGBRegressor(), {'model__learning_rate': [0.01, 0.1]}),
                'Neural Network (MLP)': (MLPRegressor(max_iter=500), {'model__hidden_layer_sizes': [(50,), (100,)], 'model__alpha': [0.0001, 0.001]}),
                'LightGBM': (lgb.LGBMRegressor(verbose=-1), {'model__n_estimators': [50, 100]})
            }
            metric = 'r2'

        # 3. Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = []
        best_score = -np.inf
        best_model_name = ""
        
        for name, (model, params) in models.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            search = RandomizedSearchCV(pipe, params, n_iter=3, cv=2, scoring=metric, n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
            score = search.score(X_test, y_test)
            results.append([name, f"{score:.4f}"])
            if score > best_score:
                best_score = score
                best_model_name = name

        table = tabulate(results, headers=["Model", f"Score ({metric})"], tablefmt="github")
        return f"### AutoML Results for {target_col}\nBest Model: **{best_model_name}** ({best_score:.4f})\n\n{table}"
        
    except Exception as e:
        return f"AutoML failed: {e}"

def get_schema_context(dfs):
    """
    Returns a string context describing the tables and columns for LLM prompts.
    Uses safe SQL names for consistency.
    """
    context = ""
    for name, df in dfs.items():
        safe_name = name.replace(".", "_").replace("-", "_").replace(" ", "_")
        cols_str = ", ".join(list(df.columns))
        dtypes = df.dtypes.to_string()
        context += f"Table: {safe_name}\nColumns: {cols_str}\nDtypes:\n{dtypes}\n\n"
    return context

def execute_sql_query(query, dfs):
    """
    Executes a SQL query on a dictionary of dataframes using sqlite3.
    Returns the result as a pandas DataFrame.
    """
    import sqlite3
    conn = sqlite3.connect(":memory:")
    try:
        # Register tables
        for name, df in dfs.items():
            # Ensure names are safe and MATCH get_schema_context
            safe_name = name.replace(".", "_").replace("-", "_").replace(" ", "_")
            df.to_sql(safe_name, conn, index=False)
        
        # Execute query
        result_df = pd.read_sql_query(query, conn)
        return result_df
    finally:
        conn.close()

def auto_preprocess(df, target_col=None):
    """
    Deterministically cleans data: handles NaNs, encodes categories, and scales numerics.
    Useful for ensuring no 'string-to-float' errors occur.
    """
    df = df.copy()
    if target_col and target_col in df.columns:
        # Drop rows where target is NaN before splitting
        df = df.dropna(subset=[target_col])
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    # 1. Drop obvious IDs
    id_cols = [c for c in X.columns if 'id' in c.lower() or 'vin' in c.lower() or 'name' in c.lower()]
    X = X.drop(columns=id_cols)

    # 2. Identify types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # 3. Simple Cleaning & Encoding
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    
    # One-Hot Encoding for small categories
    X = pd.get_dummies(X, columns=[c for c in cat_cols if X[c].nunique() < 50], drop_first=True)
    
    # 4. Final scrub: remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # 5. Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if not X.empty:
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        return (X_scaled, y) if y is not None else X_scaled
    return (X, y) if y is not None else X

def generate_combined_insights(dfs, api_key):
    """
    Generates structured insights grouped by table, including cross-table relationships.
    """
    try:
        # Robust LLM Init based on Key
        if api_key.startswith("sk-or-"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="nvidia/nemotron-3-nano-30b-a3b:free", api_key=api_key, base_url="https://openrouter.ai/api/v1", temperature=0.3)
        else:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key, temperature=0.3)
        
        context = ""
        for name, df in dfs.items():
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            context += f"Table: {name}\nShape: {df.shape}\nColumns: {list(df.columns)}\nNumeric Columns: {numeric_cols}\nSample:\n{df.head(3).to_string()}\n\n"
            
        system_prompt = "You are a Senior Principal Data Analyst. Goal: Provide structured, per-table analysis for multiple datasets."
        user_prompt = f"""
        Analyze the following datasets. Organize your response as follows:
        1. For EACH table, provide EXACTLY 3 high-density bullet points.
        2. Format: **[Table Name] Insights**
           - Bullet 1
           - Bullet 2
           - Bullet 3
        
        STRICT RULE: Only 3 bullets per table. Focus on:
        - Data scale and missingness.
        - Primary key or distribution of key numeric features.
        - Strategic observation or potential analysis pathway.
        
        Data Context:
        {context}
        
        Summaries:
        """
        
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = response.content.strip()
        
        # Parse into a dictionary: {Table Name: summary_text}
        summaries = {}
        # Splitting by bolded table names: **[Table Name] Insights**
        parts = re.split(r"\*\*\[?(.*?)\]? Insights\*\*", content)
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                table_name = parts[i].strip()
                summary_text = parts[i+1].strip() if i+1 < len(parts) else ""
                summaries[table_name] = summary_text
        else:
            # Fallback for unexpected format
            summaries["Data"] = content
            
        return summaries
    except Exception as e:
        return {"Error": str(e)}

def get_advanced_stats(df):
    """
    Computes advanced statistics: IQR, Skewness, Kurtosis, Correlation, etc.
    Returns a formatted string summary.
    """
    summary = ""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty: return "No numerical columns for advanced stats."
    
    # Basic Stats
    desc = num_df.describe().T
    desc['median'] = num_df.median()
    desc['mode'] = num_df.mode().iloc[0]
    desc['iqr'] = num_df.quantile(0.75) - num_df.quantile(0.25)
    desc['skew'] = num_df.skew()
    desc['kurt'] = num_df.kurtosis()
    
    summary += "### Column Profiles (Numerical)\n"
    summary += desc[['mean', 'median', 'mode', 'std', 'min', 'max', 'iqr', 'skew', 'kurt']].to_string()
    
    # Correlation Matrix (Top 5 most correlated pairs for brevity)
    if len(num_df.columns) > 1:
        corr = num_df.corr().unstack().sort_values(ascending=False)
        corr = corr[corr < 1].head(10) # Top 5 pairs (each pair appears twice)
        summary += "\n\n### Top Correlations\n"
        summary += corr.to_string()
        
    return summary

def render_analysis_plots():
    """
    Renders all visualizations stored in session state and provides download buttons.
    """
    if "analysis_plots" in st.session_state and st.session_state.analysis_plots:
        for i, plot_data in enumerate(st.session_state.analysis_plots):
            if isinstance(plot_data, dict) and "html" in plot_data:
                components.html(plot_data["html"], height=500, scrolling=True)
                st.download_button(
                    label="Download Chart (HTML)",
                    data=plot_data["html"],
                    file_name="visualization.html",
                    mime="text/html",
                    key=f"dl_{i}_{len(plot_data['html'])}"
                )
            elif isinstance(plot_data, str):
                # Fallback for old string format
                components.html(plot_data, height=500, scrolling=True)
        st.session_state.analysis_plots = []

# --- Static Content for Walkthrough ---
MODE_INFO = {
    "SQL Code": {
        "download_note": "**Tip**: After the query is generated, you can click the button below to execute it and download the results as a CSV.",
        "use_cases": [
            "Enter 'Show all columns from orders' to get a clean SELECT query.",
            "Enter 'Join orders with users on userId' to get a relational join query.",
            "Enter 'Total sales by month' to get a GROUP BY SQL query.",
            "Enter 'Filter users older than 25' to get a WHERE clause query.",
            "Enter 'Top 10 products by price' to get an ORDER BY LIMIT query.",
            "Enter 'List distinct cities from users' to get a SELECT DISTINCT query."
        ]
    },
    "Python Code": {
        "download_note": "**Tip**: After the code is generated, you can click the button below to run it and download the processed data as a CSV.",
        "use_cases": [
            "Enter 'Calculate rolling 7-day average of sales' to get pandas rolling mean code.",
            "Enter 'Handle missing values in price column' to get imputer/fillna code.",
            "Enter 'Create a new column profit = sales - cost' to get column assignment code.",
            "Enter 'Group by category and sum quantity' to get groupby aggregate code.",
            "Enter 'Normalize the age column' to get MinMaxScaler/StandardScaler code.",
            "Enter 'Sort data by date and reset index' to get sort_values and reset_index code."
        ]
    },
    "R Code": {
        "download_note": "**Tip**: After the R code is generated, you can click the button below to perform the operation and download the result as a CSV.",
        "use_cases": [
            "Enter 'Select columns date and sales' to get dplyr select code.",
            "Enter 'Filter for rows where sales > 100' to get filter() code.",
            "Enter 'Mutate a new column total=qty*price' to get mutate() code.",
            "Enter 'Summarize mean price by group' to get summarize() code.",
            "Enter 'Arrange by descending date' to get arrange(desc()) code.",
            "Enter 'Pivot data to wide format' to get pivot_wider() code."
        ]
    },
    "Ask Questions": {
        "use_cases": [
            "Enter 'What is the average price of items?' to get a numeric answer.",
            "Enter 'Summarize the key trends in my data' to get a high-level AI analysis.",
            "Enter 'Run a linear regression for Sales vs Price' to get statistical model results.",
            "Enter 'Is there a correlation between Age and Spend?' to get a correlation matrix.",
            "Enter 'Find the best classification model for target Churn' to get AutoML results.",
            "Enter 'Predict the next month's demand' to get a forecasting/regression output."
        ]
    },
    "Generate Graphs": {
        "use_cases": [
            "Enter 'Plot a bar chart of sales per region' to get an interactive bar graph.",
            "Enter 'Show me the distribution of customer ages' to get a histogram.",
            "Enter 'Generate a line graph for monthly revenue' to get a time-series plot.",
            "Enter 'Create a scatter plot for Budget vs ROI' to get a relationship chart.",
            "Enter 'Show a heat map of correlations' to get a visual matrix.",
            "Enter 'Plot a pie chart of market share' to get a compositional visualization."
        ]
    }
}

def main():
    st.set_page_config(page_title="Data Analysis Agent", page_icon=None, layout="wide")
    apply_apple_style()
    st.title("Data Analysis Agent")

    # --- Quick Start Guide (Always Visible) ---
    with st.expander("Quick Start Guide", expanded=True):
        st.markdown("""
        1. **STEP 1:** **Select your Mode** in the sidebar.
        2. **STEP 2:** **Add your Data** (CSV or XLSX) via the uploader in the sidebar.
        3. **STEP 3:** **Type your request** in the chat box below.
        4. **STEP 4:** **Hit Enter** to generate your result!
        """)

    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [] # Reset on load if needed or keep history
    if "analysis_plots" not in st.session_state:
        st.session_state.analysis_plots = []
    if "data_summary" not in st.session_state:
        st.session_state.data_summary = None
    if "deep_insights" not in st.session_state:
        st.session_state.deep_insights = None
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("API Configuration")
        # Allow user to input keys securely
        openrouter_key = st.text_input("OpenRouter API Key", type="password", help="Used for Gemini and Llama models via OpenRouter")
        nvidia_key = st.text_input("NVIDIA API Key (Optional)", type="password", help="Used for NVIDIA NIM endpoints if configured")
        
        st.session_state.openrouter_key = openrouter_key
        st.session_state.nvidia_key = nvidia_key
        
        # Primary key logic for tools
        api_key = openrouter_key if openrouter_key else nvidia_key
        st.session_state.api_key = api_key

        if not api_key:
            st.warning("Please enter at least one API key to proceed.")

        st.header("Data Source")
        uploaded_files = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"], accept_multiple_files=True)
        
        if uploaded_files:
            if "dfs" not in st.session_state: st.session_state.dfs = {}
            new_files = False
            for f in uploaded_files:
                safe_name = "df_" + os.path.splitext(f.name)[0].replace(" ", "_").lower()
                if safe_name not in st.session_state.dfs:
                    try:
                        if f.name.endswith('.csv'):
                            df = pd.read_csv(f)
                        else:
                            df = pd.read_excel(f)
                        st.session_state.dfs[safe_name] = df
                        new_files = True
                    except Exception as e:
                        st.error(f"Error loading {f.name}: {e}")
            if new_files:
                st.success(f"Loaded {len(st.session_state.dfs)} datasets.")
                # AUTO-AI INSIGHTS IN CHAT
                with st.spinner("Analyzing data..."):
                    summaries = generate_combined_insights(st.session_state.dfs, api_key)
                    
                    # Store onboarding report as a message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "### Data Onboarding Report\n\nYour data is loaded and analyzed below.",
                        "is_onboarding": True,
                        "summaries": summaries
                    })
        else:
            st.session_state.dfs = {}

        st.divider()

        st.header("Mode Selection")
        # 5 Options as requested
        agent_mode = st.radio(
            "Select Agent Mode:",
            ["SQL Code", "Python Code", "R Code", "Ask Questions", "Generate Graphs"]
        )
        
        st.divider()
        
        # Stop & Reset Button
        if st.button("Stop & Reset", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- Chat Interface ---
    # Clear history if mode changes? Optional. For now let's keep a shared history or clear it.
    # To keep it simple, we just show messages.
    if "last_mode" not in st.session_state or st.session_state.last_mode != agent_mode:
        st.session_state.last_mode = agent_mode
        mode_data = MODE_INFO[agent_mode]
        intro_msg = f"Switched to **{agent_mode}** mode.\n\n"
        if "download_note" in mode_data:
             intro_msg += f"{mode_data['download_note']}\n\n"
        intro_msg += f"**Example Use Cases:**\n- " + "\n- ".join(mode_data['use_cases'])
        st.session_state.messages.append({"role": "assistant", "content": intro_msg})

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # 1. Onboarding Snapshots
            if msg["role"] == "assistant" and msg.get("is_onboarding"):
                summaries = msg.get("summaries", {})
                for name, df in st.session_state.dfs.items():
                    st.write(f"#### Snapshot: `{name}`")
                    st.dataframe(df.head(5), use_container_width=True)
                    summary = summaries.get(name) or summaries.get(name.replace("df_", "")) or next(iter(summaries.values())) if summaries else "Summary unavailable."
                    st.write(f"**AI Insights for {name}:**")
                    st.markdown(summary)
                    st.divider()
            # 1. Check for Custom Styled Message (Light Red Instruction)
            if msg.get("custom_styled_msg"):
                st.markdown(f'<div style="color: #ff4b4b; background-color: #ffeaea; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-weight: 500; border: 1px solid #ffcaca;">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

            # 2. Results CSV Download
            if "result_csv_data" in msg:
                st.download_button(
                    label="Download Results (CSV)",
                    data=msg["result_csv_data"],
                    file_name="analysis_result.csv",
                    mime="text/csv",
                    type="primary",
                    key=f"dl_msg_{i}",
                    use_container_width=True
                )

            # 3. Interactive Graphs
            if "plot_html" in msg:
                components.html(msg["plot_html"], height=500, scrolling=True)
                # Rename button to DOWNLOAD GRAPH for dash history
                btn_label = "DOWNLOAD GRAPH" if msg.get("custom_styled_msg") else "Download Chart (HTML)"
                st.download_button(
                    label=btn_label,
                    data=msg["plot_html"],
                    file_name="dashboard.html" if msg.get("custom_styled_msg") else "visualization.html",
                    mime="text/html",
                    type="primary" if msg.get("custom_styled_msg") else "secondary",
                    key=f"dl_plot_{i}_{len(msg['plot_html'])}",
                    use_container_width=True
                )
            
    # Render Plots (Top level / History)
    if st.session_state.analysis_plots:
        with st.chat_message("assistant"):
            render_analysis_plots()

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
                        msg_already_appended = False
                        # Init LLM
                        if api_key.startswith("sk-or-"):
                            from langchain_openai import ChatOpenAI
                            llm = ChatOpenAI(model="nvidia/nemotron-3-nano-30b-a3b:free", api_key=api_key, base_url="https://openrouter.ai/api/v1")
                        else:
                            from langchain_nvidia_ai_endpoints import ChatNVIDIA
                            llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key)
                        
                        # INTERACTIONS
                        if agent_mode == "Ask Questions":
                            # INTERACTIVE MODE - Calculations & Modeling Focus
                            tools = [run_pandas_code, run_linear_regression, run_manova, run_automl]
                            schema_context = get_schema_context(st.session_state.dfs)
                            
                            template = '''You are a Senior Data Scientist & Analyst. Answer the user's question using the provided tools.
                            
                            SCHEMA CONTEXT (Tables & Columns):
                            {schema_context}
                            
                            AVAILABLE TOOLS:
                            {tools}
                            
                            INSTRUCTIONS:
                            1. If the user asks for "predictions", "modeling", "machine learning", or "MLP/Neural Network", use `run_automl`.
                            2. For specific statistical summaries of linear relationships, use `run_linear_regression`.
                            3. For group-based multivariate analysis, use `run_manova`.
                            4. For general questions like "what is the average...", use `run_pandas_code`.
                            5. If you can answer based ONLY on the SCHEMA CONTEXT (e.g. "What columns are in table X?"), do so directly.
                            6. **TABULAR OUTPUT**: If the user asks for a list, top items, averages by group, or any structured data, you MUST judge if a table is appropriate. If so, format your `Final Answer` using a clear **Markdown Table**.
                            
                            ALWAYS use this format:
                            Thought: you should always think about what to do
                            Action: the action to take, should be one of [{tool_names}]
                            Action Input: the input to the action
                            Observation: the result of the action
                            ... (this Thought/Action/Action Input/Observation can repeat N times)
                            Thought: I now know the final answer
                            Final Answer: the final answer to the original input question
                            
                            Question: {input}
                            Thought:{agent_scratchpad}
                            '''
                            df_names = list(st.session_state.dfs.keys())
                            
                            prompt_template = PromptTemplate.from_template(template).partial(df_names=str(df_names), schema_context=schema_context)
                            agent = create_react_agent(llm, tools, prompt_template)
                            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=15)
                            response = agent_executor.invoke({"input": prompt})
                            output = response['output']
                            
                            
                                # Clean up handled by overwrite next time or explicit delete if desired
                            
                        elif agent_mode == "Generate Graphs":
                            # MINIMALIST DASHBOARD WORKFLOW (Direct to HTML with Client-Side Support)
                            context_str = ""
                            for name, d in st.session_state.dfs.items():
                                buffer = io.StringIO()
                                d.info(buf=buffer)
                                df_info = buffer.getvalue()
                                # Use high-density samples for initial aggregation
                                head_csv = d.head(100).to_csv(index=False)
                                context_str += f"\n--- TABLE: {name} (Shape: {d.shape}) ---\n"
                                context_str += f"SCHEMA & INFO:\n{df_info}\n"
                                context_str += f"SAMPLE DATA (CSV):\n{head_csv}\n"

                            progress_bar = st.progress(0, text="Generating graph")
                            import time
                            for percent_complete in range(100):
                                time.sleep(0.01)
                                progress_bar.progress(percent_complete + 1, text="Generating graph")
                                
                            # Force the user-requested model and specific API key
                            graph_api_key = st.session_state.get("openrouter_key", "")
                            if not graph_api_key:
                                st.error("OpenRouter API Key missing! Please enter it in the sidebar.")
                                st.stop()

                            if graph_api_key.startswith("sk-or-"):
                                from langchain_openai import ChatOpenAI
                                llm_graph = ChatOpenAI(model="google/gemini-3-flash-preview", api_key=graph_api_key, base_url="https://openrouter.ai/api/v1", temperature=0.3)
                            else:
                                from langchain_openai import ChatOpenAI
                                llm_graph = ChatOpenAI(model="google/gemini-3-flash-preview", api_key=graph_api_key)

                            system_prompt = """You are a DataArchitect and Principal Architect.
                            
                            OBJECTIVE: Transform raw dataset into a premium, minimalist, single-file HTML dashboard.
                            
                            STRICT UI/UX REQUIREMENTS:
                            - NO HEADINGS: Do NOT include any branding headings like 'Health Analytics', 'v1.0', or 'Principal Architect' in the HTML.
                            - CLEANLINESS: REMOVE all subheadings and section headers. Keep the layout extremely minimalist and clutter-free.
                            - Sidebar: Sleek Sidebar for data management.
                            - Main Area: Large Main Area for visualization.
                            - Typography: High-legibility (Inter or Plus Jakarta Sans).
                            - COMPONENTS: 
                                - Exactly ONE main graph (Bar, Line, etc.).
                                - Exactly TWO clean summary data cards.
                                - A clear Upload/Switch Data function supporting CSV and XLSX.
                            - STYLING: Subtle gradients, rounded-2xl corners, glassmorphism.
                            
                            MANDATORY TECHNICAL CONSTRAINTS:
                            1. Output exactly ONE standalone .html file.
                            2. Include these libraries via CDN in <head>: Tailwind CSS, Chart.js, PapaParse, SheetJS.
                            3. HYBRID LOADING: Pre-aggregate the data into `const initialData`.
                            4. BROWSER-ONLY PROCESSING: All parsing for NEW uploads happens client-side.
                            
                            Return ONLY the raw HTML. No explanations."""

                            user_prompt = f"""
                            User Question: "{prompt}"
                            
                            DATA CONTEXT:
                            {context_str}
                            
                            Generate the ultra-clean premium dashboard (NO HEADINGS) now.
                            """
                            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                            response = llm_graph.invoke(messages)
                            html_plot = response.content.strip()

                            # Cleanup
                            import re
                            html_match = re.search(r'```html\n(.*?)```', html_plot, re.DOTALL) or re.search(r'```(.*?)```', html_plot, re.DOTALL)
                            if html_match:
                                html_plot = html_match.group(1).strip()
                            
                            progress_bar.empty()
                            
                            # 1. Light Red Instruction at the Top
                            output_msg = "Please upload the data set below to get the accurate results"
                            st.markdown(f'<div style="color: #ff4b4b; background-color: #ffeaea; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-weight: 500; border: 1px solid #ffcaca;">{output_msg}</div>', unsafe_allow_html=True)
                            
                            # 2. Graph Rendering
                            components.html(html_plot, height=600, scrolling=True)
                            
                            # 3. DOWNLOAD GRAPH Button below the graph
                            st.download_button(
                                label="DOWNLOAD GRAPH",
                                data=html_plot,
                                file_name="dashboard.html",
                                mime="text/html",
                                type="primary",
                                key=f"dl_dash_{len(html_plot)}",
                                use_container_width=True
                            )
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output_msg, 
                                "plot_html": html_plot,
                                "custom_styled_msg": True # Flag for history rendering
                            })
                            msg_already_appended = True
                            
                        else:
                            # CODE GENERATION MODE (SQL, Python, R)
                            schema_context = get_schema_context(st.session_state.dfs)
                            
                            if agent_mode == "SQL Code":
                                system_prompt = f"""You are a Senior SQL Developer.
                                Task: Generate a valid SQL query to answer the user request.
                                
                                Schema:
                                {schema_context}
                                
                                IMPORTANT:
                                1. Use the EXACT table names provided in the schema.
                                2. Return ONLY the SQL query code block. Include a brief explanation.
                                """
                                response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
                                output = response.content
                                
                            elif agent_mode == "Python Code":
                                schema_context = get_schema_context(st.session_state.dfs)
                                lang = "Python"
                                instruction = "Generate valid Python pandas code assuming dataframes are loaded as named."
                                system_prompt = f"""You are a generic Data Coding Assistant.
                                Task: {instruction}
                                Language: {lang}
                                
                                Schema:
                                {schema_context}
                                
                                Return ONLY the code block and a brief explanation.
                                """
                                response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
                                output = response.content
                            elif agent_mode == "R Code":
                                schema_context = get_schema_context(st.session_state.dfs)
                                lang = "R"
                                instruction = "Generate valid R code (tidyverse) assuming dataframes are loaded."
                                system_prompt = f"""You are a generic Data Coding Assistant.
                                Task: {instruction}
                                Language: {lang}
                                
                                Schema:
                                {schema_context}
                                
                                Return ONLY the code block and a brief explanation.
                                """
                                response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
                                output = response.content

                        if not msg_already_appended:
                            st.markdown(output)
                            st.session_state.messages.append({"role": "assistant", "content": output})
                        
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- Post-Chat Actions ---
    # Show results download first if it exists from immediate execution (e.g. SQL)
    if os.path.exists("analysis_output.csv") and agent_mode not in ["Ask Questions", "Generate Graphs", "SQL Code", "Python Code", "R Code"]:
        st.divider()
        with open("analysis_output.csv", "rb") as f:
            st.download_button(
                label="Download Last SQL Result (CSV)",
                data=f,
                file_name="analysis_result.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

    # Show "Perform" button for SQL/Python/R modes that need secondary execution
    if agent_mode in ["SQL Code", "Python Code", "R Code"] and "last_prompt" in st.session_state:
        st.divider()
        if st.button("Perform Operation and Generate CSV", type="secondary", use_container_width=True):
            with st.spinner("Executing and Generating Results..."):
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
                    
                    if agent_mode == "SQL Code":
                        role_description = "You are a Senior SQL Developer."
                        task_instruction = "Generate a valid SQL query to answer the user request."
                        custom_requirements = """1. Use the EXACT table names provided in the schema.
2. Return ONLY the SQL query code block."""
                    else:
                        role_description = "You are a Senior Python Data Scientist & Machine Learning Expert."
                        task_instruction = "Write robust Python pandas/sklearn code to answer the user request and SAVE the result to a CSV file."
                        custom_requirements = """1. **SMART AUTOML**: If the user asks for 'predictions', 'modeling', or 'best model', compare at least TWO models (e.g., RandomForest and XGBoost) and select the better one.
2. **NEURAL NETWORKS**: Use `MLPClassifier` or `MLPRegressor` if deep learning is requested.
3. **COLUMN VERIFICATION**: ALWAYS print(df.columns) at the start and ensure the requested columns exist before indexing."""

                    system_prompt = f"""{role_description}
                    Task: {task_instruction}
                    
                    Dataframes (Loaded in 'dfs' dict):
                    - Keys: {list(st.session_state.dfs.keys())}
                    - Primary Shortcut: `df` (points to the first dataframe)
                    - Schema: 
                    {schema_info}
                    
                    CRITICAL REQUIREMENTS:
                    {custom_requirements}
                    """
                    
                    if agent_mode != "SQL Code":
                        system_prompt += """
                    4. Use `dfs['key']` or the `df` shortcut to access dataframes.
                    5. **AUTO-CLEANING**: Use the provided `auto_preprocess(df, target_col=None)` function to prepare your data. It handles NaNs, IDs, and Categorical encoding (OHE) automatically.
                    6. **STRICT PRE-TRAIN RULE**: You MUST call `X = auto_preprocess(X)` or `X, y = auto_preprocess(df, target_col='target')` before fitting any model. This guarantees no string-to-float errors.
                    7. **RESULTS**: Use `print()` to display results. 
                    8. **EXECUTIVE SUMMARY**: Format all your `print()` outputs as a professional, structured **Executive Summary** using Markdown. 
                       - Use headers (###), bold text, and Markdown tables for technical metrics.
                       - **LAYMAN EXPLANATIONS**: For every technical metric (e.g., Accuracy, R2, ROC AUC), you MUST include a one-sentence layman explanation (e.g., "This means the model correctly guesses 85% of the cases").
                       - **PLAIN LANGUAGE INSIGHTS**: Conclude with a section titled "### Interpretation & Key Insights" containing 3-4 simple bullet points that explain what the results mean for a non-technical person (the "so what").
                    9. **OUTPUT**: Save result to 'analysis_output.csv' using `to_csv(index=False)`.
                    10. Return ONLY valid Python code. No markdown.
                    """
                    else:
                        system_prompt += "\n3. Return ONLY the SQL query code block. No explanation."
                    
                    from langchain_core.messages import HumanMessage, SystemMessage
                    response = llm_code.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt_text)])
                    code = response.content.strip()
                    
                    # Clean Code
                    if agent_mode == "SQL Code":
                        code = code.replace("```sql", "").replace("```", "").strip()
                    else:
                        code_match = re.search(r"```python\n?(.*?)\n?```", code, re.DOTALL)
                        if code_match:
                            code = code_match.group(1).strip()
                        else:
                            code = code.replace("```python", "").replace("```", "").strip()
                    
                    if "Final Answer:" in code:
                        code = code.split("Final Answer:")[0].strip()
                    
                    # Validating Code Safety (For Python/R)
                    if agent_mode != "SQL Code" and ("os.system" in code or "sys.modules" in code):
                        st.error("Unsafe code detected. Operation aborted.")
                    else:
                        if agent_mode == "SQL Code":
                            try:
                                result_df = execute_sql_query(code, st.session_state.dfs)
                                result_df.to_csv("analysis_output.csv", index=False)
                                
                                result_text = "### SQL Query Result Preview\n"
                                if not result_df.empty:
                                    result_text += result_df.head(10).to_markdown(index=False)
                                else:
                                    result_text += "*Query executed successfully but returned zero rows.*"
                                st.success("Execution Successful!")
                            except Exception as sql_e:
                                st.error(f"SQL Error: {sql_e}")
                                with st.expander("Show Generated Query"):
                                    st.code(code, language='sql')
                                result_text = ""
                        else:
                            # Prepare execution context with all DS libraries pre-loaded
                            old_stdout = sys.stdout
                            new_stdout = io.StringIO()
                            sys.stdout = new_stdout
                            
                            try:
                                # Context with all modeling tools available
                                first_df = next(iter(st.session_state.dfs.values()))
                                local_vars = {
                                    "dfs": st.session_state.dfs, 
                                    "df": first_df,
                                    "pd": pd, 
                                    "np": np, 
                                    "sm": sm,
                                    "sqlite3": __import__("sqlite3"),
                                    "auto_preprocess": auto_preprocess,
                                    "train_test_split": train_test_split,
                                    "r2_score": r2_score,
                                    "accuracy_score": accuracy_score,
                                    "StandardScaler": StandardScaler,
                                    "OneHotEncoder": OneHotEncoder,
                                    "RandomForestClassifier": RandomForestClassifier,
                                    "RandomForestRegressor": RandomForestRegressor,
                                    "MLPClassifier": MLPClassifier,
                                    "MLPRegressor": MLPRegressor,
                                    "xgb": xgb,
                                    "lgb": lgb
                                }
                                exec(code, local_vars)
                                result_text = new_stdout.getvalue()
                                st.success("Execution Successful!")
                            except Exception as exec_e:
                                st.error(f"Execution Error: {exec_e}")
                                with st.expander("Show Generated Code"):
                                    st.code(code, language='python')
                                result_text = new_stdout.getvalue()
                            finally:
                                sys.stdout = old_stdout
                        
                        # Display Text Results
                        if result_text.strip():
                            st.subheader("Executive Summary")
                            st.markdown(result_text)
                            
                            # Immediate Download Button after summary
                            csv_data = None
                            if os.path.exists("analysis_output.csv"):
                                with open("analysis_output.csv", "rb") as f:
                                    csv_data = f.read()
                                    st.download_button(
                                        label="Download Results (CSV)",
                                        data=csv_data,
                                        file_name="analysis_result.csv",
                                        mime="text/csv",
                                        type="primary",
                                        use_container_width=True
                                    )
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": result_text,
                                "result_csv_data": csv_data
                            })
                except Exception as e:
                    st.error(f"Generation Error: {e}")

if __name__ == "__main__":
    main()