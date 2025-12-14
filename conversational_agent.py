import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import io
import os

# --- Configuration & Setup ---
st.set_page_config(page_title="Conversational Data Agent", page_icon="💬", layout="wide")

# Initialize Session State for Data
if "df" not in st.session_state:
    st.session_state.df = None

# --- TOOLS Definition ---
# Tools must be stateless or access global state (st.session_state.df) carefully.
# In a real app, passing df as an argument or using a class-based tool is better.
# For Streamlit, we'll access st.session_state.df directly in tools for simplicity of the "Agent" reasoning.

@tool
def get_dataset_info() -> str:
    """Returns basic information about the loaded dataset (columns, dtypes, head)."""
    if st.session_state.df is None:
        return "No data loaded."
    
    df = st.session_state.df
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head().to_markdown()
    return f"Dataset Info:\n{info_str}\n\nFirst 5 rows:\n{head_str}"

@tool
def get_summary_statistics() -> str:
    """Calculates descriptive statistics for numeric columns."""
    if st.session_state.df is None:
        return "No data loaded."
    return st.session_state.df.describe().to_markdown()

@tool
def run_linear_regression(target_col: str, feature_cols: list[str]) -> str:
    """Runs a Linear Regression model. Args: target_col (string), feature_cols (list of strings)."""
    df = st.session_state.df
    if df is None: return "No data."
    
    try:
        # Simple cleanup
        model_df = df[[target_col] + feature_cols].dropna()
        X = model_df[feature_cols]
        y = model_df[target_col]
        
        # Using statsmodels for detailed report
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return str(model.summary())
    except Exception as e:
        return f"Regression failed: {e}"

@tool
def run_svm_regression(target_col: str, feature_col: str) -> str:
    """Runs Support Vector Regression (SVR) for a single feature. Args: target_col, feature_col."""
    df = st.session_state.df
    if df is None: return "No data."
    
    try:
        model_df = df[[target_col, feature_col]].dropna()
        X = model_df[[feature_col]]
        y = model_df[target_col]
        
        svr = SVR()
        svr.fit(X, y)
        y_pred = svr.predict(X)
        r2 = r2_score(y, y_pred)
        
        return f"SVR Trained. R2 Score: {r2:.4f}"
    except Exception as e:
        return f"SVM failed: {e}"

@tool
def run_manova(dependent_vars: list[str], independent_var: str) -> str:
    """Runs MANOVA. Args: dependent_vars (list of cols), independent_var (categorical group col)."""
    df = st.session_state.df
    if df is None: return "No data."
    
    try:
        # MANOVA requires formula interface in statsmodels usually: 'dep1 + dep2 ~ indep'
        deps_str = " + ".join(dependent_vars)
        formula = f"{deps_str} ~ {independent_var}"
        
        manova = MANOVA.from_formula(formula, data=df)
        return str(manova.mv_test())
    except Exception as e:
        return f"MANOVA failed: {e}"

# UI Helper for Plots (Agents struggle to return plots directly, so we use a specialized tool or handle it in UI)
# However, we can create a tool that "generates code" or "plots to a buffer".
# For this conversational agent, let's keep it simple: The agent effectively "describes" the result, 
# but for plots, we might need a dedicated "Plotting Agent" or simple regex trigger. 
# Let's try to let the Agent return a "Action: Plot" string if we want, or just stick to text analysis first as requested.
# But user asked for visualization.
@tool
def generate_plot(plot_type: str, x_col: str, y_col: str = None, hue: str = None) -> str:
    """Generates a plot. plot_type can be 'scatter', 'bar', 'box', 'histogram', 'heatmap'. x_col and y_col are column names."""
    df = st.session_state.df
    if df is None: return "No data."
    
    try:
        fig, ax = plt.subplots()
        if plot_type == 'scatter':
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif plot_type == 'bar':
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif plot_type == 'box':
            sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif plot_type == 'histogram':
            sns.histplot(data=df, x=x_col, kde=True, hue=hue, ax=ax)
        elif plot_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        
        # Save to a session state list to display later, because tools run in backend
        if "plots" not in st.session_state:
            st.session_state.plots = []
        st.session_state.plots.append(fig)
        
        return f"Plot ({plot_type}) generated and saved to display queue."
    except Exception as e:
        return f"Plotting failed: {e}"


# --- Main App ---
def main():
    st.title("💬 Conversational Data Science Agent")
    st.markdown("Powered by **Vertex AI** & **LangChain**")
    
    # sidebar
    with st.sidebar:
        st.header("1. Configuration")
        # Allow user to provide key if env var not set
        api_key = st.text_input("GCP/Vertex Key (if needed)", type="password", help="If using Application Default Credentials, leave blank.")
        project_id = st.text_input("GCP Project ID", value="august-emitter-480918-u2")
        
        st.header("2. Data Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success(f"Loaded {len(st.session_state.df)} rows.")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your Data Science Agent. Upload a dataset and ask me anything about it. I can run regressions, MANOVAs, or generate plots."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Display any plots generated during the conversation turn
    # This is a bit tricky in Streamlit chat flow, usually we display them immediately.
    # We will check if 'plots' exists and clear it after showing.
    if "plots" in st.session_state and st.session_state.plots:
        with st.chat_message("assistant"):
            for fig in st.session_state.plots:
                st.pyplot(fig)
        st.session_state.plots = [] # Clear queue

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AGENT EXECUTION
        if st.session_state.df is not None:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Setup Agent
                        # We use ChatVertexAI model
                        llm = ChatVertexAI(
                            model="gemini-1.5-flash-001",
                            temperature=0,
                            project=project_id,
                            # safety settings if needed
                        )
                        
                        tools = [get_dataset_info, get_summary_statistics, run_linear_regression, run_svm_regression, run_manova, generate_plot]
                        
                        # Create Agent
                        # Using tool_calling_agent which is optimized for function calling models like Gemini
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", "You are a helpful Data Science Assistant. You have access to a pandas dataframe. Use the provided tools to answer the user's question. If you generate a plot, mention it in the text."),
                            ("placeholder", "{chat_history}"),
                            ("human", "{input}"),
                            ("placeholder", "{agent_scratchpad}"),
                        ])
                        
                        agent = create_tool_calling_agent(llm, tools, prompt_template)
                        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                        
                        response = agent_executor.invoke({"input": prompt, "chat_history": []}) # We could manage history properly
                        
                        output_text = response['output']
                        st.markdown(output_text)
                        st.session_state.messages.append({"role": "assistant", "content": output_text})

                        # Check for plots again after execution
                        if "plots" in st.session_state and st.session_state.plots:
                            for fig in st.session_state.plots:
                                st.pyplot(fig)
                            st.session_state.plots = []
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
             with st.chat_message("assistant"):
                 st.warning("Please upload a dataset first.")


if __name__ == "__main__":
    main()
