import streamlit as st
import pandas as pd
import os
import io
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

def get_schema_string(df, table_name):
    """Generates a pseudo-SQL schema definition from a Pandas DataFrame."""
    schema_parts = []
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "INT"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "FLOAT"
        elif pd.api.types.is_bool_dtype(dtype):
            sql_type = "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "DATETIME"
        else:
            sql_type = "TEXT"
        schema_parts.append(f"{col} {sql_type}")
    
    columns_def = ", ".join(schema_parts)
    return f"CREATE TABLE {table_name} ({columns_def});"

def main():
    st.set_page_config(page_title="Multi-Language Data Agent", page_icon="�", layout="wide")
    st.title("� Multi-Language Data Agent")
    st.markdown("Upload multiple CSVs (tables) and ask questions. I will generate **SQL, Python, or R Code** to answer them.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        # Default key
        default_key = "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"
        api_key = st.text_input("API Key (NVIDIA or OpenRouter)", value=default_key, type="password")
        
        st.header("Database Tables")
        uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
        
        st.info("💡 Filenames will be used as Table Names (e.g., `users.csv` -> table `users`)")

    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload your tables and ask me a question! I can join tables and write complex SQL for you."}]

    # --- Schema Extraction ---
    schemas = []
    # Dictionary to store dfs if we want to preview them later (optional)
    # data_frames = {} 
    
    if uploaded_files:
        st.subheader("Active Database Schema")
        schema_text_display = ""
        
        for uploaded_file in uploaded_files:
            # Table name from filename
            table_name = os.path.splitext(uploaded_file.name)[0]
            # Sanitize table name (keep simple)
            table_name = "".join(c for c in table_name if c.isalnum() or c == '_')
            
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                # Generate Schema
                schema_def = get_schema_string(df, table_name)
                schemas.append(schema_def)
                
                with st.expander(f"Table: {table_name}"):
                    st.code(schema_def, language="sql")
                    st.dataframe(df.head(3))
                    
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
                
        full_schema_context = "\n".join(schemas)
    else:
        full_schema_context = ""

    # --- Chat Interface ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Language Selection (placed here or sidebar, sidebar is cleaner but here is more context-aware if per-query)
    # Let's put it in the sidebar for global setting or near input. Sidebar is better for "Session Mode".
    
    with st.sidebar:
        st.header("Output Settings")
        output_language = st.radio("Generate Code In:", ["SQL", "Python (Pandas)", "R"])

    if prompt := st.chat_input("Ask a question about your data..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        if not uploaded_files:
            with st.chat_message("assistant"):
                st.warning("Please upload at least one CSV file first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner(f"Writing {output_language} Code..."):
                    try:
                        # Init LLM based on Key Type
                        if api_key.startswith("sk-or-"):
                            from langchain_openai import ChatOpenAI
                            # OpenRouter uses OpenAI-compatible API
                            llm = ChatOpenAI(
                                model="meta-llama/llama-3.1-70b-instruct", 
                                api_key=api_key,
                                base_url="https://openrouter.ai/api/v1"
                            )
                            # st.toast("Using OpenRouter (Llama 3.1 70B)")
                        else:
                            # Default to NVIDIA NIM
                            llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key)
                        
                        # Dynamic System Prompt based on Language
                        if output_language == "SQL":
                            lang_instruction = "3. **Construct** the correct SQL query. Join tables where necessary."
                            lang_format = "SQL query in a ```sql block"
                        elif output_language == "Python (Pandas)":
                            lang_instruction = "3. **Construct** the correct Python/Pandas code. Assume DataFrames are loaded with names matching the table names (e.g. `df_users`)."
                            lang_format = "Python code in a ```python block"
                        else: # R
                            lang_instruction = "3. **Construct** the correct R code (using dplyr/tidyverse). Assume dataframes are loaded with names matching the table names."
                            lang_format = "R code in a ```r block"

                        system_prompt = (
                            f"You are an expert Data Analyst & Code Generator. Your task is to generate valid {output_language} code based on the provided schema.\n"
                            "1. **Analyze** the user's natural language request.\n"
                            "2. **Analyze** the provided database/dataframe schema.\n"
                            f"{lang_instruction}\n"
                            "4. **Output Format**:\n"
                            f"   - First, provide the {lang_format}.\n"
                            "   - Second, provide a brief, clear explanation of how the code works.\n"
                            f"   - Do NOT assume columns that don't exist. Use ONLY the provided schema.\n\n"
                            f"**Data Schema (Table/DataFrame Name -> Columns):**\n{full_schema_context}"
                        )
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        
                        response = llm.invoke(messages)
                        content = response.content
                        
                        st.markdown(content)
                        st.session_state.messages.append({"role": "assistant", "content": content})
                        
                    except Exception as e:
                        st.error(f"Error generating code: {e}")

if __name__ == "__main__":
    main()
