import streamlit as st
import pandas as pd
import json
import io
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def get_column_profile(df):
    """
    Generates a metadata dictionary for the dataframe.
    """
    profile = {}
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        null_count = int(col_data.isnull().sum())
        total_count = len(col_data)
        null_pct = round((null_count / total_count) * 100, 2)
        unique_count = col_data.nunique()
        
        # Get samples (unique non-null values)
        samples = col_data.dropna().unique()[:5].tolist()
        
        # Min/Max for numeric/datetime
        min_val = None
        max_val = None
        if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_datetime64_any_dtype(col_data):
            try:
                min_val = str(col_data.min())
                max_val = str(col_data.max())
            except:
                pass

        profile[col] = {
            "dtype": dtype,
            "null_percentage": f"{null_pct}%",
            "unique_values": unique_count,
            "min_description": min_val,
            "max_description": max_val,
            "sample_values": [str(x) for x in samples]
        }
    return profile

def apply_cleaning_plan(df, plan):
    """
    Applies the cleaning actions from the JSON plan to the DataFrame.
    """
    df_clean = df.copy()
    
    try:
        if "columns" in plan:
            for col_info in plan["columns"]:
                col = col_info.get("column_name")
                if col not in df_clean.columns:
                    continue
                
                actions = col_info.get("recommended_actions", [])
                for action in actions:
                    action_type = action.get("action_type")
                    params = action.get("parameters", {})
                    
                    # 1. Missing Value Handling
                    if action_type == "missing_value_handling":
                        strategy = params.get("strategy", "drop")
                        if strategy == "drop_rows":
                            df_clean = df_clean.dropna(subset=[col])
                        elif strategy == "impute_mean":
                            if pd.api.types.is_numeric_dtype(df_clean[col]):
                                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                        elif strategy == "impute_median":
                            if pd.api.types.is_numeric_dtype(df_clean[col]):
                                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        elif strategy == "impute_mode":
                            mode_val = df_clean[col].mode()
                            if not mode_val.empty:
                                df_clean[col] = df_clean[col].fillna(mode_val[0])
                        elif strategy == "fill_value":
                            fill_val = params.get("value", 0)
                            df_clean[col] = df_clean[col].fillna(fill_val)

                    # 2. Type Casting
                    elif action_type == "type_cast":
                        target_type = params.get("target_type", "")
                        if "int" in target_type or "float" in target_type or "numeric" in target_type:
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                        elif "datetime" in target_type:
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

                    # 3. Duplicate Handling (usually dataset level, but can be triggered here)
                    elif action_type == "duplicate_handling":
                         df_clean = df_clean.drop_duplicates(subset=[col]) # Dedupe based on single col if specified
    
    except Exception as e:
        st.error(f"Error applying plan: {e}")
        
    return df_clean

def main():
    st.set_page_config(page_title="Data Cleaning Agent", page_icon="🧹", layout="wide")
    st.title("🧹 Data Cleaning Agent")
    st.markdown("Upload a dataset. I will **plan** and **execute** cleaning for you. You can also **chat** with your data below.")

    # --- Session State for Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! detailed profiling is ready. Ask me anything about your data quality or specific columns."}]

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        default_key = "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"
        api_key = st.text_input("OpenRouter API Key", value=default_key, type="password")
        
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    # --- Main Logic ---
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded `{uploaded_file.name}` with {len(df)} rows and {len(df.columns)} columns.")
            
            # Profiling
            with st.spinner("Profiling Data Metadata..."):
                profile_data = get_column_profile(df)
                
            with st.expander("View Extracted Metadata (Input to Agent)", expanded=False):
                st.json(profile_data)
            
            if st.button("Clean Data"):
                if not api_key:
                    st.error("Please provide an OpenRouter API Key.")
                    return

                with st.spinner("Analyzing & Cleaning..."):
                    try:
                        llm = ChatOpenAI(
                            model="meta-llama/llama-3.1-70b-instruct", 
                            api_key=api_key,
                            base_url="https://openrouter.ai/api/v1",
                            temperature=0.1 # Low temp for deterministic output
                        )

                        system_prompt = """You are a Data Cleaning Decision Agent.

Your role is to ANALYZE dataset profiling metadata and DECIDE the appropriate data cleaning actions.

INPUT YOU RECEIVE:
1. Dataset metadata (column names, data types, null %, min/max, samples)

YOUR RESPONSIBILITIES:
- Detect data quality issues
- Recommend cleaning actions (deterministic rules)
- OUTPUT JSON with 'parameters' that describe EXACTLY how to fix it.

ALLOWED CLEANING ACTIONS & PARAMETERS:
1. missing_value_handling
   - parameters: {"strategy": "drop_rows" | "impute_mean" | "impute_mode" | "fill_value", "value": "..."}
2. type_cast
   - parameters: {"target_type": "numeric" | "datetime" | "string"}
3. duplicate_handling
   - parameters: {"subset": ["col_name"]}
4. no_action

OUTPUT FORMAT (JSON ONLY, NO MARKDOWN):
{
  "dataset": "<dataset_name>",
  "columns": [
    {
      "column_name": "<string>",
      "detected_issues": ["<issue_type>"],
      "recommended_actions": [
        {
          "action_type": "<allowed_action>",
          "parameters": { <see above> },
          "confidence": <float 0–1>
        }
      ]
    }
  ]
}
"""
                        
                        user_message = f"Dataset Name: {uploaded_file.name}\nMetadata:\n{json.dumps(profile_data, indent=2)}"
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ]
                        
                        response = llm.invoke(messages)
                        content = response.content.strip()
                        
                        # Cleanup
                        if content.startswith("```json"): content = content[7:]
                        if content.endswith("```"): content = content[:-3]
                            
                        cleaning_plan = json.loads(content)
                        
                        st.subheader("1. Proposed Cleaning Plan")
                        st.json(cleaning_plan)
                        
                        # Apply Cleaning
                        df_cleaned = apply_cleaning_plan(df, cleaning_plan)
                        
                        st.subheader("2. Cleaned Data Preview")
                        st.dataframe(df_cleaned.head())
                        st.caption(f"Original Shape: {df.shape} -> Cleaned Shape: {df_cleaned.shape}")
                        
                        # Download Buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Download Cleaned Data (CSV)",
                                data=df_cleaned.to_csv(index=False),
                                file_name=f"cleaned_{uploaded_file.name}",
                                mime="text/csv"
                            )
                        with col2:
                            st.download_button(
                                label="Download Cleaning Plan (JSON)",
                                data=json.dumps(cleaning_plan, indent=2),
                                file_name=f"cleaning_plan_{uploaded_file.name}.json",
                                mime="application/json"
                            )
                        
                    except json.JSONDecodeError:
                        st.error("Failed to parse JSON response. Raw output:")
                        st.text(content)
                    except Exception as e:
                        st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # --- Chat Interface ---
    st.divider()
    st.subheader("💬 Q&A with Data Cleaning Expert")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about data quality, specific columns, or cleaning rules..."):
        if not api_key:
            st.warning("Please enter your OpenRouter API Key in the sidebar.")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_llm = ChatOpenAI(
                        model="meta-llama/llama-3.1-70b-instruct", 
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
                    
                    # Context construction
                    context_str = ""
                    if uploaded_file and 'profile_data' in locals():
                        context_str = f"Dataset Metadata: {json.dumps(profile_data, indent=2)}"
                    
                    system_msg = (
                        "You are a helpful Data Quality Expert. Answer user questions about the dataset based on the provided metadata profile.\n"
                        "Explain data issues clearly and suggest why certain cleaning steps (like imputation or dropping) might be needed.\n"
                        f"{context_str}"
                    )
                    
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = chat_llm.invoke(messages)
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    
                except Exception as e:
                    st.error(f"Chat Error: {e}")

if __name__ == "__main__":
    main()
