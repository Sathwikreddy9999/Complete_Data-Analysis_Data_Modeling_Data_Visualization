import re
import pandas as pd
import streamlit as st
import json
import numpy as np
import os
from sklearn.impute import SimpleImputer
from rapidfuzz import process, fuzz
from langchain_openai import ChatOpenAI
from style_utils import apply_apple_style

def apply_advanced_cleaning(df, selected_challenges):
    df_clean = df.copy()
    log = []

    # 1. Colors/Nominal -> One-Hot Encoding (drop_first=True)
    if "Colors/Nominal" in selected_challenges:
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
            log.append(f"Applied One-Hot Encoding (drop_first) on: {list(cat_cols)}")
        else:
            log.append("No categorical columns found for One-Hot Encoding.")

    # 2. Mixed Formats -> Regex Cleaning (Extract numbers)
    if "Mixed Formats" in selected_challenges:
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                sample = df_clean[col].dropna().astype(str).head(10)
                if any(sample.str.match(r'.*\d+.*')):
                    extracted = df_clean[col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
                    if extracted.notna().sum() > 0.5 * len(df_clean):
                        df_clean[col] = pd.to_numeric(extracted, errors='coerce')
                        log.append(f"Fixed Mixed Formats in column: {col}")

    # 3. Missing Values -> Logical Imputation (Mean vs Median)
    if "Missing Values" in selected_challenges:
        num_cols = df_clean.select_dtypes(include=['number']).columns
        for col in num_cols:
            if df_clean[col].isnull().any():
                skewness = df_clean[col].skew()
                # If skewed, use median. Otherwise mean.
                if abs(skewness) > 1:
                    strategy = 'median'
                    fill_val = df_clean[col].median()
                else:
                    strategy = 'mean'
                    fill_val = df_clean[col].mean()
                
                df_clean[col] = df_clean[col].fillna(fill_val)
                log.append(f"Imputed {col} using {strategy} (skew: {round(skewness, 2)})")
        
        cat_cols_rem = df_clean.select_dtypes(include=['object']).columns
        if len(cat_cols_rem) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_clean[cat_cols_rem] = imputer_cat.fit_transform(df_clean[cat_cols_rem])
            log.append(f"Imputed Categorical Missing Values (Mode) in: {list(cat_cols_rem)}")

    # 4. Near-Duplicates -> RapidFuzz
    if "Near-Duplicates" in selected_challenges:
        str_cols = df_clean.select_dtypes(include=['object']).columns
        for col in str_cols:
            if df_clean[col].nunique() < 2000: 
                unique_vals = df_clean[col].dropna().unique().tolist()
                replacements = {}
                for val in unique_vals:
                    if val in replacements: continue 
                    matches = process.extract(val, unique_vals, scorer=fuzz.WRatio, limit=None, score_cutoff=90)
                    for match_tuple in matches:
                        match_val = match_tuple[0]
                        if match_val != val:
                            replacements[match_val] = val
                if replacements:
                    df_clean[col] = df_clean[col].replace(replacements)
                    log.append(f"Consolidated {len(replacements)} near-duplicates in: {col}")

    # 5. Outliers -> IQR Method (Winsorization)
    if "Outliers" in selected_challenges:
        num_cols = df_clean.select_dtypes(include=['number']).columns
        for col in num_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            if not outliers.empty:
                # Winsorization: Cap at 1st and 99th percentile
                p1 = df_clean[col].quantile(0.01)
                p99 = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(lower=p1, upper=p99)
                log.append(f"Capped outliers in {col} using IQR (Winsorization at 1%/99%)")

    return df_clean, log


def get_column_profile(df):
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

                    # 2. Type Casting & Schema Enforcement
                    elif action_type == "type_cast" or action_type == "schema_enforce":
                        target_type = params.get("target_type", "")
                        if params.get("regex_cleanup") or "numeric" in target_type or "float" in target_type:
                            if params.get("flag_errors"):
                                # Scenario 4: Flag errors instead of deleting
                                original_vals = df_clean[col].copy()
                                cleaned = df_clean[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                                df_clean[col] = pd.to_numeric(cleaned, errors='coerce')
                                # Label errors
                                mask = df_clean[col].isna() & original_vals.notna()
                                if mask.any():
                                    df_clean[col] = df_clean[col].astype(object)
                                    df_clean.loc[mask, col] = "Error_Manual_Review"
                            else:
                                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                        elif "datetime" in target_type:
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

                    # 3. Duplicate Handling
                    elif action_type == "duplicate_handling":
                        if params.get("fuzzy"):
                            # Logic similar to manual fuzzy matching
                            threshold = params.get("threshold", 90)
                            unique_vals = df_clean[col].dropna().unique().tolist()
                            replacements = {}
                            for val in unique_vals:
                                if val in replacements: continue
                                matches = process.extract(val, unique_vals, scorer=fuzz.WRatio, limit=None, score_cutoff=threshold)
                                for match_tuple in matches:
                                    match_val = match_tuple[0]
                                    if match_val != val: replacements[match_val] = val
                            df_clean[col] = df_clean[col].replace(replacements)
                        else:
                            df_clean = df_clean.drop_duplicates(subset=[col])

                    # 4. Outlier Handling (Scenario 5)
                    elif action_type == "outlier_handling":
                        if params.get("method") == "iqr_winsorize":
                            Q1 = df_clean[col].quantile(0.25)
                            Q3 = df_clean[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_p = params.get("limits", [0.01, 0.99])[0]
                            upper_p = params.get("limits", [0.01, 0.99])[1]
                            p_low = df_clean[col].quantile(lower_p)
                            p_high = df_clean[col].quantile(upper_p)
                            df_clean[col] = df_clean[col].clip(lower=p_low, upper=p_high)

                    # 5. Encoding (Scenario 1)
                    elif action_type == "encoding":
                        if params.get("type") == "one_hot":
                            df_clean = pd.get_dummies(df_clean, columns=[col], drop_first=params.get("drop_first", True))
    
    except Exception as e:
        st.error(f"Error applying plan: {e}")
        
    return df_clean

def main():
    st.set_page_config(page_title="Data Cleaning Agent", page_icon=None, layout="wide")
    apply_apple_style()
    
    st.title("Data Cleaning Agent")
    st.markdown("Upload a dataset. Choose **Advanced Options** in the sidebar for specific remediations.")

    # --- Session State for Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Detailed profiling is ready. Ask me anything about your data quality."}]

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        # Secure API Key Loading
        default_key = "YOUR_API_KEY_HERE"
        api_key = os.getenv("OPENROUTER_API_KEY", default_key)
        
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
        
        st.divider()
        st.header("Advanced Cleaning (Manual)")
        cleaning_challenges = st.multiselect(
            "Select Challenges:",
            ["Colors/Nominal", "Missing Values", "Near-Duplicates", "Mixed Formats", "Outliers"],
            key="adv_options"
        )
        run_adv = st.button("Run Advanced Cleaning")


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

            st.subheader("Data Preview (First 5 Rows)")
            st.dataframe(df.head())

            # --- Advanced Cleaning Execution ---
            if run_adv:
                if not cleaning_challenges:
                    st.warning("Select items from the sidebar first.")
                else:
                    with st.spinner("Applying Advanced Strategies..."):
                        adv_clean_df, adv_logs = apply_advanced_cleaning(df, cleaning_challenges)
                        st.subheader("Advanced Cleaning Results")
                        if adv_logs:
                             for log_item in adv_logs: st.caption(f"• {log_item}")
                        else:
                             st.info("No changes needed based on selection.")
                        
                        st.dataframe(adv_clean_df.head())
                        st.download_button("Download Cleaned Data", adv_clean_df.to_csv(index=False), f"advanced_cleaned_{uploaded_file.name}")
                        st.success("You can continue to use the AI Agent below on the original data, or use this cleaned version.")
            # --- AI Agent Section ---
            st.divider()
            st.subheader("AI Data Cleaning Agent")
            special_instructions = st.text_area(
                "Special Instructions (Optional):", 
                placeholder="e.g., 'For the Age column, if the distribution is skewed, use median. Otherwise use mean.'",
                help="Instructions here will override default logic."
            )
            
            if st.button("AI Agent: Plan & Clean Data"):
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
2. User Special Instructions (Optional)

YOUR RESPONSIBILITIES:
- Detect data quality issues.
- Recommend cleaning actions based on standard practices or USER INSTRUCTIONS.
- If User provides instructions like "use IQR for outliers", you MUST follow them.
- OUTPUT JSON with 'parameters' that describe EXACTLY how to fix it.

ALLOWED CLEANING ACTIONS & PARAMETERS:
1. missing_value_handling
   - parameters: {"strategy": "drop_rows" | "impute_mean" | "impute_median" | "impute_mode" | "fill_value", "value": "..."}
2. type_cast
   - parameters: {"target_type": "numeric" | "datetime" | "string", "regex_cleanup": boolean, "flag_errors": boolean}
3. duplicate_handling
   - parameters: {"subset": ["col_name"], "fuzzy": boolean, "threshold": 0-100}
4. outlier_handling
   - parameters: {"method": "iqr_winsorize", "limits": [0.01, 0.99]}
5. encoding
   - parameters: {"type": "one_hot", "drop_first": true}
6. no_action

OUTPUT FORMAT (JSON ONLY, NO MARKDOWN):
{
  "dataset": "<dataset_name>",
  "columns": [
    {
      "column_name": "<string>",
      "detected_issues": ["<string>"],
      "recommended_actions": [
        {
          "action_type": "<allowed_action>",
          "parameters": { ... },
          "confidence": <float 0–1>
        }
      ]
    }
  ]
}
"""
                        
                        user_message = f"Dataset Name: {uploaded_file.name}\nMetadata:\n{json.dumps(profile_data, indent=2)}\n\n"
                        if special_instructions:
                            user_message += f"USER SPECIAL INSTRUCTIONS:\n{special_instructions}"
                        
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
    st.subheader("Q&A with Data Cleaning Expert")

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
