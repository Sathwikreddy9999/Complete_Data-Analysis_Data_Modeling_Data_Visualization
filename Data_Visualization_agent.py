import streamlit as st
import pandas as pd
import requests
import io
import json
from langchain_openai import ChatOpenAI
import streamlit.components.v1 as components
import os
from style_utils import apply_apple_style

def generate_design_prompt(df_head, df_info, business_domain, report_type, api_key):
    """
    Analyzes data and generates a specific PROMPT for the coding agent.
    """
    try:
        llm = ChatOpenAI(
            model="meta-llama/llama-3.1-70b-instruct",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3
        )
        
        system_prompt = f"You are a Senior Design Architect specializing in {report_type} dashboards."
        user_prompt = f"""
Goal: Write a strict, detailed INSTRUCTION PROMPT for a Frontend Developer to build a {report_type}-style dashboard.

Context:
- Domain: {business_domain}
- Style: {report_type} (Strictly mimic the layout, color palette, and component style of {report_type})
- Data Sample: {df_head}
- Data Info: {df_info}

Your Prompts MUST include:
1. Specific layout instructions typical of {report_type} (e.g., Tiled layout for Tableau, Canvas for Power BI).
2. Which specific columns to use for X and Y axes.
3. Which chart types (Bar, Line, etc.) are best for this data.
4. Color palette instructions matching {report_type}'s default themes.

Output only the PROMPT text that I will feed into the coding agent.
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Prompt generation failed: {e}"

def generate_dashboard_html(df_head, df_info, business_domain, report_type, design_safe_prompt, api_key):
    """
    Generates HTML dashboard using the generated prompt.
    """
    try:
        llm = ChatOpenAI(
            model="meta-llama/llama-3.1-70b-instruct",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2
        )
        
        # specific layout rules
        layout_rules = ""
        if report_type == "Power BI":
            layout_rules = """
            - LAYOUT: Use a 'Canvas' style approach. Main container background #f3f2f1.
            - CARDS: White background (#ffffff) with subtle shadow (box-shadow: 0 4px 8px rgba(0,0,0,0.1)) and rounded corners (5px).
            - HEADER: Top navigation bar, left aligned title.
            - KPIs: Top row, distinct 'Card' visual.
            """
        elif report_type == "Tableau":
            layout_rules = """
            - LAYOUT: Tiled, dashboard grid layout. Minimal whitespace between containers.
            - FONT: Use a font similar to 'Tableau Book' or system sans-serif.
            - CONTAINERS: Distinct borders, white background.
            - LEGENDS: Floating or right-sidebar attached.
            """
        elif report_type == "Looker":
            layout_rules = """
            - LAYOUT: Top-heavy filter bar (gray/purple accent).
            - STYLE: Very flat design, no shadows.
            - FONTS: 'Open Sans' or 'Roboto'.
            - COLORS: Use Looker's default purple/gray palette for accents.
            """
        else: # Generic/Google Data Studio
             layout_rules = "- LAYOUT: Clean 2-column or 3-column grid. Standard Material Design vibes."

        system_prompt = f"""You are a Frontend Data Visualization Expert. 
Your goal is to write a SINGLE, self-contained HTML file that visualizes the provided data.

Style Preference: **{report_type}** 
STRICT LAYOUT RULES:
{layout_rules}

Requirements:
1. Use **TailwindCSS** for styling (via CDN).
2. Use **Chart.js** (via CDN) for visualizations.
3. The design must be modern, clean, and business-professional.
4. Input data is provided as a CSV string snippet. Embed this data directly into the JavaScript variables for the charts.
5. **CRITICAL**: The main page body background MUST always be WHITE (#ffffff). Do not use dark mode or gray backgrounds.
6. EXECUTING THE DESIGN REQUIREMENTS:
"""

        user_prompt = f"""
Domain: {business_domain}
Data Sample (First 5 rows):
{df_head}

Data Info:
{df_info}

DESIGN INSTRUCTIONS (Follow Strictly):
{design_safe_prompt}

Generate the full HTML dashboard code now. Return ONLY the raw HTML.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm.invoke(messages)
        html_code = response.content.strip()
        
        # Cleanup
        if html_code.startswith("```html"):
            html_code = html_code[7:]
        elif html_code.startswith("```"):
            html_code = html_code[3:]
        if html_code.endswith("```"):
            html_code = html_code[:-3]
            
        return html_code

    except Exception as e:
        return f"Error generating dashboard: {e}"

def main():
    st.set_page_config(page_title="Data Viz Generator", page_icon=None, layout="wide")
    apply_apple_style()
    st.title("Recursive Data Visualization Agent")
    st.markdown("Upload data -> AI Analyzes -> AI Codes -> Live Dashboard.")

    # Sidebar Configuration
    st.sidebar.header("Configuration")
    default_key = "YOUR_API_KEY_HERE"
    # Secure API Key Loading
    api_key = os.getenv("OPENROUTER_API_KEY", default_key)
    
    business_domain = st.sidebar.selectbox(
        "Select Business Domain",
        ["Banking", "Utility", "Retail", "Healthcare", "Manufacturing", "Marketing", "Sales", "HR"]
    )
    
    report_type = st.sidebar.selectbox(
        "Dashboard Style",
        ["Power BI", "Tableau", "Looker", "QlikSense", "Google Data Studio"]
    )

    uploaded_file = st.file_uploader("Upload Data File (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data Analysis
            st.write("### 1. Data Snapshot")
            st.dataframe(df.head())
            
            df_head = df.head().to_csv(index=False)
            
            buffer = io.StringIO()
            df.info(buf=buffer)
            df_info = buffer.getvalue()
            
            st.info(f"Loaded {len(df)} rows. Agent is ready.")

            if st.button("Analyze & Generate Dashboard"):
                if not api_key:
                    st.error("Please provide an API Key.")
                    return

                # 1. Prompt Generation Step
                with st.spinner(f"Analyzing Data & Engineering {report_type} Style Prompt..."):
                    generated_prompt = generate_design_prompt(df_head, df_info, business_domain, report_type, api_key)
                
                # 2. Code Generation Step (Using the Prompt)
                with st.spinner("Writing HTML/JS Code based on Prompt..."):
                    html_code = generate_dashboard_html(df_head, df_info, business_domain, report_type, generated_prompt, api_key)
                
                # 3. Output Handling
                if "Error" in html_code:
                    st.error(html_code)
                else:
                    st.success("Dashboard Code Generated Successfully!")
                    
                    
                    
                    # SECTION 1: PREVIEW (TOP) - Always Visible
                    st.divider()
                    st.subheader(f"1. Live Dashboard Preview ({report_type} Style)")
                    components.html(html_code, height=800, scrolling=True)
                    
                    st.download_button(
                        label="Download Dashboard.html",
                        data=html_code,
                        file_name="dashboard.html",
                        mime="text/html"
                    )

                    # SECTION 2: PROMPT (BOTTOM)
                    st.divider()
                    with st.expander("View AI Design Prompt Used", expanded=False):
                        st.markdown(generated_prompt)

        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
