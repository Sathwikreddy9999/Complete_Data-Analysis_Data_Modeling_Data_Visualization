import streamlit as st
import pandas as pd
import requests
import os
import io
from PIL import Image

def generate_image(prompt, api_key):
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    payload = {
        "text_prompts": [{"text": prompt, "weight": 1}],
        "cfg_scale": 5,
        "sampler": "K_EULER_ANCESTRAL",
        "seed": 0,
        "steps": 25
    }
    
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        response_body = response.json()
        # NVIDIA API returns base64 encoded image in artifacts
        import base64
        image_b64 = response_body["artifacts"][0]["base64"]
        return base64.b64decode(image_b64)
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def main():
    st.set_page_config(page_title="Data Viz Idea Generator", page_icon="🎨", layout="wide")
    
    st.title("🎨 Data Visualization Idea Generator")
    st.markdown("Generate professional dashboard concepts based on your data and business needs.")

    # Sidebar Configuration
    st.sidebar.header("Configuration")
    # Default key provided by user in chat history
    default_key = "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"
    api_key = st.sidebar.text_input("NVIDIA API Key", value=default_key, type="password")
    
    if not api_key:
        st.sidebar.warning("Please enter your NVIDIA API Key to generate images.")

    # Main Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Select Report Platform",
            ["Power BI", "Tableau", "Looker", "QlikSense", "Google Data Studio"]
        )
        
    with col2:
        business_domain = st.selectbox(
            "Select Business Domain",
            ["Banking", "Utility", "Retail", "Healthcare", "Manufacturing", "Marketing", "Sales", "HR"]
        )

    uploaded_file = st.file_uploader("Upload Data File (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data Analysis for Prompt
            columns = df.columns.tolist()
            dtypes = df.dtypes.to_dict()
            
            # Identify potential metrics (numeric columns) and dimensions (categorical columns)
            numeric_cols = [col for col, dtype in dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]
            categorical_cols = [col for col, dtype in dtypes.items() if pd.api.types.is_object_dtype(dtype)]
            
            st.write("### Data Snapshot")
            st.dataframe(df.head())
            
            st.info(f"Detected {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.")

            if st.button("Generate Report Idea"):
                if not api_key:
                    st.error("Please provide an NVIDIA API Key in the sidebar.")
                else:
                    with st.spinner("Dreaming up a dashboard..."):
                        # Construct Prompt
                        metrics_str = ", ".join(numeric_cols[:5]) # Limit to top 5 to keep prompt clean
                        dimensions_str = ", ".join(categorical_cols[:5])
                        
                        prompt = (
                            f"A high-quality, professional {report_type} dashboard for the {business_domain} industry. "
                            f"The dashboard visualizes data with metrics like {metrics_str} and dimensions like {dimensions_str}. "
                            f"Modern UI, clean layout, sophisticated color palette, detailed charts, graphs, and KPIs. "
                            f"4k resolution, photorealistic, business intelligence interface."
                        )
                        
                        st.write("**Prompt used:**", prompt)
                        
                        try:
                            image_bytes = generate_image(prompt, api_key)
                            # Display Image
                            image = Image.open(io.BytesIO(image_bytes))
                            st.image(image, caption=f"Generated {report_type} Concept for {business_domain}", use_column_width=True)
                            st.success("Generation Complete!")
                        except Exception as e:
                            st.error(f"Failed to generate image: {e}")

        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
