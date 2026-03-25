import streamlit as st
import os

def apply_apple_style():
    """
    Applies a minimalist 'Apple-style' CSS theme.
    Features:
    - San Francisco / System Fonts (SF Pro)
    - Clean whitespace, no clutter
    - Rounded UI elements (12px-18px radii)
    - Gentle shadows, frosted glass effects
    - Pure colors, minimal gradients
    """
    
    # Check for API Key in environment, warn if missing (but continue)
    if not os.getenv("OPENROUTER_API_KEY"):
        # We don't stop execution here, but dev should know.
        pass

    st.markdown("""
        <style>
        /* FONT STACK */
        html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 16px;
            color: #f5f5f7 !important; /* Apple standard light text on dark */
            -webkit-font-smoothing: antialiased;
        }

        /* DARK THEME BASE */
        .stApp {
            background-color: #010101 !important; /* Deep Apple Dark */
        }

        /* HEADERS - Clean, lighter weight */
        h1, h2, h3, h4, h5, h6, label {
            color: #ffffff !important;
        }
        h1 {
            font-weight: 600;
            letter-spacing: -0.02em;
            font-size: 2.2rem;
            padding-bottom: 0.5rem;
        }
        h2 {
            font-weight: 600;
            letter-spacing: -0.015em;
            font-size: 1.5rem;
            margin-top: 1.5rem;
        }
        h3 {
            font-weight: 500;
            font-size: 1.25rem;
        }

        /* BUTTONS - Apple Human Interface Guidelines */
        .stButton > button {
            background-color: #0071e3 !important; /* Apple Blue */
            color: white !important;
            border: none !important;
            border-radius: 980px; /* Pill shape or highly rounded */
            padding: 0.6rem 1.2rem;
            font-size: 0.95rem;
            font-weight: 500;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            transition: all 0.2s cubic-bezier(0.25, 0.1, 0.25, 1);
        }
        .stButton > button:hover {
            background-color: #0077ED;
            box-shadow: 0 4px 12px rgba(0,113,227,0.3);
            transform: translateY(-1px);
        }
        .stButton > button:active {
            transform: scale(0.97);
            background-color: #006edb;
        }
        
        /* SECONDARY / OUTLINE BUTTONS (if any) */
        /* Streamlit doesn't distinguish easily without keys, but general override */

        /* INPUTS & CARDS */
        div[data-testid="stExpander"], div.stDataFrame, div[data-testid="stJson"], .stDataFrame > div {
            background: #1c1c1e !important; /* Apple dark grey card */
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.08) !important;
            box-shadow: none;
            color: #f5f5f7 !important;
        }
        
        .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div {
            background-color: #2c2c2e !important; /* Apple Input Dark Grey */
            border-radius: 10px;
            border: 1px solid transparent !important;
            padding: 0.5rem;
            color: #ffffff !important;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            background-color: #3a3a3c !important;
            border-color: #0071e3 !important;
            box-shadow: 0 0 0 3px rgba(0,113,227,0.3) !important;
        }

        /* SIDEBAR - Translucent/Frosted look simulation */
        section[data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
        }
        
        /* HIDE STREAMLIT BRANDING TEXT (Cleanliness) */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;} /* Hides the top colored bar if possible */
        
        /* ALERTS - Soften them */
        div[data-baseweb="notification"] {
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
            background-color: #1c1c1e !important;
            color: #f5f5f7 !important;
        }
        
        </style>
    """, unsafe_allow_html=True)
