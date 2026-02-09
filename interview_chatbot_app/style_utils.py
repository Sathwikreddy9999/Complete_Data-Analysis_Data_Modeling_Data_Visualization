import streamlit as st

def apply_premium_style():
    """
    Applies a minimalist, glassmorphic 'Apple-style' theme with advanced animations.
    """
    st.markdown("""
        <style>
        /* GLOBAL DARK OVERRIDE */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
            background-color: #010101 !important;
            color: #ffffff !important;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        /* HEADER TEXT */
        h1, h2, h3, h4, h5, h6, p, span, label, li {
            color: #f5f5f7 !important;
            font-weight: 400;
        }

        /* GRADIENT H1 - ULTRA PREMIUM */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.05em !important;
            background: linear-gradient(135deg, #ffffff 0%, #a1a1a6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem !important;
            margin-bottom: 0.2rem !important;
            padding-bottom: 0.2rem;
            animation: fadeIn 1.2s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* GLASS CARD EFFECT / CHAT BUBBLES */
        .stChatMessage, [data-testid="stVerticalBlock"] > div > div > .stMarkdown {
            /* Targeting container-like elements for glass effect */
        }
        
        div.stMarkdown > div[data-testid="stContainer"] {
            background: rgba(255, 255, 255, 0.04) !important;
            backdrop-filter: blur(30px) saturate(180%);
            -webkit-backdrop-filter: blur(30px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 22px !important;
            margin-bottom: 1.5rem !important;
            padding: 24px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            animation: slideUp 0.6s cubic-bezier(0.23, 1, 0.32, 1);
        }
        
        div.stMarkdown > div[data-testid="stContainer"]:hover {
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            transform: translateY(-2px);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* SIDEBAR - DEEP DARK */
        section[data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
            width: 300px !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p {
            font-size: 0.95rem;
            color: #a1a1a6 !important;
        }

        /* BUTTONS - ACTION BLUE */
        .stButton > button {
            background: #0071e3 !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.6rem 1.8rem !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: auto;
        }
        .stButton > button:hover {
            background: #0077ed !important;
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(0, 113, 227, 0.4);
        }
        .stButton > button:active {
            transform: scale(0.98);
        }

        /* CHAT INPUT AREA */
        [data-testid="stChatInput"] {
            background-color: #010101 !important;
            padding-bottom: 2rem !important;
        }
        
        [data-testid="stChatInput"] textarea {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 15px !important;
            font-size: 1.05rem !important;
        }
        
        /* FORM INPUTS & TEXTAREA */
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
        }
        
        /* FILE UPLOADER */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px dashed rgba(255, 255, 255, 0.2) !important;
            border-radius: 16px !important;
            padding: 20px !important;
        }

        /* ONBOARDING CONTAINER */
        .onboarding-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            margin-top: 20px;
            animation: slideUp 0.8s ease-out;
        }

        /* STATUS INDICATORS */
        .listening-pulse {
            display: inline-block;
            width: 12px;
            height: 12px;
            background-color: #ff3b30;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.5); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }

        /* ALERT BOXES (Cleaned) */
        .stAlert {
            background: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 14px !important;
        }

        /* SCROLLBAR - MINIMAL */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: #333333;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #444444;
        }
        
        /* HIDE STREAMLIT BRANDING */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)
