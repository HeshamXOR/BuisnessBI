"""
Streamlit Dashboard — Main Entry Point
========================================
Autonomous Multi-Agent Generative AI Decision Intelligence Platform

Launch: streamlit run app/main.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

from app.pages import data_upload, data_overview, visualizations
from app.pages import ai_insights, multi_agent, recommendations
from app.pages import ml_insights


# ─── Page Config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Decision Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
    :root {
        --bg-main: #0B1220;
        --bg-panel: #121A2B;
        --bg-panel-2: #1A2336;
        --text-main: #F8FAFC;
        --text-soft: #F8FAFC;
        --text-muted: #F8FAFC;
        --accent: #22D3EE;
        --accent-2: #38BDF8;
        --border: #2D3A52;
    }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #111a2d 0%, var(--bg-main) 45%);
        color: var(--text-main) !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-panel) 0%, #0E1525 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    /* Core typography visibility */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stText, .stCaption, label, [data-testid="stWidgetLabel"],
    [data-testid="stExpander"] p {
        color: var(--text-main) !important;
        line-height: 1.65;
        font-size: 1.02rem;
    }
    .stCaption {
        color: var(--text-main) !important;
        font-size: 0.95rem !important;
    }

    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        letter-spacing: 0.2px;
    }

    /* Inputs and selectors */
    [data-baseweb="input"],
    [data-baseweb="select"],
    textarea,
    input {
        background: var(--bg-panel-2) !important;
        color: var(--text-main) !important;
        border-color: var(--border) !important;
    }

    /* Metrics */
    [data-testid="stMetricLabel"] {
        color: var(--text-soft) !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }

    /* DataFrame readability */
    [data-testid="stDataFrame"] * {
        color: var(--text-main) !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--bg-panel) !important;
        border-radius: 8px !important;
        color: var(--text-main) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
        color: #041019;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(34, 211, 238, 0.35);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }

    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #22D3EE, #38BDF8, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2rem;">🧠</h1>
        <h3 class="gradient-text">AI Decision Intelligence</h3>
        <p style="color: #F8FAFC; font-size: 0.85rem;">
            Multi-Agent Analysis Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    pages = {
        "📤 Data Upload / Load": "data_upload",
        "📋 Data Overview": "data_overview",
        "📊 Visualizations": "visualizations",
        "🧠 ML Insights": "ml_insights",
        "🤖 AI Insights (LLM)": "ai_insights",
        "🔗 Multi-Agent Analysis": "multi_agent",
        "🎯 Recommendations": "recommendations"
    }

    selected_page = st.radio(
        "Navigation",
        list(pages.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")

    # LLM Status
    st.markdown("### ⚡ System Status")
    try:
        from llm.llm_client import LLMClient
        client = LLMClient()
        status = client.check_connection()
        if status.get("connected"):
            st.success(f"🟢 LLM: {client.model}")
            if status.get("model_available"):
                st.caption("Model loaded and ready")
            else:
                st.caption(f"Model '{client.model}' not pulled yet")
        else:
            st.warning("🟡 Ollama not connected")
            st.caption("Using fallback mode")
    except Exception:
        st.warning("🟡 Fallback mode active")

    st.markdown("---")
    st.caption("Built with local LLMs + Streamlit")
    st.caption("Powered by Ollama")

# ─── Page Routing ──────────────────────────────────────────────────

page_key = pages[selected_page]

if page_key == "data_upload":
    data_upload.render()
elif page_key == "data_overview":
    data_overview.render()
elif page_key == "visualizations":
    visualizations.render()
elif page_key == "ml_insights":
    ml_insights.render()
elif page_key == "ai_insights":
    ai_insights.render()
elif page_key == "multi_agent":
    multi_agent.render()
elif page_key == "recommendations":
    recommendations.render()
