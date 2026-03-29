"""Shared Streamlit theme helpers for consistent dark high-contrast pages."""

import streamlit as st


def apply_dark_page_style() -> None:
    """Apply high-contrast dark styling for individual Streamlit pages."""
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at 20% 0%, #111a2d 0%, #0B1220 45%);
                color: #FFFFFF !important;
            }

            .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
            .stText, .stCaption, label, [data-testid="stWidgetLabel"],
            [data-testid="stExpander"] p, [data-testid="stHeader"] {
                color: #FFFFFF !important;
            }

            [data-testid="stCaptionContainer"] p,
            .stCaption {
                color: #E2E8F0 !important;
            }

            [data-testid="stSidebar"] * {
                color: #FFFFFF !important;
            }

            [data-testid="stDataFrame"] * {
                color: #FFFFFF !important;
            }

            [data-baseweb="input"],
            [data-baseweb="select"],
            textarea,
            input {
                background: #1A2336 !important;
                color: #FFFFFF !important;
                border-color: #2D3A52 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
