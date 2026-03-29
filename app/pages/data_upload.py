"""
Data Upload / Load Page
========================
Upload CSV files or load pre-generated datasets.
Supports ANY CSV — the system auto-detects dataset type.
"""

import os
import sys
import streamlit as st
import pandas as pd
import io

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_loader import load_all_datasets, get_dataset_info
from utils.dataset_detector import DatasetDetector
from components.theme import apply_dark_page_style


def _read_uploaded_csv(uploaded_file, sep_value, encoding_choice: str) -> pd.DataFrame:
    """Robust CSV reader with encoding fallback for arbitrary files."""
    raw = uploaded_file.getvalue()
    encodings = [encoding_choice] if encoding_choice != "auto" else ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                sep=sep_value,
                engine="python" if sep_value is None else "c",
                encoding=enc,
            )
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not parse CSV. Last parser error: {last_error}")


def render():
    """Render the Data Upload page."""
    apply_dark_page_style()
    st.markdown("# 📤 Data Upload / Load")
    st.markdown(
        "Upload **any business CSV** or load the demo datasets. "
        "The system automatically detects dataset type and structure."
    )
    st.markdown("---")

    # ─── Option 1: Load Pre-generated Datasets ────────────────

    st.markdown("### 🗄️ Load Demo Datasets")
    st.info(
        "Pre-generated datasets: Sales (1K rows), Marketing (500), "
        "Customers (800), GitHub (600)."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Load Demo Data", type="primary"):
            with st.spinner("Loading datasets..."):
                try:
                    datasets = load_all_datasets("data")
                    st.session_state["datasets"] = datasets
                    st.success(f"✅ Loaded {len(datasets)} datasets!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ─── Option 2: Upload ANY CSV ─────────────────────────────

    st.markdown("---")
    st.markdown("### 📁 Upload Any CSV")
    st.caption("Upload any business CSV — the system auto-detects the dataset type")

    parse_col1, parse_col2 = st.columns(2)
    with parse_col1:
        delimiter_choice = st.selectbox(
            "Delimiter",
            ["Auto-detect", ",", ";", "Tab", "|"],
            index=0,
            help="Use Auto-detect for most files. Change only if parsing looks wrong.",
        )
    with parse_col2:
        encoding_choice = st.selectbox(
            "Encoding",
            ["auto", "utf-8", "utf-8-sig", "cp1252", "latin1"],
            index=0,
            help="Auto tries common encodings for uploaded files.",
        )

    sep_value = {
        "Auto-detect": None,
        ",": ",",
        ";": ";",
        "Tab": "\t",
        "|": "|",
    }[delimiter_choice]

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if "datasets" not in st.session_state:
            st.session_state["datasets"] = {}

        for uploaded_file in uploaded_files:
            try:
                df = _read_uploaded_csv(uploaded_file, sep_value, encoding_choice)
                name = os.path.splitext(uploaded_file.name)[0]
                st.session_state["datasets"][name] = df

                # Auto-detect type
                detector = DatasetDetector(df, name)
                summary = detector.get_detection_summary()

                st.success(
                    f"✅ **{name}**: {len(df)} rows × {len(df.columns)} cols — "
                    f"Detected as **{summary['detected_type'].title()}** "
                    f"({summary['confidence']:.0%} confidence)"
                )

                with st.expander(f"🔍 Detection Details: {name}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Type", summary["detected_type"].title())
                    with col2:
                        st.metric("Confidence", f"{summary['confidence']:.0%}")
                    with col3:
                        st.metric("Charts Available", summary["recommended_charts"])

                    missing_pct = (df.isna().sum().sum() / max(1, df.shape[0] * df.shape[1])) * 100
                    st.caption(
                        f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]} | "
                        f"Missing cells: {missing_pct:.2f}%"
                    )

                    if summary["date_columns"]:
                        st.write(f"📅 Date columns: {', '.join(summary['date_columns'])}")
                    if summary["monetary_columns"]:
                        st.write(f"💰 Monetary columns: {', '.join(summary['monetary_columns'])}")
                    if summary["categorical_columns"]:
                        st.write(f"📁 Category columns: {', '.join(summary['categorical_columns'][:5])}")

                    st.markdown("**Preview (first 8 rows)**")
                    st.dataframe(df.head(8), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Error reading {uploaded_file.name}: {str(e)}")

    # ─── Show Loaded Datasets ─────────────────────────────────

    if "datasets" in st.session_state and st.session_state["datasets"]:
        st.markdown("---")
        st.markdown("### ✅ Currently Loaded Datasets")

        for name, df in st.session_state["datasets"].items():
            detector = DatasetDetector(df, name)
            info = get_dataset_info(df)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"📊 {name.title()}", f"{info['rows']} rows")
            with col2:
                st.metric("Columns", info['columns'])
            with col3:
                st.metric("Type", detector.detected_type.title())
            with col4:
                st.metric("Memory", f"{info['memory_mb']} MB")

    # ─── Re-Generate ──────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 🔄 Re-Generate Demo Datasets")
    if st.button("♻️ Re-Generate"):
        with st.spinner("Generating..."):
            try:
                from data.generate_datasets import generate_all_datasets
                datasets_dict = generate_all_datasets("data")
                st.session_state["datasets"] = {
                    "sales": datasets_dict["sales_data"],
                    "marketing": datasets_dict["marketing_data"],
                    "customers": datasets_dict["customers_data"],
                    "github": datasets_dict["github_repos"]
                }
                st.success("✅ New datasets generated!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    render()
