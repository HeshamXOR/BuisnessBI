"""
Data Overview Page
===================
Display summary statistics, shapes, dtypes, and data quality for all loaded datasets.
"""

import os
import sys
import streamlit as st
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_loader import get_dataset_info
from utils.analysis import get_summary_statistics, compute_correlation_matrix
from components.ui_elements import section_header
from components.theme import apply_dark_page_style


def render():
    """Render the Data Overview page."""
    apply_dark_page_style()
    st.markdown("# 📋 Data Overview")
    st.markdown("Explore dataset structure, statistics, and quality metrics.")
    st.markdown("---")

    if "datasets" not in st.session_state or not st.session_state["datasets"]:
        st.warning("⚠️ No datasets loaded. Go to **Data Upload** to load data.")
        return

    datasets = st.session_state["datasets"]

    # ─── Dataset Selector ──────────────────────────────────────

    selected_dataset = st.selectbox(
        "Select Dataset",
        list(datasets.keys()),
        format_func=lambda x: f"📊 {x.title()}"
    )

    df = datasets[selected_dataset]
    info = get_dataset_info(df)

    # ─── Quick Stats ───────────────────────────────────────────

    st.markdown("### 📏 Dataset Shape")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{info['rows']:,}")
    with col2:
        st.metric("Columns", info['columns'])
    with col3:
        st.metric("Memory", f"{info['memory_mb']} MB")
    with col4:
        total_missing = sum(info['missing_values'].values())
        st.metric("Missing Values", f"{total_missing:,}")

    st.markdown("---")

    # ─── Column Info ───────────────────────────────────────────

    tabs = st.tabs(["📝 Data Types", "🔢 Statistics", "📉 Missing Values",
                     "🔗 Correlations", "👁️ Data Preview"])

    with tabs[0]:
        st.markdown("### Column Data Types")
        dtype_df = pd.DataFrame({
            "Column": info["column_names"],
            "Data Type": [info["dtypes"][col] for col in info["column_names"]],
            "Category": ["Numeric" if col in info["numeric_columns"]
                         else "Categorical" for col in info["column_names"]]
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("### Descriptive Statistics")
        stats = get_summary_statistics(df)

        if stats["numeric_summary"]:
            st.markdown("#### Numeric Columns")
            numeric_stats_df = pd.DataFrame(stats["numeric_summary"]).T
            st.dataframe(numeric_stats_df.round(2), use_container_width=True)

        if stats["categorical_summary"]:
            st.markdown("#### Categorical Columns")
            for col, col_stats in stats["categorical_summary"].items():
                with st.expander(f"📌 {col} ({col_stats['unique_values']} unique)"):
                    st.write(f"Top value: **{col_stats['top_value']}** "
                             f"({col_stats['top_count']} occurrences)")
                    st.write("Distribution:", col_stats['distribution'])

    with tabs[2]:
        st.markdown("### Missing Values Analysis")
        missing_data = pd.DataFrame({
            "Column": list(info["missing_values"].keys()),
            "Missing Count": list(info["missing_values"].values()),
            "Missing %": list(info["missing_pct"].values())
        })
        missing_data = missing_data[missing_data["Missing Count"] > 0]

        if missing_data.empty:
            st.success("✅ No missing values detected!")
        else:
            st.dataframe(missing_data, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("### Correlation Matrix")
        corr = compute_correlation_matrix(df)
        if not corr.empty:
            import plotly.express as px
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig.update_layout(
                paper_bgcolor="#1A1D23",
                plot_bgcolor="#0E1117",
                font=dict(color="#FAFAFA")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns for correlation analysis.")

    with tabs[4]:
        st.markdown("### Data Preview")
        n_rows = st.slider("Number of rows", 5, 50, 10)
        st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
