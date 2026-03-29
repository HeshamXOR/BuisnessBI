"""
Visualizations Page
====================
Dynamic interactive Plotly dashboards — auto-generates charts for ANY dataset.
Also keeps domain-specific charts for known dataset types.
"""

import os
import sys

import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset_detector import DatasetDetector
from components.charts import (
    revenue_trend_chart, top_products_chart, revenue_by_region_chart,
    revenue_by_category_chart, campaign_performance_chart, channel_comparison_chart,
    customer_segments_chart, churn_risk_distribution_chart, segment_ltv_chart,
    github_stats_chart, language_popularity_chart, code_quality_chart,
    auto_generate_charts, chart_has_meaningful_data
)
from components.theme import apply_dark_page_style


def _try_chart(chart_fn, df, label):
    """Safely render only meaningful domain-specific charts."""
    try:
        fig = chart_fn(df)
        if fig is not None and chart_has_meaningful_data(fig):
            st.plotly_chart(fig, use_container_width=True)
            return True
        return False
    except Exception:
        return False


def render():
    """Render the Visualizations page."""
    apply_dark_page_style()
    st.markdown("# 📊 Interactive Visualizations")
    st.markdown("Explore your data through auto-generated and domain-specific charts.")
    st.markdown("---")

    if "datasets" not in st.session_state or not st.session_state["datasets"]:
        st.warning("⚠️ No datasets loaded. Go to **Data Upload** to load data.")
        return

    datasets = st.session_state["datasets"]

    for name, df in datasets.items():
        detector = DatasetDetector(df, name)
        detected = detector.detected_type

        st.markdown(f"## 📊 {name.title()} — *{detected.title()} Data* "
                     f"({detector.confidence:.0%} confidence)")

        rendered_domain = 0

        # ─── Domain-Specific Charts (if recognized) ────────────

        if detected == "sales":
            col1, col2 = st.columns(2)
            with col1:
                rendered_domain += int(_try_chart(revenue_trend_chart, df, "Revenue Trend"))
            with col2:
                rendered_domain += int(_try_chart(top_products_chart, df, "Top Products"))
            col3, col4 = st.columns(2)
            with col3:
                rendered_domain += int(_try_chart(revenue_by_region_chart, df, "Revenue by Region"))
            with col4:
                rendered_domain += int(_try_chart(revenue_by_category_chart, df, "Revenue by Category"))

        elif detected == "marketing":
            col1, col2 = st.columns(2)
            with col1:
                rendered_domain += int(_try_chart(campaign_performance_chart, df, "Campaign Perf"))
            with col2:
                rendered_domain += int(_try_chart(channel_comparison_chart, df, "Channel Comparison"))

        elif detected == "customers":
            col1, col2 = st.columns(2)
            with col1:
                rendered_domain += int(_try_chart(customer_segments_chart, df, "Segments"))
            with col2:
                rendered_domain += int(_try_chart(churn_risk_distribution_chart, df, "Churn Risk"))
            rendered_domain += int(_try_chart(segment_ltv_chart, df, "Segment LTV"))

        elif detected == "tech":
            col1, col2 = st.columns(2)
            with col1:
                rendered_domain += int(_try_chart(github_stats_chart, df, "GitHub Stats"))
            with col2:
                rendered_domain += int(_try_chart(language_popularity_chart, df, "Language Pop"))
            rendered_domain += int(_try_chart(code_quality_chart, df, "Code Quality"))

        else:
            # ─── Unknown dataset — use auto-generated charts ───
            st.caption("📐 Auto-generated charts based on detected column types")

        if detected != "generic" and rendered_domain == 0:
            st.info("No strong domain charts available from current data. Showing intelligent auto-charts below.")

        # ─── Always show auto-generated charts too ─────────────
        with st.expander(f"🔧 Auto-Detected Charts for {name.title()}", expanded=(detected == "generic")):
            chart_specs = detector.get_chart_recommendations()
            auto_charts = auto_generate_charts(df, chart_specs, max_charts=6)

            if auto_charts:
                for i in range(0, len(auto_charts), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(auto_charts):
                            with col:
                                chart_title, fig = auto_charts[idx]
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No auto-charts could be generated for this dataset.")

        st.markdown("---")


if __name__ == "__main__":
    render()
