"""
Visualizations Page
====================
Dynamic interactive Plotly dashboards — auto-generates charts for ANY dataset.
Domain-specific charts use fuzzy column matching so they work with varied schemas.
"""

import os
import sys

import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset_detector import DatasetDetector, resolve_column
from components.charts import (
    revenue_trend_chart, top_products_chart, revenue_by_region_chart,
    revenue_by_category_chart, campaign_performance_chart, channel_comparison_chart,
    customer_segments_chart, churn_risk_distribution_chart, segment_ltv_chart,
    github_stats_chart, language_popularity_chart, code_quality_chart,
    auto_generate_charts, chart_has_meaningful_data
)
from components.theme import apply_dark_page_style


def _try_chart(chart_fn, df, label, **kwargs):
    """Safely render only meaningful domain-specific charts."""
    try:
        fig = chart_fn(df, **kwargs) if kwargs else chart_fn(df)
        if fig is not None and chart_has_meaningful_data(fig):
            st.plotly_chart(fig, use_container_width=True)
            return True
        return False
    except Exception:
        return False


def _render_domain_charts(df, detected_type):
    """Render domain-specific charts with fuzzy column resolution.
    Returns count of successfully rendered charts."""
    rendered = 0

    if detected_type == "sales":
        date_col = resolve_column(df, ["date", "order_date", "transaction_date",
                                        "created_at", "sale_date"]) or "date"
        val_col = resolve_column(df, ["revenue", "total_revenue", "sales",
                                       "amount", "total_sales"]) or "revenue"

        col1, col2 = st.columns(2)
        with col1:
            rendered += int(_try_chart(
                revenue_trend_chart, df, "Revenue Trend",
                date_col=date_col, value_col=val_col))
        with col2:
            rendered += int(_try_chart(top_products_chart, df, "Top Products"))
        col3, col4 = st.columns(2)
        with col3:
            rendered += int(_try_chart(
                revenue_by_region_chart, df, "Revenue by Region"))
        with col4:
            rendered += int(_try_chart(
                revenue_by_category_chart, df, "Revenue by Category"))

    elif detected_type == "marketing":
        col1, col2 = st.columns(2)
        with col1:
            rendered += int(_try_chart(
                campaign_performance_chart, df, "Campaign Perf"))
        with col2:
            rendered += int(_try_chart(
                channel_comparison_chart, df, "Channel Comparison"))

    elif detected_type == "customers":
        col1, col2 = st.columns(2)
        with col1:
            rendered += int(_try_chart(
                customer_segments_chart, df, "Segments"))
        with col2:
            rendered += int(_try_chart(
                churn_risk_distribution_chart, df, "Churn Risk"))
        rendered += int(_try_chart(segment_ltv_chart, df, "Segment LTV"))

    elif detected_type == "tech":
        col1, col2 = st.columns(2)
        with col1:
            rendered += int(_try_chart(
                github_stats_chart, df, "GitHub Stats"))
        with col2:
            rendered += int(_try_chart(
                language_popularity_chart, df, "Language Pop"))
        rendered += int(_try_chart(code_quality_chart, df, "Code Quality"))

    return rendered


def render():
    """Render the Visualizations page."""
    apply_dark_page_style()
    st.markdown("# 📊 Interactive Visualizations")
    st.markdown(
        "Explore your data through intelligent auto-generated and "
        "domain-specific charts."
    )
    st.markdown("---")

    if "datasets" not in st.session_state or not st.session_state["datasets"]:
        st.warning("⚠️ No datasets loaded. Go to **Data Upload** to load data.")
        return

    datasets = st.session_state["datasets"]

    for name, df in datasets.items():
        detector = DatasetDetector(df, name)
        detected = detector.detected_type
        secondary = detector.secondary_type

        type_label = detected.title()
        if secondary:
            type_label += f" / {secondary.title()}"

        st.markdown(
            f"## 📊 {name.title()} — *{type_label} Data* "
            f"({detector.confidence:.0%} confidence)"
        )

        # ─── Domain-Specific Charts ────────────────────────────
        rendered_domain = 0

        if detected != "generic":
            rendered_domain += _render_domain_charts(df, detected)

        # Also try secondary type if it has different domain charts
        if secondary and secondary != detected:
            if rendered_domain > 0:
                st.markdown(f"#### 📌 Additional {secondary.title()} Charts")
            rendered_domain += _render_domain_charts(df, secondary)

        if detected == "generic":
            st.caption("📐 Auto-generated charts based on detected column types")

        if detected != "generic" and rendered_domain == 0:
            st.info(
                "No strong domain charts available from current data. "
                "Showing intelligent auto-charts below."
            )

        # ─── Auto-Generated Charts (always visible) ────────────
        st.markdown(f"### 🔧 Smart Auto-Charts for {name.title()}")
        chart_specs = detector.get_chart_recommendations()
        auto_charts = auto_generate_charts(df, chart_specs, max_charts=12)

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
