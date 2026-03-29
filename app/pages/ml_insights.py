"""
ML Insights Page
=================
Fully automated ML-powered visual insights:
  • Anomaly Detection  — IsolationForest overlay scatter
  • Customer Clustering — K-Means colored scatter
  • Trend & Forecast   — Linear forecast overlay
All computations run automatically based on detected column types.
"""

import os
import sys

import streamlit as st
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset_detector import DatasetDetector, resolve_column
from components.charts import (
    anomaly_overlay_chart,
    cluster_scatter_chart,
    trend_forecast_chart,
    chart_has_meaningful_data,
)
from components.theme import apply_dark_page_style


# ─── Helpers ──────────────────────────────────────────────────────

def _safe_import_ml():
    """Lazily import ML modules so the page works even if sklearn is missing."""
    try:
        from ml.anomaly_detection import detect_anomalies, get_anomaly_report
        from ml.clustering import perform_clustering, get_cluster_summary
        from ml.trend_analysis import (
            compute_trend_signal,
            forecast_simple,
            get_trend_report,
        )
        return {
            "detect_anomalies": detect_anomalies,
            "get_anomaly_report": get_anomaly_report,
            "perform_clustering": perform_clustering,
            "get_cluster_summary": get_cluster_summary,
            "compute_trend_signal": compute_trend_signal,
            "forecast_simple": forecast_simple,
            "get_trend_report": get_trend_report,
        }
    except ImportError as exc:
        st.error(f"⚠️ ML dependencies missing: {exc}")
        return None


def _pick_xy(numeric_cols, preferred_x=None, preferred_y=None):
    """Choose two distinct numeric columns for scatter axes."""
    candidates = list(numeric_cols)
    x = preferred_x if preferred_x in candidates else (candidates[0] if candidates else None)
    remaining = [c for c in candidates if c != x]
    y = preferred_y if preferred_y in remaining else (remaining[0] if remaining else None)
    return x, y


# ─── Page Render ──────────────────────────────────────────────────

def render():
    apply_dark_page_style()
    st.markdown("# 🧠 ML Insights")
    st.markdown(
        "Automated machine-learning signals — anomaly detection, clustering, "
        "and trend forecasting — generated from your data."
    )
    st.markdown("---")

    if "datasets" not in st.session_state or not st.session_state["datasets"]:
        st.warning("⚠️ No datasets loaded. Go to **Data Upload** to load data.")
        return

    ml = _safe_import_ml()
    if ml is None:
        return

    datasets = st.session_state["datasets"]

    for name, df in datasets.items():
        detector = DatasetDetector(df, name)
        numeric_cols = detector.numeric_columns
        date_col = detector.get_primary_date()
        primary_metric = detector.get_primary_metric()

        st.markdown(f"## 📊 {name.title()}")

        if len(numeric_cols) < 2:
            st.info(f"Dataset **{name}** has fewer than 2 numeric columns — "
                    "ML visuals require at least 2.")
            st.markdown("---")
            continue

        tabs = st.tabs([
            "🚨 Anomaly Detection",
            "🎯 Clustering",
            "📈 Trend & Forecast",
        ])

        # ─── 1. Anomaly Detection ─────────────────────────────────
        with tabs[0]:
            _render_anomaly_section(df, detector, ml, numeric_cols)

        # ─── 2. Clustering ────────────────────────────────────────
        with tabs[1]:
            _render_clustering_section(df, detector, ml, numeric_cols)

        # ─── 3. Trend & Forecast ──────────────────────────────────
        with tabs[2]:
            _render_trend_section(df, detector, ml, date_col, primary_metric)

        st.markdown("---")


# ─── Section Renderers ───────────────────────────────────────────

def _render_anomaly_section(df, detector, ml, numeric_cols):
    """Run IsolationForest and display overlay scatter + report."""
    try:
        features = numeric_cols[:8]  # cap to avoid slow runs
        result_df, meta = ml["detect_anomalies"](df, features=features)

        n_anomalies = meta["anomalies_found"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records Analyzed", f"{meta['total_records']:,}")
        with col2:
            st.metric("Anomalies Found", f"{n_anomalies:,}")
        with col3:
            st.metric("Anomaly Rate", f"{meta['anomaly_rate']}%")

        # Pick best axes for the scatter
        preferred_x = detector.get_primary_metric()
        preferred_y = (detector.monetary_columns[1]
                       if len(detector.monetary_columns) > 1
                       else (numeric_cols[1] if len(numeric_cols) > 1 else None))
        x_col, y_col = _pick_xy(numeric_cols, preferred_x, preferred_y)

        if x_col and y_col:
            fig = anomaly_overlay_chart(result_df, x_col, y_col)
            if fig is not None and chart_has_meaningful_data(fig):
                st.plotly_chart(fig, use_container_width=True)

            # Second scatter with different axes if enough columns
            alt_cols = [c for c in numeric_cols if c not in (x_col, y_col)]
            if alt_cols:
                x2, y2 = x_col, alt_cols[0]
                fig2 = anomaly_overlay_chart(result_df, x2, y2)
                if fig2 is not None and chart_has_meaningful_data(fig2):
                    st.plotly_chart(fig2, use_container_width=True)

        # Text report
        with st.expander("📋 Detailed Anomaly Report"):
            report = ml["get_anomaly_report"](result_df, features=features)
            st.markdown(f"```\n{report}\n```")

    except Exception as exc:
        st.warning(f"Could not run anomaly detection: {exc}")


def _render_clustering_section(df, detector, ml, numeric_cols):
    """Run K-Means and display cluster scatter + summary."""
    try:
        features = numeric_cols[:6]
        n_clusters = min(5, max(2, len(df) // 100))  # auto-pick k

        result_df, meta = ml["perform_clustering"](
            df, features=features, n_clusters=n_clusters
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters", meta["n_clusters"])
        with col2:
            st.metric("Silhouette Score", f"{meta['silhouette_score']:.3f}")
        with col3:
            st.metric("Features Used", len(meta["features_used"]))

        # Cluster sizes bar
        sizes = meta["cluster_sizes"]
        st.markdown("**Cluster sizes:** "
                    + "  |  ".join(f"C{k}: {v}" for k, v in sizes.items()))

        # Scatter — pick two axes
        preferred_x = resolve_column(df, ["lifetime_value", "revenue", "stars",
                                          "spend", "salary", "amount"])
        preferred_y = resolve_column(df, ["satisfaction_score", "engagement_score",
                                          "churn_risk", "roi", "forks",
                                          "code_quality_score"])
        x_col, y_col = _pick_xy(numeric_cols, preferred_x, preferred_y)

        if x_col and y_col:
            fig = cluster_scatter_chart(result_df, x_col, y_col)
            if fig is not None and chart_has_meaningful_data(fig):
                st.plotly_chart(fig, use_container_width=True)

            # Second view with different Y
            alt = [c for c in numeric_cols if c not in (x_col, y_col)]
            if alt:
                fig2 = cluster_scatter_chart(result_df, x_col, alt[0])
                if fig2 is not None and chart_has_meaningful_data(fig2):
                    st.plotly_chart(fig2, use_container_width=True)

        # Text summary
        with st.expander("📋 Cluster Summary"):
            summary = ml["get_cluster_summary"](result_df, features=features)
            st.markdown(f"```\n{summary}\n```")

    except Exception as exc:
        st.warning(f"Could not run clustering: {exc}")


def _render_trend_section(df, detector, ml, date_col, primary_metric):
    """Compute trend signal + forecast and display chart."""
    if not date_col:
        st.info("No date column detected — trend analysis requires time-series data.")
        return
    if not primary_metric:
        st.info("No numeric metric found for trend analysis.")
        return

    try:
        # Trend signal
        signal = ml["compute_trend_signal"](df, date_col, primary_metric)

        direction_emoji = {
            "upward": "📈", "downward": "📉", "flat": "➡️",
            "insufficient_data": "❓"
        }.get(signal["trend_direction"], "❓")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Direction",
                      f"{direction_emoji} {signal['trend_direction'].title()}")
        with col2:
            st.metric("Total Growth", f"{signal['total_growth_pct']}%")
        with col3:
            st.metric("R² (fit quality)", f"{signal['r_squared']:.3f}")
        with col4:
            st.metric("Volatility", f"{signal['volatility']}%")

        # Forecast
        forecast_df = ml["forecast_simple"](
            df, date_col, primary_metric, periods=3, freq="M"
        )

        # Build historical df for the chart
        actual = forecast_df[forecast_df["type"] == "actual"].copy()
        forecast_only = forecast_df[forecast_df["type"] == "forecast"].copy()

        # Connect forecast to last actual point
        if not actual.empty and not forecast_only.empty:
            bridge = actual.tail(1).copy()
            bridge["type"] = "forecast"
            forecast_only = pd.concat([bridge, forecast_only], ignore_index=True)

        fig = trend_forecast_chart(
            actual, forecast_only,
            date_col="date", value_col="value",
            title=f"{primary_metric.replace('_', ' ').title()} — Trend & 3-Month Forecast"
        )
        if fig is not None and chart_has_meaningful_data(fig):
            st.plotly_chart(fig, use_container_width=True)

        # Iterate over other metrics for extra trends
        other_metrics = [c for c in detector.monetary_columns
                         if c != primary_metric][:2]
        for metric in other_metrics:
            try:
                sig = ml["compute_trend_signal"](df, date_col, metric)
                fc = ml["forecast_simple"](df, date_col, metric, periods=3)
                act = fc[fc["type"] == "actual"]
                fco = fc[fc["type"] == "forecast"]
                if not act.empty and not fco.empty:
                    bridge = act.tail(1).copy()
                    bridge["type"] = "forecast"
                    fco = pd.concat([bridge, fco], ignore_index=True)
                fig2 = trend_forecast_chart(
                    act, fco,
                    date_col="date", value_col="value",
                    title=f"{metric.replace('_', ' ').title()} — Trend & Forecast"
                )
                if fig2 is not None and chart_has_meaningful_data(fig2):
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass

        # Text report
        with st.expander("📋 Trend Report"):
            metrics_map = {primary_metric.replace("_", " ").title(): primary_metric}
            for m in other_metrics:
                metrics_map[m.replace("_", " ").title()] = m
            report = ml["get_trend_report"](df, date_col, metrics_map)
            st.markdown(f"```\n{report}\n```")

    except Exception as exc:
        st.warning(f"Could not run trend analysis: {exc}")


if __name__ == "__main__":
    render()
