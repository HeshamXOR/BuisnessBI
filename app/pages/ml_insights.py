"""Dash ML insights page."""

import os
import sys

from dash import dcc, html
import dash_bootstrap_components as dbc

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import state
from components.charts import (
    anomaly_overlay_chart,
    chart_has_meaningful_data,
    cluster_scatter_chart,
    trend_forecast_chart,
)
from ml.anomaly_detection import detect_anomalies
from ml.clustering import perform_clustering
from ml.trend_analysis import compute_trend_signal, forecast_simple
from utils.dataset_detector import DatasetDetector


GRAPH_CONFIG = {"displayModeBar": True, "responsive": True}


def _graph(fig, title):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, className="mb-3"),
                dcc.Graph(figure=fig, config=GRAPH_CONFIG),
            ]
        ),
        className="mb-3",
    )


def render():
    datasets = state.get_datasets()
    if not datasets:
        return html.Div(
            [
                html.H2("🧠 ML Insights"),
                dbc.Alert(
                    "No datasets loaded yet. Go to Data Upload first.", color="warning"
                ),
            ],
            className="p-3",
        )

    children = [
        html.H2("🧠 ML Insights"),
        html.P(
            "Automated anomaly detection, clustering, and trend forecasting powered by scikit-learn."
        ),
        html.Hr(),
    ]

    for name, df in datasets.items():
        detector = DatasetDetector(df, name)
        children.extend(
            [
                html.H3(name.title()),
                html.P(f"Detected as {detector.detected_type.title()}"),
                html.Hr(),
            ]
        )

        numeric_cols = detector.numeric_columns[:]
        date_col = detector.get_primary_date()
        primary_metric = detector.get_primary_metric()

        # Trend analysis
        if date_col and primary_metric and primary_metric in df.columns:
            try:
                forecast_df = forecast_simple(
                    df, date_col, primary_metric, periods=3, freq="M"
                )
                historical_df = forecast_df[forecast_df["type"] == "actual"]
                signal = compute_trend_signal(df, date_col, primary_metric, freq="M")
                fig = trend_forecast_chart(
                    historical_df,
                    forecast_df,
                    date_col="date",
                    value_col="value",
                    title=f"{primary_metric.replace('_', ' ').title()} Trend & Forecast",
                )
                if chart_has_meaningful_data(fig):
                    children.append(
                        _graph(
                            fig,
                            f"Trend Forecast: {primary_metric.replace('_', ' ').title()}",
                        )
                    )
                    children.append(
                        dbc.Alert(
                            f"Trend direction: {signal['trend_direction']} | Growth: {signal['total_growth_pct']}% | Volatility: {signal['volatility']}%",
                            color="dark",
                        )
                    )
            except Exception as exc:
                children.append(
                    dbc.Alert(f"Trend analysis skipped: {exc}", color="secondary")
                )

        # Anomaly detection
        if len(numeric_cols) >= 2:
            try:
                anomaly_df, meta = detect_anomalies(
                    df, features=numeric_cols[:4], contamination=0.08
                )
                fig = anomaly_overlay_chart(
                    anomaly_df, numeric_cols[0], numeric_cols[1]
                )
                if chart_has_meaningful_data(fig):
                    children.append(
                        _graph(
                            fig,
                            f"Anomaly Detection: {numeric_cols[0]} vs {numeric_cols[1]}",
                        )
                    )
                    children.append(
                        dbc.Alert(
                            f"Anomalies found: {meta['anomalies_found']} ({meta['anomaly_rate']}%) using {', '.join(meta['features_used'])}",
                            color="dark",
                        )
                    )
            except Exception as exc:
                children.append(
                    dbc.Alert(f"Anomaly detection skipped: {exc}", color="secondary")
                )

        # Clustering
        if len(numeric_cols) >= 2:
            try:
                cluster_df, meta = perform_clustering(
                    df,
                    features=numeric_cols[:4],
                    n_clusters=min(4, max(2, len(df) // 50 or 2)),
                )
                x_col = meta["features_used"][0]
                y_col = (
                    meta["features_used"][1]
                    if len(meta["features_used"]) > 1
                    else meta["features_used"][0]
                )
                fig = cluster_scatter_chart(cluster_df, x_col, y_col)
                if chart_has_meaningful_data(fig):
                    children.append(_graph(fig, "Cluster Visualization"))
                    children.append(
                        dbc.Alert(
                            f"Clusters: {meta['n_clusters']} | Silhouette score: {meta['silhouette_score']} | Features: {', '.join(meta['features_used'])}",
                            color="dark",
                        )
                    )
            except Exception as exc:
                children.append(
                    dbc.Alert(f"Clustering skipped: {exc}", color="secondary")
                )

        children.append(html.Hr())

    return html.Div(children, className="p-3")


def register_callbacks(app):
    return None
