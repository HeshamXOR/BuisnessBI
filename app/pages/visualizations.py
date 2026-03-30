"""Dash visualizations page with Plotly-native rendering."""

import os
import sys

from dash import dcc, html, Input, Output, State, MATCH
import dash_bootstrap_components as dbc

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import state
from components.charts import (
    auto_chart,
    auto_generate_charts,
    campaign_performance_chart,
    channel_comparison_chart,
    chart_has_meaningful_data,
    code_quality_chart,
    customer_segments_chart,
    github_stats_chart,
    language_popularity_chart,
    nps_by_region_chart,
    nps_distribution_chart,
    nps_trend_chart,
    revenue_by_category_chart,
    revenue_by_region_chart,
    revenue_trend_chart,
    segment_ltv_chart,
    sessions_vs_nps_chart,
    top_products_chart,
    churn_risk_distribution_chart,
)
from utils.dataset_detector import DatasetDetector, resolve_column
from llm.insight_generator import generate_insights

GRAPH_CONFIG = {"displayModeBar": True, "responsive": True}


def _graph_card(fig, title: str = ""):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, className="mb-3") if title else None,
                dcc.Graph(figure=fig, config=GRAPH_CONFIG),
            ]
        ),
        className="mb-3",
    )


def _pair_rows(figures):
    rows = []
    for i in range(0, len(figures), 2):
        cols = []
        for fig_title, fig in figures[i : i + 2]:
            cols.append(dbc.Col(_graph_card(fig, fig_title), width=6))
        rows.append(dbc.Row(cols, className="mb-2"))
    return rows


def _dataset_header(detector, df):
    heatmap_fig = None
    if len(detector.numeric_columns) >= 3:
        candidate = auto_chart(
            df,
            {
                "type": "heatmap",
                "columns": detector.numeric_columns[:8],
                "title": "Correlation Heatmap",
            },
        )
        if candidate and chart_has_meaningful_data(candidate):
            heatmap_fig = candidate

    return dbc.Row(
        [
            dbc.Col(
                _graph_card(heatmap_fig, "Correlation Heatmap")
                if heatmap_fig is not None
                else dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Dataset Summary"),
                            html.P(f"Rows: {len(df):,} | Columns: {len(df.columns)}"),
                            html.P(
                                f"Primary metric: {detector.get_primary_metric() or 'N/A'} | Primary date: {detector.get_primary_date() or 'N/A'}"
                            ),
                        ]
                    ),
                    className="mb-3",
                ),
                width=6,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Detected Roles"),
                            html.Ul(
                                [
                                    html.Li(
                                        f"Date: {', '.join(detector.date_columns[:3]) if detector.date_columns else 'None'}"
                                    ),
                                    html.Li(
                                        f"Numeric: {', '.join(detector.numeric_columns[:4]) if detector.numeric_columns else 'None'}"
                                    ),
                                    html.Li(
                                        f"Categorical: {', '.join(detector.categorical_columns[:4]) if detector.categorical_columns else 'None'}"
                                    ),
                                    html.Li(
                                        f"Recommended charts: {len(detector.get_chart_recommendations())}"
                                    ),
                                ]
                            ),
                        ]
                    ),
                    className="mb-3",
                ),
                width=6,
            ),
        ]
    )


def _try_chart(chart_fn, *args, **kwargs):
    try:
        fig = chart_fn(*args, **kwargs)
        if fig is not None and chart_has_meaningful_data(fig):
            return fig
    except Exception:
        return None
    return None


def _render_domain_charts(df, detected_type):
    figures = []

    if detected_type == "sales":
        date_col = (
            resolve_column(
                df,
                ["date", "order_date", "transaction_date", "created_at", "sale_date"],
            )
            or "date"
        )
        val_col = (
            resolve_column(
                df, ["revenue", "total_revenue", "sales", "amount", "total_sales"]
            )
            or "revenue"
        )
        for title, fn, kwargs in [
            (
                "Revenue Trend",
                revenue_trend_chart,
                {"date_col": date_col, "value_col": val_col},
            ),
            ("Top Products", top_products_chart, {}),
            ("Revenue by Region", revenue_by_region_chart, {}),
            ("Revenue by Category", revenue_by_category_chart, {}),
        ]:
            fig = _try_chart(fn, df, **kwargs)
            if fig:
                figures.append((title, fig))

    elif detected_type == "marketing":
        for title, fn in [
            ("Campaign Performance", campaign_performance_chart),
            ("Channel Comparison", channel_comparison_chart),
        ]:
            fig = _try_chart(fn, df)
            if fig:
                figures.append((title, fig))

    elif detected_type == "customers":
        for title, fn in [
            ("Customer Segments", customer_segments_chart),
            ("Churn Risk Distribution", churn_risk_distribution_chart),
            ("Segment Lifetime Value", segment_ltv_chart),
        ]:
            fig = _try_chart(fn, df)
            if fig:
                figures.append((title, fig))

    elif detected_type == "tech":
        for title, fn in [
            ("GitHub Stats", github_stats_chart),
            ("Language Popularity", language_popularity_chart),
            ("Code Quality", code_quality_chart),
        ]:
            fig = _try_chart(fn, df)
            if fig:
                figures.append((title, fig))

    elif detected_type == "survey":
        for title, fn in [
            ("NPS Breakdown", nps_distribution_chart),
            ("NPS Trend", nps_trend_chart),
            ("NPS by Region", nps_by_region_chart),
            ("Sessions vs NPS", sessions_vs_nps_chart),
        ]:
            fig = _try_chart(fn, df)
            if fig:
                figures.append((title, fig))

    return figures


def render():
    datasets = state.get_datasets()
    if not datasets:
        return html.Div(
            [
                html.H2("📊 Visualizations"),
                dbc.Alert(
                    "No datasets loaded yet. Go to Data Upload first.", color="warning"
                ),
            ],
            className="p-3",
        )

    dataset_tabs = []

    for name, df in datasets.items():
        detector = DatasetDetector(df, name)
        type_label = detector.detected_type.title()
        if detector.secondary_type:
            type_label += f" / {detector.secondary_type.title()}"

        tab_children = [
            html.H3(f"{name.title()}"),
            html.P(f"Detected as {type_label} ({detector.confidence:.0%} confidence)"),
            html.Small(
                f"Roles: {len(detector.date_columns)} date, {len(detector.numeric_columns)} numeric, {len(detector.categorical_columns)} categorical, {len(detector.monetary_columns)} monetary"
            ),
            html.Hr(),
            html.H4("🧠 AI Executive Insights", className="mt-3"),
            dbc.Button(
                f"Generate Insights for {name.title()}",
                id={"type": "generate-insights-btn", "index": name},
                color="primary",
                className="mb-3"
            ),
            dcc.Loading(
                id={"type": "loading-insights", "index": name},
                type="circle",
                color="var(--accent)",
                children=html.Div(id={"type": "insights-output", "index": name}, className="mb-4")
            ),
            html.Hr(),
            _dataset_header(detector, df),
        ]

        domain_figures = []
        if detector.detected_type != "generic":
            domain_figures.extend(_render_domain_charts(df, detector.detected_type))
        if (
            detector.secondary_type
            and detector.secondary_type != detector.detected_type
        ):
            domain_figures.extend(_render_domain_charts(df, detector.secondary_type))

        if domain_figures:
            tab_children.append(html.H4("Domain-Specific Charts"))
            tab_children.extend(_pair_rows(domain_figures))
        else:
            tab_children.append(
                dbc.Alert(
                    "No strong domain-specific charts available. Showing intelligent auto-charts below.",
                    color="info",
                )
            )

        auto_specs = detector.get_chart_recommendations()
        auto_figures = auto_generate_charts(df, auto_specs, max_charts=10)
        tab_children.append(html.H4("Smart Auto-Charts"))
        if auto_figures:
            tab_children.extend(_pair_rows(auto_figures))
        else:
            tab_children.append(
                dbc.Alert(
                    "No auto-charts could be generated for this dataset.",
                    color="secondary",
                )
            )

        dataset_tabs.append(dbc.Tab(tab_children, label=name.title()))

    children = [
        html.H2("📊 Interactive Visualizations"),
        html.P(
            "Plotly-native dashboards rendered directly in Dash. Each dataset gets a dedicated tab with domain charts, auto-charts, and correlation views."
        ),
        html.Hr(),
        dbc.Tabs(dataset_tabs),
    ]

    return html.Div(children, className="p-3")


def register_callbacks(app):
    @app.callback(
        Output({"type": "insights-output", "index": MATCH}, "children"),
        Input({"type": "generate-insights-btn", "index": MATCH}, "n_clicks"),
        State({"type": "generate-insights-btn", "index": MATCH}, "id"),
        prevent_initial_call=True,
    )
    def update_insights(n_clicks, btn_id):
        if not n_clicks:
            return ""
        dataset_name = btn_id["index"]
        df = state.get_dataset(dataset_name)
        if df is None:
            return dbc.Alert("Dataset not found.", color="danger")
        
        try:
            detector = DatasetDetector(df, dataset_name)
            kpis = detector.compute_auto_kpis()
            insight_text = generate_insights(dataset_name, kpis)
            return dbc.Card(dbc.CardBody(dcc.Markdown(insight_text)), className="border-info")
        except Exception as e:
            return dbc.Alert(f"Failed to generate insights: {e}", color="danger")
