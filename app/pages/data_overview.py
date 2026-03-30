"""Dash data overview page."""

import json
import os
import sys
from typing import Any

from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import state
from components.charts import auto_chart, chart_has_meaningful_data
from utils.analysis import get_summary_statistics
from utils.data_loader import get_dataset_info
from utils.dataset_detector import DatasetDetector


def _metric_card(label: str, value: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Small(label, className="text-muted"),
                    html.Div(value, className="metric-value"),
                ]
            ),
            className="mb-3",
        ),
        width=3,
    )


def _table_from_df(df: pd.DataFrame, page_size: int = 10):
    safe_df = df.copy()
    safe_df.columns = [str(c) for c in safe_df.columns]

    records = [
        {column: _to_dash_primitive(value) for column, value in row.items()}
        for row in safe_df.to_dict("records")
    ]

    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in safe_df.columns],
        data=records,
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": "#1A2336",
            "color": "#F8FAFC",
            "border": "1px solid #2D3A52",
            "textAlign": "left",
            "fontFamily": "Inter, sans-serif",
            "maxWidth": "240px",
            "whiteSpace": "normal",
        },
        style_header={"backgroundColor": "#121A2B", "fontWeight": "600"},
    )


def _to_dash_primitive(value: Any):
    """Convert DataFrame values to DataTable-compatible primitives."""
    if value is None:
        return ""

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if pd.isna(value) or value == float("inf") or value == float("-inf"):
            return ""
        return value

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return _to_dash_primitive(value.item())
        except Exception:
            pass

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def _quality_summary(df: pd.DataFrame, info: dict, detector: DatasetDetector):
    missing_total = sum(info["missing_values"].values())
    duplicate_rows = int(df.duplicated().sum())
    widest_category = max(
        (df[col].nunique() for col in detector.categorical_columns), default=0
    )
    return dbc.Row(
        [
            _metric_card(
                "Missing %",
                f"{round(missing_total / max(len(df) * max(len(df.columns), 1), 1) * 100, 2)}%",
            ),
            _metric_card("Duplicates", f"{duplicate_rows:,}"),
            _metric_card("Numeric Columns", str(len(detector.numeric_columns))),
            _metric_card("Largest Category Set", str(widest_category)),
        ]
    )


def render():
    datasets = state.get_datasets()
    if not datasets:
        return html.Div(
            [
                html.H2("📋 Data Overview"),
                dbc.Alert(
                    "No datasets loaded yet. Go to Data Upload first.", color="warning"
                ),
            ],
            className="p-3",
        )

    children = [
        html.H2("📋 Data Overview"),
        html.P(
            "Dataset summaries, previews, quality diagnostics, and descriptive statistics."
        ),
        html.Hr(),
    ]

    for name, df in datasets.items():
        info = get_dataset_info(df)
        detector = DatasetDetector(df, name)
        stats = get_summary_statistics(df)

        summary_text = (
            f"Detected as {detector.detected_type.title()}"
            + (
                f" / {detector.secondary_type.title()}"
                if detector.secondary_type
                else ""
            )
            + f" ({detector.confidence:.0%} confidence)"
        )
        children.extend(
            [
                html.H3(f"{name.title()}"),
                html.P(summary_text),
                dbc.Row(
                    [
                        _metric_card("Rows", f"{info['rows']:,}"),
                        _metric_card("Columns", str(info["columns"])),
                        _metric_card(
                            "Missing Cells", f"{sum(info['missing_values'].values()):,}"
                        ),
                        _metric_card("Memory", f"{info['memory_mb']:.2f} MB"),
                    ]
                ),
                _quality_summary(df, info, detector),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("Detected Column Roles"),
                            html.Ul(
                                [
                                    html.Li(
                                        f"Date columns: {', '.join(detector.date_columns) if detector.date_columns else 'None'}"
                                    ),
                                    html.Li(
                                        f"Monetary columns: {', '.join(detector.monetary_columns) if detector.monetary_columns else 'None'}"
                                    ),
                                    html.Li(
                                        f"Numeric columns: {', '.join(detector.numeric_columns) if detector.numeric_columns else 'None'}"
                                    ),
                                    html.Li(
                                        f"Categorical columns: {', '.join(detector.categorical_columns) if detector.categorical_columns else 'None'}"
                                    ),
                                ]
                            ),
                        ]
                    ),
                    className="mb-3",
                ),
            ]
        )

        overview_tabs = [
            dbc.Tab(_table_from_df(df.head(20)), label="Preview"),
            dbc.Tab(
                _table_from_df(
                    pd.DataFrame(
                        [
                            {
                                "column": col,
                                "dtype": dtype,
                                "missing": info["missing_values"][col],
                                "missing_pct": info["missing_pct"][col],
                            }
                            for col, dtype in info["dtypes"].items()
                        ]
                    ),
                    page_size=20,
                ),
                label="Schema",
            ),
        ]

        numeric_summary = stats.get("numeric_summary", {})
        if numeric_summary:
            numeric_df = (
                pd.DataFrame(numeric_summary)
                .T.reset_index()
                .rename(columns={"index": "column"})
            )
            overview_tabs.append(
                dbc.Tab(
                    _table_from_df(numeric_df, page_size=10), label="Numeric Summary"
                )
            )

        cat_summary = stats.get("categorical_summary", {})
        if cat_summary:
            cat_df = (
                pd.DataFrame(cat_summary)
                .T.reset_index()
                .rename(columns={"index": "column"})
            )
            overview_tabs.append(
                dbc.Tab(
                    _table_from_df(cat_df, page_size=10), label="Categorical Summary"
                )
            )

        if len(detector.numeric_columns) >= 3:
            heatmap = auto_chart(
                df,
                {
                    "type": "heatmap",
                    "columns": detector.numeric_columns[:10],
                    "title": f"{name.title()} Correlation Heatmap",
                },
            )
            if heatmap and chart_has_meaningful_data(heatmap):
                overview_tabs.append(
                    dbc.Tab(
                        dcc.Graph(
                            figure=heatmap,
                            config={"displayModeBar": True, "responsive": True},
                        ),
                        label="Correlation Heatmap",
                    )
                )

        children.append(dbc.Tabs(overview_tabs, className="mb-3"))

        children.append(html.Hr())

    return html.Div(children, className="p-3")


def register_callbacks(app):
    return None
