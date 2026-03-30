"""Dash data upload and dataset loading page."""

import base64
import io
import os
import sys
from typing import List, Optional

from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import state
from utils.data_cleaner import auto_clean_dataset
from utils.data_loader import get_dataset_info, load_all_datasets
from utils.dataset_detector import DatasetDetector


def _decode_upload(contents: str) -> bytes:
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


def _read_uploaded_table(
    filename: str, contents: str, sep_value: Optional[str], encoding_choice: str
) -> pd.DataFrame:
    """Decode a Dash upload payload and parse CSV, TSV, or Excel files."""
    raw = _decode_upload(contents)
    lowered = filename.lower()
    if lowered.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))

    encodings = (
        [encoding_choice]
        if encoding_choice != "auto"
        else ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    )
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                sep=sep_value,
                engine="python" if sep_value is None else "c",
                encoding=encoding,
            )
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not parse table file. Last parser error: {last_error}")


def _dataset_cards() -> html.Div:
    datasets = state.get_datasets()
    if not datasets:
        return html.Div(
            dbc.Alert("No datasets loaded yet.", color="secondary"),
            className="mt-3",
        )

    cards: List[dbc.Card] = []
    for name, df in datasets.items():
        info = get_dataset_info(df)
        detector = DatasetDetector(df, name)
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(f"📊 {name.title()}", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Small("Rows"),
                                            html.Div(f"{info['rows']:,}"),
                                        ]
                                    )
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Small("Columns"),
                                            html.Div(f"{info['columns']}"),
                                        ]
                                    )
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Small("Detected Type"),
                                            html.Div(detector.detected_type.title()),
                                        ]
                                    )
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Small("Memory"),
                                            html.Div(f"{info['memory_mb']:.2f} MB"),
                                        ]
                                    )
                                ),
                            ]
                        ),
                    ]
                ),
                className="mb-3",
            )
        )
    return html.Div(cards)


def render():
    """Render the data upload page."""
    return html.Div(
        [
            html.H2("📤 Data Upload / Load"),
            html.P(
                "Upload CSV, TSV, or Excel files or load the demo datasets. Uploaded files are shared across the Dash app."
            ),
            html.Hr(),
            html.H4("🗄️ Demo Datasets"),
            dbc.Button(
                "Load Demo Data",
                id="dash-load-demo-btn",
                color="primary",
                className="me-2",
            ),
            dbc.Button(
                "Clear Loaded Data",
                id="dash-clear-data-btn",
                color="secondary",
                outline=True,
            ),
            html.Div(id="dash-demo-load-result", className="mt-3"),
            html.Hr(),
            html.H4("📁 Upload CSV Files"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Delimiter"),
                            dcc.Dropdown(
                                id="dash-delimiter-choice",
                                options=[
                                    {"label": "Auto-detect", "value": "auto"},
                                    {"label": "Comma (,) ", "value": ","},
                                    {"label": "Semicolon (;) ", "value": ";"},
                                    {"label": "Tab", "value": "\t"},
                                    {"label": "Pipe (|)", "value": "|"},
                                ],
                                value="auto",
                                clearable=False,
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Encoding"),
                            dcc.Dropdown(
                                id="dash-encoding-choice",
                                options=[
                                    {"label": "Auto", "value": "auto"},
                                    {"label": "UTF-8", "value": "utf-8"},
                                    {"label": "UTF-8 with BOM", "value": "utf-8-sig"},
                                    {"label": "Windows-1252", "value": "cp1252"},
                                    {"label": "Latin-1", "value": "latin1"},
                                ],
                                value="auto",
                                clearable=False,
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        dbc.Checklist(
                            id="dash-auto-clean-toggle",
                            options=[{"label": "Auto-clean uploads", "value": "clean"}],
                            value=["clean"],
                            switch=True,
                            className="pt-4",
                        ),
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            dcc.Upload(
                id="dash-upload-csv",
                children=html.Div(
                    [
                        "Drag and drop CSV / TSV / Excel files here or ",
                        html.Strong("browse"),
                    ]
                ),
                style={
                    "width": "100%",
                    "height": "100px",
                    "lineHeight": "100px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "10px",
                    "textAlign": "center",
                    "cursor": "pointer",
                    "marginBottom": "12px",
                },
                multiple=True,
            ),
            dcc.Loading(
                id="loading-upload",
                type="circle",
                color="var(--accent)",
                children=html.Div(id="dash-upload-result")
            ),
            html.Hr(),
            html.H4("✅ Loaded Datasets"),
            html.Div(id="dash-loaded-datasets", children=_dataset_cards()),
        ],
        className="p-3",
    )


def register_callbacks(app):
    @app.callback(
        [
            Output("dash-demo-load-result", "children"),
            Output("dash-loaded-datasets", "children"),
        ],
        Input("dash-load-demo-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def load_demo_data(n_clicks):
        datasets = load_all_datasets("data")
        state.set_datasets(datasets)
        return dbc.Alert(
            f"✅ Loaded {len(datasets)} demo datasets.", color="success"
        ), _dataset_cards()

    @app.callback(
        [
            Output("dash-demo-load-result", "children", allow_duplicate=True),
            Output("dash-loaded-datasets", "children", allow_duplicate=True),
        ],
        Input("dash-clear-data-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_loaded_data(n_clicks):
        state.clear_datasets()
        return dbc.Alert(
            "Cleared all loaded datasets.", color="warning"
        ), _dataset_cards()

    @app.callback(
        [
            Output("dash-upload-result", "children"),
            Output("dash-loaded-datasets", "children", allow_duplicate=True),
        ],
        Input("dash-upload-csv", "contents"),
        [
            State("dash-upload-csv", "filename"),
            State("dash-delimiter-choice", "value"),
            State("dash-encoding-choice", "value"),
            State("dash-auto-clean-toggle", "value"),
        ],
        prevent_initial_call=True,
    )
    def handle_upload(contents, filenames, delimiter, encoding, clean_flags):
        if not contents:
            return html.Div(), _dataset_cards()

        if isinstance(contents, str):
            contents = [contents]
        if isinstance(filenames, str):
            filenames = [filenames]

        sep_map = {"auto": None, ",": ",", ";": ";", "\t": "\t", "|": "|"}
        sep_value = sep_map.get(delimiter, None)
        auto_clean = "clean" in (clean_flags or [])
        alerts = []

        for content, filename in zip(contents, filenames or []):
            try:
                df = _read_uploaded_table(filename, content, sep_value, encoding)
                if auto_clean:
                    df = auto_clean_dataset(df)
                dataset_name = os.path.splitext(filename)[0]
                state.set_dataset(dataset_name, df)
                detector = DatasetDetector(df, dataset_name)
                alerts.append(
                    dbc.Alert(
                        [
                            html.Strong(f"{dataset_name}: "),
                            html.Span(
                                f"{len(df):,} rows × {len(df.columns)} cols | detected as {detector.detected_type.title()} ({detector.confidence:.0%})"
                            ),
                        ],
                        color="success",
                        className="mb-2",
                    )
                )
            except Exception as exc:
                alerts.append(
                    dbc.Alert(
                        f"❌ Error reading {filename}: {exc}",
                        color="danger",
                        className="mb-2",
                    )
                )

        return html.Div(alerts), _dataset_cards()
