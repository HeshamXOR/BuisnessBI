"""Dash AI insights page using the shared LLM client."""

import os
import sys

from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import state
from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from llm.response_parser import ResponseParser
from utils.dataset_detector import DatasetDetector


DEFAULT_PROMPT = "Summarize the most important patterns, risks, opportunities, and recommended next actions from this dataset."


def _fallback_context(df, dataset_name: str) -> str:
    """Build a minimal context when rich dataset detection fails."""
    sample = df.head(5).to_string(index=False)
    return (
        f"Dataset: {dataset_name}\n"
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Columns: {', '.join(map(str, df.columns))}\n\n"
        f"Sample Data:\n{sample}"
    )


def _build_context(datasets, dataset_name: str):
    """Create robust LLM context for one dataset or all datasets."""
    if dataset_name == "__all__":
        chunks = []
        for name, df in datasets.items():
            try:
                chunks.append(DatasetDetector(df, name).get_analysis_context())
            except Exception:
                chunks.append(_fallback_context(df, name))
        return "\n\n---\n\n".join(chunks), "all loaded datasets"

    df = datasets.get(dataset_name)
    if df is None:
        raise ValueError("Selected dataset is no longer available.")

    try:
        context = DatasetDetector(df, dataset_name).get_analysis_context()
    except Exception:
        context = _fallback_context(df, dataset_name)

    return context, dataset_name.title()


def render():
    datasets = state.get_datasets()
    if not datasets:
        return html.Div(
            [
                html.H2("🤖 AI Insights"),
                dbc.Alert(
                    "No datasets loaded yet. Go to Data Upload first.", color="warning"
                ),
            ],
            className="p-3",
        )

    options = [{"label": "All Loaded Datasets", "value": "__all__"}] + [
        {"label": name.title(), "value": name} for name in datasets.keys()
    ]
    last_result = state.get_last_ai_insight()
    initial_markdown = last_result["markdown"] if last_result else ""

    return html.Div(
        [
            html.H2("🤖 AI Insights (LLM)"),
            html.P(
                "Run single-dataset LLM analysis with the same Ollama backend used by the analyst agents."
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Dataset"),
                            dcc.Dropdown(
                                id="ai-dataset-select",
                                options=options,
                                value=options[0]["value"],
                                clearable=False,
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Analysis Prompt"),
                            dcc.Textarea(
                                id="ai-prompt",
                                value=DEFAULT_PROMPT,
                                style={
                                    "width": "100%",
                                    "height": "120px",
                                    "backgroundColor": "#1A2336",
                                    "color": "#F8FAFC",
                                    "border": "1px solid #2D3A52",
                                },
                            ),
                        ],
                        width=8,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Button(
                "Generate Insight",
                id="ai-generate-btn",
                color="primary",
                className="mb-3",
            ),
            html.Div(id="ai-status", className="mb-3"),
            dbc.Card(
                dbc.CardBody(dcc.Markdown(id="ai-output", children=initial_markdown))
            ),
        ],
        className="p-3",
    )


def register_callbacks(app):
    @app.callback(
        [Output("ai-status", "children"), Output("ai-output", "children")],
        Input("ai-generate-btn", "n_clicks"),
        [State("ai-dataset-select", "value"), State("ai-prompt", "value")],
        prevent_initial_call=True,
    )
    def generate_ai_insight(n_clicks, dataset_name, prompt):
        if not n_clicks:
            return no_update, no_update

        datasets = state.get_datasets()
        if not datasets:
            return dbc.Alert(
                "No datasets are currently loaded. Upload data first.", color="warning"
            ), no_update

        selected_dataset = dataset_name or "__all__"
        try:
            context, result_label = _build_context(datasets, selected_dataset)

            client = LLMClient(request_timeout=45)
            parser = ResponseParser()
            prompt_text = PromptTemplates.dynamic_analysis(
                detected_type=(
                    selected_dataset
                    if selected_dataset != "__all__"
                    else "cross-functional"
                ),
                context=context,
            )

            prompt = (prompt or "").strip()
            if prompt:
                prompt_text += f"\n\n## User Focus\n{prompt}"

            markdown = client.generate(
                prompt=prompt_text,
                system_prompt=PromptTemplates.SYSTEM_ANALYST_STRICT,
                temperature=0.2,
            )
            markdown = parser.clean_response(markdown)
            if not markdown:
                markdown = (
                    "## No Insight Generated\n\n"
                    "The model returned an empty response. Please try again with a more specific prompt."
                )

            state.set_last_ai_insight(
                {"dataset": selected_dataset, "markdown": markdown}
            )
            return dbc.Alert(
                f"Generated insight for {result_label}.", color="success"
            ), markdown
        except Exception as e:
            return dbc.Alert(f"Could not generate insight: {e}", color="danger"), no_update
