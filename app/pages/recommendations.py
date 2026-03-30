"""Dash recommendations page."""

import os
import sys

from dash import dcc, html
from dash.dependencies import Input, Output
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


def _build_context_from_datasets() -> str:
    blocks = []
    for name, df in state.get_datasets().items():
        try:
            detector = DatasetDetector(df, name)
            blocks.append(detector.get_analysis_context())
        except Exception:
            sample = df.head(5).to_string(index=False)
            blocks.append(
                f"Dataset: {name}\n"
                f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                f"Columns: {', '.join(map(str, df.columns))}\n\n"
                f"Sample Data:\n{sample}"
            )
    return "\n\n---\n\n".join(blocks)


def render():
    datasets = state.get_datasets()
    if not datasets:
        return html.Div(
            [
                html.H2("🎯 Recommendations"),
                dbc.Alert(
                    "No datasets loaded yet. Go to Data Upload first.", color="warning"
                ),
            ],
            className="p-3",
        )

    latest = state.get_last_recommendations() or ""
    existing_multi_agent = state.get_last_multi_agent_result()
    info = (
        "Using the latest multi-agent recommendation output."
        if existing_multi_agent and latest
        else "No multi-agent recommendation cached yet. Generate one directly from the currently loaded datasets."
    )

    return html.Div(
        [
            html.H2("🎯 Recommendations"),
            html.P(
                "Generate a concise action plan from the loaded datasets or reuse the latest multi-agent recommendations."
            ),
            dbc.Alert(info, color="dark"),
            dbc.Button(
                "Generate Recommendations",
                id="recommendations-run-btn",
                color="primary",
                className="mb-3",
            ),
            html.Div(id="recommendations-status", className="mb-3"),
            dbc.Card(
                dbc.CardBody(dcc.Markdown(id="recommendations-output", children=latest))
            ),
        ],
        className="p-3",
    )


def register_callbacks(app):
    @app.callback(
        [
            Output("recommendations-status", "children"),
            Output("recommendations-output", "children"),
        ],
        Input("recommendations-run-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def generate_recommendations(n_clicks):
        try:
            multi_agent_result = state.get_last_multi_agent_result()
            if multi_agent_result and multi_agent_result.get("recommendations"):
                text = multi_agent_result["recommendations"]
                state.set_last_recommendations(text)
                return dbc.Alert(
                    "Showing recommendations from the latest multi-agent analysis run.",
                    color="success",
                ), text

            context = _build_context_from_datasets()
            client = LLMClient(request_timeout=45)
            parser = ResponseParser()
            prompt = PromptTemplates.recommendation_prompt(context)
            recommendations = client.generate(
                prompt=prompt,
                system_prompt=PromptTemplates.SYSTEM_STRATEGIST,
                temperature=0.2,
            )
            recommendations = parser.clean_response(recommendations)
            state.set_last_recommendations(recommendations)
            return dbc.Alert(
                "Recommendations generated from the current datasets.", color="success"
            ), recommendations
        except Exception as e:
            return dbc.Alert(
                f"Could not generate recommendations: {e}", color="danger"
            ), state.get_last_recommendations() or ""
