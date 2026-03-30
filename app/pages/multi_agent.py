"""Dash multi-agent analysis page."""

import os
import sys

from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.orchestrator import AgentOrchestrator
from app import state
from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from llm.response_parser import ResponseParser
from utils.dataset_detector import DatasetDetector


def _normalize_for_agents(datasets):
    normalized = {}
    for name, df in datasets.items():
        detector = DatasetDetector(df, name)
        dtype = detector.detected_type
        if dtype == "sales" and "sales" not in normalized:
            normalized["sales"] = df
        elif dtype == "marketing" and "marketing" not in normalized:
            normalized["marketing"] = df
        elif dtype == "customers" and "customers" not in normalized:
            normalized["customers"] = df
        elif dtype == "tech" and "github" not in normalized:
            normalized["github"] = df

    for name, df in datasets.items():
        lowered = name.lower()
        if lowered == "github" and "github" not in normalized:
            normalized["github"] = df
    return normalized


def _generic_dataset_reports(datasets, matched_keys):
    reports = {}
    client = LLMClient(request_timeout=45)
    parser = ResponseParser()
    for name, df in datasets.items():
        detector = DatasetDetector(df, name)
        if detector.detected_type in {"sales", "marketing", "customers", "tech"}:
            if detector.detected_type == "tech":
                mapped_key = "github"
            else:
                mapped_key = detector.detected_type
            if mapped_key in matched_keys:
                continue

        prompt = PromptTemplates.dynamic_analysis(
            detected_type=detector.detected_type,
            context=detector.get_analysis_context(),
        )
        report = client.generate(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_ANALYST_STRICT,
            temperature=0.2,
        )
        report = parser.clean_response(report)
        reports[name] = report
    return reports


def _render_result_cards(result):
    reports = result.get("reports", {})
    strategic = result.get("strategic_report") or ""
    recommendations = result.get("recommendations") or ""
    status = result.get("agent_status", {})
    metadata = result.get("metadata", {})

    generic_reports = result.get("generic_reports", {})

    return html.Div(
        [
            dbc.Alert(
                f"Run finished in {metadata.get('total_execution_time', 0)}s using model {metadata.get('llm_model', 'unknown')}",
                color="success",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Pipeline Status"),
                        html.Ul(
                            [
                                html.Li(f"{agent}: {agent_status}")
                                for agent, agent_status in status.items()
                            ]
                        ),
                    ]
                ),
                className="mb-3",
            ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        dcc.Markdown(text), title=f"{name.title()} Report"
                    )
                    for name, text in reports.items()
                ]
                + [
                    dbc.AccordionItem(
                        dcc.Markdown(text), title=f"{name.title()} Generic Report"
                    )
                    for name, text in generic_reports.items()
                ]
                + [
                    dbc.AccordionItem(
                        dcc.Markdown(strategic), title="Strategic Synthesis"
                    ),
                    dbc.AccordionItem(
                        dcc.Markdown(recommendations), title="Recommendations"
                    ),
                ],
                start_collapsed=True,
            ),
        ]
    )


def render():
    datasets = state.get_datasets()
    if not datasets:
        return html.Div(
            [
                html.H2("🔗 Multi-Agent Analysis"),
                dbc.Alert(
                    "No datasets loaded yet. Go to Data Upload first.", color="warning"
                ),
            ],
            className="p-3",
        )

    compatible = _normalize_for_agents(datasets)
    result = state.get_last_multi_agent_result()

    return html.Div(
        [
            html.H2("🔗 Multi-Agent Analysis"),
            html.P(
                "Run the specialist analyst agents, then synthesize the reports through the strategy agent."
            ),
            dbc.Alert(
                "Compatible datasets detected: "
                + (", ".join(sorted(compatible.keys())) if compatible else "none"),
                color="dark",
            ),
            dbc.Button(
                "Run Full Analysis",
                id="multi-agent-run-btn",
                color="primary",
                className="mb-3",
            ),
            html.Div(id="multi-agent-status", className="mb-3"),
            html.Div(
                id="multi-agent-output",
                children=_render_result_cards(result) if result else html.Div(),
            ),
        ],
        className="p-3",
    )


def register_callbacks(app):
    @app.callback(
        [
            Output("multi-agent-status", "children"),
            Output("multi-agent-output", "children"),
        ],
        Input("multi-agent-run-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_multi_agent(n_clicks):
        try:
            datasets = _normalize_for_agents(state.get_datasets())
            if not datasets:
                return dbc.Alert(
                    "No compatible datasets detected for the specialist agents.",
                    color="danger",
                ), html.Div()

            orchestrator = AgentOrchestrator(LLMClient(request_timeout=45))
            result = orchestrator.run_full_analysis(datasets, include_ml=True)
            generic_reports = _generic_dataset_reports(
                state.get_datasets(), set(datasets.keys())
            )
            if generic_reports:
                result["generic_reports"] = generic_reports
            state.set_last_multi_agent_result(result)
            state.set_last_recommendations(result.get("recommendations") or "")
            return dbc.Alert(
                "Multi-agent analysis completed.", color="success"
            ), _render_result_cards(result)
        except Exception as e:
            return dbc.Alert(f"Multi-agent analysis failed: {e}", color="danger"), html.Div()
