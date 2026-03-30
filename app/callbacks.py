"""Global callbacks for page routing and shared status."""

from dash import html
from dash.dependencies import Input, Output

from app.pages import data_upload, data_overview, visualizations
from app.pages import ml_insights, ai_insights, multi_agent, recommendations

# Page renderers
PAGE_RENDERERS = {
    "data_upload": data_upload.render,
    "data_overview": data_overview.render,
    "visualizations": visualizations.render,
    "ml_insights": ml_insights.render,
    "ai_insights": ai_insights.render,
    "multi_agent": multi_agent.render,
    "recommendations": recommendations.render,
}


def register_callbacks(app):
    """Register all global callbacks."""

    # Page navigation callback
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname):
        """Route to the appropriate page based on URL."""
        if not pathname or pathname == "/":
            pathname = "/data_upload"

        # Remove leading slash
        page_id = pathname.lstrip("/")

        # Get the page renderer
        renderer = PAGE_RENDERERS.get(page_id)

        if renderer:
            try:
                return renderer()
            except Exception as e:
                return html.Div(
                    [
                        html.H4(f"Error loading page: {page_id}"),
                        html.P(str(e)),
                        html.A("Go to Data Upload", href="/data_upload"),
                    ],
                    className="p-4",
                )
        else:
            return html.Div(
                [
                    html.H4("Page not found"),
                    html.P(f"Path: {pathname}"),
                    html.A("Go to Data Upload", href="/data_upload"),
                ],
                className="p-4",
            )

    # Update navigation active state
    @app.callback(
        [
            Output(f"nav-{page['id']}", "active")
            for page in [
                {"id": "data_upload"},
                {"id": "data_overview"},
                {"id": "visualizations"},
                {"id": "ml_insights"},
                {"id": "ai_insights"},
                {"id": "multi_agent"},
                {"id": "recommendations"},
            ]
        ],
        Input("url", "pathname"),
    )
    def update_nav_active(pathname):
        """Update which nav link is active based on current URL."""
        if not pathname or pathname == "/":
            pathname = "/data_upload"

        page_id = pathname.lstrip("/")

        return [
            True if page_id == page["id"] else False
            for page in [
                {"id": "data_upload"},
                {"id": "data_overview"},
                {"id": "visualizations"},
                {"id": "ml_insights"},
                {"id": "ai_insights"},
                {"id": "multi_agent"},
                {"id": "recommendations"},
            ]
        ]

    # System status callback (placeholder for LLM status)
    @app.callback(Output("system-status", "children"), Input("url", "pathname"))
    def update_system_status(pathname):
        """Display system/LLM connection status."""
        try:
            from llm.llm_client import LLMClient

            client = LLMClient()
            status = client.check_connection()

            if status.get("connected"):
                return html.Div(
                    [
                        html.Span("🟢", className="me-2"),
                        html.Span(f"LLM: {client.model}"),
                        html.Br(),
                        html.Span(
                            "Model loaded and ready",
                            style={"font-size": "0.8rem", "color": "#8B949E"},
                        ),
                    ]
                )
            else:
                return html.Div(
                    [
                        html.Span("🟡", className="me-2"),
                        html.Span("Ollama not connected"),
                        html.Br(),
                        html.Span(
                            "Using fallback mode",
                            style={"font-size": "0.8rem", "color": "#8B949E"},
                        ),
                    ]
                )
        except Exception:
            return html.Div(
                [
                    html.Span("🟡", className="me-2"),
                    html.Span("Fallback mode active"),
                    html.Br(),
                    html.Span(
                        "LLM not configured",
                        style={"font-size": "0.8rem", "color": "#8B949E"},
                    ),
                ]
            )

    data_upload.register_callbacks(app)
    ai_insights.register_callbacks(app)
    multi_agent.register_callbacks(app)
    recommendations.register_callbacks(app)

    return app
