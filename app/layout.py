"""Main layout for the Dash application."""

from dash import dcc, html
import dash_bootstrap_components as dbc

# Navigation items
NAV_PAGES = [
    {"id": "data_upload", "label": "📤 Data Upload / Load", "icon": "upload"},
    {"id": "data_overview", "label": "📋 Data Overview", "icon": "table"},
    {"id": "visualizations", "label": "📊 Visualizations", "icon": "chart"},
    {"id": "ml_insights", "label": "🧠 ML Insights", "icon": "brain"},
    {"id": "ai_insights", "label": "🤖 AI Insights (LLM)", "icon": "robot"},
    {"id": "multi_agent", "label": "🔗 Multi-Agent Analysis", "icon": "network"},
    {"id": "recommendations", "label": "🎯 Recommendations", "icon": "target"},
]


def build_layout():
    """Create the main app shell with sidebar and content area."""

    # Create navigation links
    nav_items = []
    for page in NAV_PAGES:
        nav_items.append(
            dbc.NavLink(
                page["label"],
                id=f"nav-{page['id']}",
                href=f"/{page['id']}",
                active="exact",
                className="nav-link",
            )
        )

    # Main layout
    layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="chart-errors", data=[]),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H1(
                                                        "🧠",
                                                        className="mb-0",
                                                        style={"font-size": "2rem"},
                                                    ),
                                                    html.H3(
                                                        "AI Decision Intelligence",
                                                        className="gradient-text mb-1",
                                                    ),
                                                    html.P(
                                                        "Multi-Agent Analysis Platform",
                                                        className="text-muted",
                                                        style={
                                                            "font-size": "0.85rem",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "text-align": "center",
                                                    "padding": "20px 0",
                                                },
                                            ),
                                            html.Hr(style={"border-color": "var(--border)"}),
                                            html.Div(nav_items),
                                            html.Hr(style={"border-color": "var(--border)"}),
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "⚡ System Status",
                                                        className="mb-2",
                                                    ),
                                                    html.Div(id="system-status"),
                                                ],
                                                className="p-3",
                                            ),
                                            html.Hr(style={"border-color": "#2D3A52"}),
                                            html.Div(
                                                [
                                                    html.Small(
                                                        "Built with Dash + Plotly",
                                                        className="text-muted",
                                                    ),
                                                    html.Br(),
                                                    html.Small(
                                                        "Powered by Ollama / Unsloth",
                                                        className="text-muted",
                                                    ),
                                                ],
                                                className="p-3",
                                            ),
                                        ],
                                        className="sidebar",
                                        style={"min-height": "100vh", "padding": "0"},
                                    )
                                ],
                                width=2,
                                className="p-0",
                            ),
                            dbc.Col(
                                [html.Div(id="page-content", className="p-4")], width=10
                            ),
                        ],
                        className="g-0",
                    )
                ],
                fluid=True,
            ),
        ]
    )

    return layout


def create_page_content(title, content):
    """Wrapper to create a standard page header and content."""
    return html.Div([html.H2(title, className="mb-4"), html.Hr(), content])


def create_metric_card(value, label, icon=None):
    """Create a metric display card."""
    return html.Div(
        [
            html.Div(icon or "", className="mb-2", style={"font-size": "1.5rem"})
            if icon
            else None,
            html.Div(value, className="metric-value"),
            html.Div(label, className="metric-label"),
        ],
        className="metric-card p-3",
    )


def create_chart_container(figure, title=None):
    """Wrap a Plotly figure in a card container."""
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5(title, className="card-title") if title else None,
                    dcc.Graph(
                        figure=figure,
                        config={"displayModeBar": True, "responsive": True},
                    ),
                ]
            )
        ],
        className="mb-3",
    )


def create_data_table(df, max_rows=10):
    """Create an HTML table from a DataFrame."""
    if df is None or df.empty:
        return html.P("No data available", className="text-muted")

    # Limit rows
    display_df = df.head(max_rows)

    # Build table
    return dbc.Table(
        [
            html.Thead([html.Tr([html.Th(col) for col in display_df.columns])]),
            html.Tbody(
                [
                    html.Tr([html.Td(str(val)) for val in row])
                    for row in display_df.values
                ]
            ),
        ],
        striped=True,
        bordered=True,
        hover=True,
        dark=True,
    )
