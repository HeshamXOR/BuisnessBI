"""Dash application initialization."""

import os
import sys

from dash import Dash
import dash_bootstrap_components as dbc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
)

# Custom dark theme CSS
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>AI Decision Intelligence Platform</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

from app.layout import build_layout

app.layout = build_layout()

from app.callbacks import register_callbacks

register_callbacks(app)


def run_server(debug=True, port=8050):
    """Run the Dash development server."""
    app.run(debug=debug, port=port, use_reloader=False)


if __name__ == "__main__":
    run_server(debug=True)
