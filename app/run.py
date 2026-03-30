"""
Run Dash Application
=====================
Entry point to start the Dash development server.

Usage:
    python app/run.py

Then open http://localhost:8050 in your browser.
"""

import sys
import os

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import app, run_server

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 AI Decision Intelligence Platform")
    print("=" * 60)
    print("Starting Dash server...")
    print("Open http://localhost:8050 in your browser")
    print("=" * 60)

    run_server(debug=True, port=8050)
