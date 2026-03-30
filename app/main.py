"""Compatibility wrapper for the new Dash app."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import run_server


if __name__ == "__main__":
    run_server(debug=True)
