"""
Helper Utilities
================
Formatting, text, and general-purpose utility functions.
"""

from datetime import datetime
from typing import Any


def format_currency(value: float, currency: str = "$") -> str:
    """Format a number as currency."""
    if abs(value) >= 1_000_000:
        return f"{currency}{value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{currency}{value / 1_000:.1f}K"
    else:
        return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a number as a percentage string."""
    return f"{value:.{decimals}f}%"


def format_number(value: float) -> str:
    """Format a large number with K/M/B suffixes."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:,.0f}"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def dataframe_to_summary_string(df, max_rows: int = 10) -> str:
    """
    Convert a DataFrame to a concise summary string for LLM context.

    Args:
        df: pandas DataFrame.
        max_rows: Maximum rows to include in the sample.

    Returns:
        Formatted string with shape, columns, and sample data.
    """
    lines = [
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns.tolist())}",
        "",
        "Sample Data:",
        df.head(max_rows).to_string(index=False),
        "",
        "Numeric Summary:",
        df.describe().round(2).to_string()
    ]
    return "\n".join(lines)


def kpis_to_string(kpis: dict) -> str:
    """Convert a KPIs dictionary to a formatted string for LLM prompts."""
    lines = []
    for key, value in kpis.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            if "pct" in key or "rate" in key or "margin" in key or "risk" in key:
                lines.append(f"- {label}: {value:.1f}%")
            elif "revenue" in key or "spend" in key or "value" in key or "cost" in key:
                lines.append(f"- {label}: {format_currency(value)}")
            else:
                lines.append(f"- {label}: {value:.2f}")
        else:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def get_timestamp() -> str:
    """Get current timestamp as formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator
