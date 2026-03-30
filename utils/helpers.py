"""
Helper Utilities
================
Formatting, text, and general-purpose utility functions.
"""

from datetime import datetime
from typing import Any

import pandas as pd


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
    return text[: max_length - 3] + "..."


def dataframe_to_summary_string(df, max_rows: int = 8) -> str:
    """
    Convert a DataFrame to a concise summary string for LLM context.

    Args:
        df: pandas DataFrame.
        max_rows: Maximum rows to include in the sample.

    Returns:
        Formatted string with shape, columns, and sample data.
    """
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(
        df.select_dtypes(include=["object", "category", "bool"]).columns
    )
    missing_total = int(df.isna().sum().sum())
    total_cells = max(int(df.shape[0] * max(df.shape[1], 1)), 1)
    missing_pct = round(missing_total / total_cells * 100, 2)

    lines = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:12]) if numeric_cols else 'None'}",
        f"Categorical/date-like columns ({len(categorical_cols)}): {', '.join(categorical_cols[:12]) if categorical_cols else 'None'}",
        f"Missing cells: {missing_total} ({missing_pct}%)",
        "",
    ]

    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().round(2)
        lines.extend(["Top Numeric Summary:", numeric_summary.to_string(), ""])

    if categorical_cols:
        lines.append("Top Categorical Distributions:")
        for col in categorical_cols[:5]:
            counts = df[col].astype(str).value_counts(dropna=False).head(5)
            rendered = ", ".join(f"{idx}={val}" for idx, val in counts.items())
            lines.append(f"- {col}: {rendered}")
        lines.append("")

    lines.extend(["Sample Rows:", df.head(max_rows).to_string(index=False)])
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


def compact_dataframe_profile(df: pd.DataFrame, top_n: int = 5) -> str:
    """Create a compact, business-friendly context block for LLM prompts."""
    profile_lines = [
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns)}",
        f"Duplicate rows: {int(df.duplicated().sum()):,}",
        f"Missing cells: {int(df.isna().sum().sum()):,}",
    ]

    numeric_cols = list(df.select_dtypes(include="number").columns)
    if numeric_cols:
        ranked = []
        for col in numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            ranked.append((col, float(vals.std()) if vals.nunique() > 1 else 0.0))
        ranked.sort(key=lambda item: item[1], reverse=True)
        if ranked:
            profile_lines.append(
                "Top numeric columns by variance: "
                + ", ".join(col for col, _ in ranked[:top_n])
            )

    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    for col in cat_cols[:top_n]:
        counts = df[col].astype(str).value_counts(dropna=False).head(3)
        preview = ", ".join(f"{idx}={val}" for idx, val in counts.items())
        profile_lines.append(f"{col} top values: {preview}")

    return "\n".join(profile_lines)


def get_timestamp() -> str:
    """Get current timestamp as formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator
