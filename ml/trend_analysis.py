"""
Trend Analysis Module
=====================
Linear regression-based trend signals and simple forecasting.
Results are passed to the LLM agent for interpretation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Optional, Tuple


def compute_trend_signal(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "M"
) -> Dict[str, Any]:
    """
    Compute trend signal using linear regression on time series data.

    Args:
        df: Input DataFrame with date and value columns.
        date_col: Name of the date column.
        value_col: Name of the value column.
        freq: Resampling frequency ('D', 'W', 'M', 'Q').

    Returns:
        Dictionary with trend metrics.
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Resample to specified frequency
    ts = df_copy.set_index(date_col)[value_col].resample(freq).sum().reset_index()
    ts.columns = ["date", "value"]

    if len(ts) < 3:
        return {
            "trend_direction": "insufficient_data",
            "slope": 0,
            "r_squared": 0,
            "data_points": len(ts)
        }

    # Create numeric time feature
    ts["time_index"] = np.arange(len(ts))

    # Fit linear regression
    X = ts[["time_index"]].values
    y = ts["value"].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    r_squared = model.score(X, y)
    predicted = model.predict(X)

    # Determine trend direction and strength
    if abs(r_squared) < 0.1:
        direction = "flat"
    elif slope > 0:
        direction = "upward"
    else:
        direction = "downward"

    # Compute growth rate
    if predicted[0] != 0:
        total_growth = (predicted[-1] - predicted[0]) / abs(predicted[0]) * 100
    else:
        total_growth = 0

    return {
        "trend_direction": direction,
        "slope": round(slope, 2),
        "r_squared": round(r_squared, 4),
        "total_growth_pct": round(total_growth, 2),
        "data_points": len(ts),
        "start_value": round(ts["value"].iloc[0], 2),
        "end_value": round(ts["value"].iloc[-1], 2),
        "min_value": round(ts["value"].min(), 2),
        "max_value": round(ts["value"].max(), 2),
        "avg_value": round(ts["value"].mean(), 2),
        "volatility": round(ts["value"].std() / max(ts["value"].mean(), 1) * 100, 2)
    }


def forecast_simple(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 3,
    freq: str = "M"
) -> pd.DataFrame:
    """
    Generate simple linear forecast for future periods.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.
        value_col: Name of the value column.
        periods: Number of future periods to forecast.
        freq: Resampling frequency.

    Returns:
        DataFrame with date, actual (historical), and forecast columns.
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Resample
    ts = df_copy.set_index(date_col)[value_col].resample(freq).sum().reset_index()
    ts.columns = ["date", "value"]

    # Create time features
    ts["time_index"] = np.arange(len(ts))

    # Fit model
    X = ts[["time_index"]].values
    y = ts["value"].values

    model = LinearRegression()
    model.fit(X, y)

    # Generate forecast dates
    last_date = ts["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1 if freq == "M" else 1),
        periods=periods,
        freq=freq
    )

    future_indices = np.arange(len(ts), len(ts) + periods).reshape(-1, 1)
    forecast_values = model.predict(future_indices)

    # Combine historical and forecast
    historical = ts[["date", "value"]].copy()
    historical["type"] = "actual"
    historical.columns = ["date", "value", "type"]

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "value": np.round(forecast_values, 2),
        "type": "forecast"
    })

    result = pd.concat([historical, forecast_df], ignore_index=True)
    return result


def get_trend_report(
    df: pd.DataFrame,
    date_col: str,
    metrics: Dict[str, str],
    freq: str = "M"
) -> str:
    """
    Generate a comprehensive trend report for multiple metrics.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.
        metrics: Dict mapping metric names to column names.
        freq: Resampling frequency.

    Returns:
        Formatted string with trend analysis for LLM context.
    """
    lines = ["Trend Analysis Report:\n"]

    for metric_name, col_name in metrics.items():
        if col_name not in df.columns:
            continue

        signal = compute_trend_signal(df, date_col, col_name, freq)

        emoji = {"upward": "📈", "downward": "📉", "flat": "➡️"}.get(
            signal["trend_direction"], "❓"
        )

        lines.append(f"### {metric_name} {emoji}")
        lines.append(f"  - Direction: {signal['trend_direction']}")
        lines.append(f"  - Total Growth: {signal['total_growth_pct']}%")
        lines.append(f"  - R²: {signal['r_squared']} (fit quality)")
        lines.append(f"  - Range: {signal['min_value']} → {signal['max_value']}")
        lines.append(f"  - Volatility: {signal['volatility']}%")
        lines.append("")

    return "\n".join(lines)
