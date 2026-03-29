"""
Analysis Module
===============
Pandas-based data analysis functions for KPI computation,
trend detection, and summary statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


def compute_kpis(df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
    """
    Compute key performance indicators based on dataset type.

    Args:
        df: Input DataFrame.
        dataset_type: One of 'sales', 'marketing', 'customers', 'github'.

    Returns:
        Dictionary of KPI name-value pairs.
    """
    kpis = {}

    if dataset_type == "sales":
        kpis = {
            "total_revenue": round(df["revenue"].sum(), 2),
            "avg_revenue_per_transaction": round(df["revenue"].mean(), 2),
            "total_units_sold": int(df["units_sold"].sum()),
            "avg_profit_margin": round(df["profit_margin"].mean() * 100, 1),
            "total_transactions": len(df),
            "unique_products": df["product"].nunique(),
            "top_region": df.groupby("region")["revenue"].sum().idxmax(),
            "top_category": df.groupby("category")["revenue"].sum().idxmax(),
            "avg_discount": round(df["discount_applied"].mean() * 100, 1),
            "revenue_std": round(df["revenue"].std(), 2)
        }

    elif dataset_type == "marketing":
        kpis = {
            "total_spend": round(df["spend"].sum(), 2),
            "total_revenue_generated": round(df["revenue_generated"].sum(), 2),
            "overall_roi": round(
                (df["revenue_generated"].sum() - df["spend"].sum()) /
                max(df["spend"].sum(), 1) * 100, 2
            ),
            "avg_ctr": round(df["ctr"].mean() * 100, 2),
            "avg_conversion_rate": round(df["conversion_rate"].mean() * 100, 2),
            "total_conversions": int(df["conversions"].sum()),
            "total_impressions": int(df["impressions"].sum()),
            "avg_cpc": round(df["cpc"].mean(), 2),
            "best_channel": df.groupby("channel")["roi"].mean().idxmax(),
            "best_campaign_type": df.groupby("campaign_type")["roi"].mean().idxmax()
        }

    elif dataset_type == "customers":
        kpis = {
            "total_customers": len(df),
            "avg_lifetime_value": round(df["lifetime_value"].mean(), 2),
            "median_lifetime_value": round(df["lifetime_value"].median(), 2),
            "avg_satisfaction": round(df["satisfaction_score"].mean(), 1),
            "avg_churn_risk": round(df["churn_risk"].mean() * 100, 1),
            "high_risk_customers": int((df["churn_risk"] > 0.5).sum()),
            "avg_nps": round(df["nps_score"].mean(), 1),
            "avg_engagement": round(df["engagement_score"].mean(), 1),
            "top_segment": df.groupby("segment")["lifetime_value"].sum().idxmax(),
            "avg_account_age_months": round(df["account_age_months"].mean(), 1)
        }

    elif dataset_type == "github":
        kpis = {
            "total_repos": len(df),
            "total_stars": int(df["stars"].sum()),
            "avg_stars": round(df["stars"].mean(), 1),
            "median_stars": int(df["stars"].median()),
            "total_forks": int(df["forks"].sum()),
            "avg_code_quality": round(df["code_quality_score"].mean(), 1),
            "top_language": df["language"].mode().iloc[0] if len(df) > 0 else "N/A",
            "repos_with_ci": int(df["has_ci_cd"].sum()),
            "repos_with_docs": int(df["has_documentation"].sum()),
            "avg_contributors": round(df["contributors"].mean(), 1)
        }

    return kpis


def compute_trends(df: pd.DataFrame, date_col: str, value_col: str,
                   freq: str = "M") -> pd.DataFrame:
    """
    Compute time-series trends by resampling.

    Args:
        df: Input DataFrame with a date column.
        date_col: Name of the date column.
        value_col: Name of the value column to aggregate.
        freq: Resampling frequency ('D', 'W', 'M', 'Q').

    Returns:
        DataFrame with date index and aggregated values + rolling average.
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.set_index(date_col)

    trend = df_copy[value_col].resample(freq).sum().reset_index()
    trend.columns = ["date", "total"]

    # Add rolling average
    window = min(3, len(trend))
    if window > 1:
        trend["rolling_avg"] = trend["total"].rolling(window=window, min_periods=1).mean().round(2)
    else:
        trend["rolling_avg"] = trend["total"]

    # Compute month-over-month growth
    trend["growth_pct"] = trend["total"].pct_change().fillna(0).round(4) * 100

    return trend


def get_top_items(df: pd.DataFrame, group_col: str, value_col: str,
                  n: int = 5, ascending: bool = False) -> pd.DataFrame:
    """
    Get top N items by a grouped aggregation.

    Args:
        df: Input DataFrame.
        group_col: Column to group by.
        value_col: Column to aggregate (sum).
        n: Number of top items to return.
        ascending: If True, return bottom N instead.

    Returns:
        DataFrame with group_col and aggregated value_col.
    """
    result = (
        df.groupby(group_col)[value_col]
        .sum()
        .sort_values(ascending=ascending)
        .head(n)
        .reset_index()
    )
    result.columns = [group_col, f"total_{value_col}"]
    return result


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive summary statistics for a DataFrame.

    Returns:
        Dictionary with descriptive stats for numeric and categorical columns.
    """
    stats = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "numeric_summary": {},
        "categorical_summary": {}
    }

    # Numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        stats["numeric_summary"][col] = {
            "mean": round(df[col].mean(), 2),
            "median": round(df[col].median(), 2),
            "std": round(df[col].std(), 2),
            "min": round(df[col].min(), 2),
            "max": round(df[col].max(), 2),
            "q25": round(df[col].quantile(0.25), 2),
            "q75": round(df[col].quantile(0.75), 2)
        }

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        value_counts = df[col].value_counts()
        stats["categorical_summary"][col] = {
            "unique_values": int(df[col].nunique()),
            "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else "N/A",
            "top_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "distribution": value_counts.head(5).to_dict()
        }

    return stats


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr().round(3)


def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using the IQR method.

    Returns:
        DataFrame containing only the outlier rows.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers
