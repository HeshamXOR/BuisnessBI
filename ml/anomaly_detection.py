"""
Anomaly Detection Module
========================
IsolationForest-based anomaly detection for identifying unusual patterns.
Results are passed to the LLM agent for interpretation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple, List


def detect_anomalies(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    contamination: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect anomalies in the dataset using IsolationForest.

    Args:
        df: Input DataFrame.
        features: List of numeric columns to analyze. If None, auto-selects all numeric.
        contamination: Expected proportion of anomalies (0.0 - 0.5).
        random_state: Random seed.

    Returns:
        Tuple of (DataFrame with 'is_anomaly' column, metadata dict).
    """
    # Auto-select numeric features
    if features is None:
        features = list(df.select_dtypes(include="number").columns)

    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 1:
        raise ValueError("Need at least 1 numeric feature for anomaly detection.")

    # Prepare data
    X = df[available_features].copy()
    X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run IsolationForest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1
    )
    predictions = iso_forest.fit_predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)

    # Mark anomalies (-1 = anomaly, 1 = normal)
    result_df = df.copy()
    result_df["is_anomaly"] = predictions == -1
    result_df["anomaly_score"] = np.round(-scores, 4)  # Higher = more anomalous

    n_anomalies = int(result_df["is_anomaly"].sum())

    metadata = {
        "features_used": available_features,
        "contamination": contamination,
        "total_records": len(df),
        "anomalies_found": n_anomalies,
        "anomaly_rate": round(n_anomalies / len(df) * 100, 2)
    }

    return result_df, metadata


def get_anomaly_report(df: pd.DataFrame, features: Optional[List[str]] = None) -> str:
    """
    Generate a human-readable anomaly report for LLM context.

    Args:
        df: DataFrame with 'is_anomaly' and 'anomaly_score' columns.
        features: Features to highlight in the report.

    Returns:
        Formatted string describing detected anomalies.
    """
    if "is_anomaly" not in df.columns:
        return "No anomaly detection data available."

    anomalies = df[df["is_anomaly"] == True]
    normal = df[df["is_anomaly"] == False]

    if features is None:
        features = list(df.select_dtypes(include="number").columns)
        features = [f for f in features if f not in ["is_anomaly", "anomaly_score"]]

    lines = [
        f"Anomaly Detection Report:",
        f"- Total records: {len(df)}",
        f"- Anomalies detected: {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}%)",
        f"- Normal records: {len(normal)}",
        "",
        "### Anomaly vs Normal Comparison:"
    ]

    for feature in features[:8]:  # Limit to top 8 features
        if feature in df.columns and df[feature].dtype in ["float64", "int64", "float32", "int32"]:
            anom_mean = round(anomalies[feature].mean(), 2) if len(anomalies) > 0 else 0
            norm_mean = round(normal[feature].mean(), 2)
            diff_pct = round(
                (anom_mean - norm_mean) / max(abs(norm_mean), 0.01) * 100, 1
            )
            direction = "↑" if diff_pct > 0 else "↓"
            lines.append(
                f"  - {feature}: anomaly_avg={anom_mean} vs normal_avg={norm_mean} "
                f"({direction}{abs(diff_pct)}%)"
            )

    # Top 5 most anomalous records
    if len(anomalies) > 0 and "anomaly_score" in df.columns:
        lines.append("\n### Top 5 Most Anomalous Records:")
        top_anomalies = anomalies.nlargest(5, "anomaly_score")
        for _, row in top_anomalies.iterrows():
            key_vals = []
            for f in features[:4]:
                if f in row.index:
                    key_vals.append(f"{f}={row[f]}")
            lines.append(f"  - Score: {row['anomaly_score']:.3f} | {', '.join(key_vals)}")

    return "\n".join(lines)
