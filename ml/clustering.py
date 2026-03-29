"""
Customer Clustering Module
===========================
K-Means clustering to identify customer segments.
Results are passed to the LLM agent for interpretation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Optional, Tuple


def perform_clustering(
    df: pd.DataFrame,
    features: Optional[list] = None,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform K-Means clustering on customer data.

    Args:
        df: Customer DataFrame.
        features: List of numeric columns to use. If None, auto-selects.
        n_clusters: Number of clusters.
        random_state: Random seed.

    Returns:
        Tuple of (DataFrame with 'cluster' column, metadata dict).
    """
    # Default features for customer clustering
    if features is None:
        features = [
            "lifetime_value", "purchase_frequency",
            "satisfaction_score", "churn_risk",
            "engagement_score", "support_tickets"
        ]

    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        raise ValueError(
            f"Need at least 2 numeric features for clustering. "
            f"Available: {available_features}"
        )

    # Prepare data
    X = df[available_features].copy()
    X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k if not specified (elbow method via silhouette)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Compute silhouette score
    sil_score = silhouette_score(X_scaled, clusters) if n_clusters > 1 else 0.0

    # Add cluster labels to dataframe
    result_df = df.copy()
    result_df["cluster"] = clusters

    metadata = {
        "n_clusters": n_clusters,
        "features_used": available_features,
        "silhouette_score": round(sil_score, 3),
        "cluster_sizes": pd.Series(clusters).value_counts().sort_index().to_dict(),
        "inertia": round(kmeans.inertia_, 2)
    }

    return result_df, metadata


def get_cluster_summary(df: pd.DataFrame, features: Optional[list] = None) -> str:
    """
    Generate a human-readable cluster summary for LLM context.

    Args:
        df: DataFrame with 'cluster' column from perform_clustering().
        features: Features to summarize per cluster.

    Returns:
        Formatted string describing each cluster's characteristics.
    """
    if "cluster" not in df.columns:
        return "No clustering data available."

    if features is None:
        features = [
            "lifetime_value", "purchase_frequency",
            "satisfaction_score", "churn_risk",
            "engagement_score", "support_tickets"
        ]

    available_features = [f for f in features if f in df.columns]

    lines = [f"Customer Clustering Analysis ({df['cluster'].nunique()} clusters):\n"]

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_data = df[df["cluster"] == cluster_id]
        size = len(cluster_data)
        pct = round(size / len(df) * 100, 1)

        lines.append(f"### Cluster {cluster_id} ({size} customers, {pct}%)")

        for feature in available_features:
            if feature in cluster_data.columns:
                mean_val = round(cluster_data[feature].mean(), 2)
                median_val = round(cluster_data[feature].median(), 2)
                lines.append(f"  - {feature}: mean={mean_val}, median={median_val}")

        # Add segment distribution if available
        if "segment" in cluster_data.columns:
            top_segment = cluster_data["segment"].mode().iloc[0]
            lines.append(f"  - Dominant segment: {top_segment}")

        lines.append("")

    return "\n".join(lines)
