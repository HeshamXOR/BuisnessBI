"""
Customer Analyst Agent
=======================
Analyzes customer data for segmentation, churn risk, and retention strategies.
Integrates ML clustering for deeper segment analysis.
"""

import pandas as pd
from typing import Optional

from agents.base_agent import BaseAgent
from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from ml.clustering import perform_clustering, get_cluster_summary


class CustomerAnalystAgent(BaseAgent):
    """
    Customer Analyst Agent — Analyzes customer segments, churn risk,
    and lifetime value. Uses ML clustering to enhance insights.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(
            name="Customer Analyst",
            role="Analyzes customer segmentation, churn risk, and retention strategies",
            llm_client=llm_client,
            dataset_type="customers"
        )
        self._cluster_metadata = None

    def analyze(self, df: pd.DataFrame, **kwargs) -> str:
        """
        Run customer analysis with optional ML clustering.

        Args:
            df: Customer DataFrame.
            **kwargs:
                include_clustering (bool): Run K-Means clustering (default True).
                n_clusters (int): Number of clusters (default 4).

        Returns:
            LLM-generated customer analysis report.
        """
        kpis, kpis_str, data_summary = self._prepare_context(df)

        # ML Clustering integration
        cluster_info = ""
        if kwargs.get("include_clustering", True):
            try:
                n_clusters = kwargs.get("n_clusters", 4)
                clustered_df, cluster_meta = perform_clustering(
                    df, n_clusters=n_clusters
                )
                cluster_info = get_cluster_summary(clustered_df)
                self._cluster_metadata = cluster_meta
                self._metadata["clustering"] = cluster_meta
                self._metadata["ml_clustering_included"] = True
            except Exception as e:
                self._metadata["ml_clustering_error"] = str(e)

        prompt = self.get_prompt(kpis_str, data_summary, cluster_info=cluster_info)
        return self._generate_report(prompt)

    def get_prompt(self, kpis_str: str, data_summary: str, **kwargs) -> str:
        """Build the customer analysis prompt with optional clustering context."""
        cluster_info = kwargs.get("cluster_info", "")
        return PromptTemplates.customer_analysis(kpis_str, data_summary, cluster_info)

    def get_cluster_metadata(self) -> Optional[dict]:
        """Get clustering metadata if clustering was performed."""
        return self._cluster_metadata
