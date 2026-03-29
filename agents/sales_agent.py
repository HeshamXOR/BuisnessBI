"""
Sales Analyst Agent
====================
Analyzes sales data to generate revenue insights, product performance,
regional analysis, and growth recommendations.
"""

import pandas as pd
from typing import Optional

from agents.base_agent import BaseAgent
from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from ml.trend_analysis import compute_trend_signal, get_trend_report


class SalesAnalystAgent(BaseAgent):
    """
    Sales Analyst Agent — Analyzes sales data for revenue trends,
    product performance, and regional insights.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(
            name="Sales Analyst",
            role="Analyzes sales performance, revenue trends, and product strategy",
            llm_client=llm_client,
            dataset_type="sales"
        )

    def analyze(self, df: pd.DataFrame, **kwargs) -> str:
        """
        Run sales analysis on the provided data.

        Args:
            df: Sales DataFrame with columns like revenue, product, region, etc.
            **kwargs: Optional 'include_trends' (bool) to add ML trend signals.

        Returns:
            LLM-generated sales analysis report.
        """
        # Prepare base context
        kpis, kpis_str, data_summary = self._prepare_context(df)

        # Optionally add ML trend signals
        trend_context = ""
        if kwargs.get("include_trends", True) and "date" in df.columns:
            try:
                trend_report = get_trend_report(
                    df, "date",
                    {"Revenue": "revenue", "Units Sold": "units_sold"},
                    freq="M"
                )
                trend_context = f"\n## ML Trend Signals\n{trend_report}"
                self._metadata["ml_trends_included"] = True
            except Exception as e:
                trend_context = ""
                self._metadata["ml_trends_error"] = str(e)

        # Build prompt
        prompt = self.get_prompt(kpis_str, data_summary + trend_context)

        # Generate report
        return self._generate_report(prompt)

    def get_prompt(self, kpis_str: str, data_summary: str, **kwargs) -> str:
        """Build the sales analysis prompt."""
        return PromptTemplates.sales_analysis(kpis_str, data_summary)
