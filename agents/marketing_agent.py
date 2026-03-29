"""
Marketing Analyst Agent
========================
Analyzes marketing campaign data for ROI, channel effectiveness,
and budget optimization recommendations.
"""

import pandas as pd
from typing import Optional

from agents.base_agent import BaseAgent
from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from ml.trend_analysis import get_trend_report


class MarketingAnalystAgent(BaseAgent):
    """
    Marketing Analyst Agent — Analyzes campaigns, channels, and ROI
    to optimize marketing spend.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(
            name="Marketing Analyst",
            role="Analyzes campaign performance, channel ROI, and budget optimization",
            llm_client=llm_client,
            dataset_type="marketing"
        )

    def analyze(self, df: pd.DataFrame, **kwargs) -> str:
        """
        Run marketing analysis on campaign data.

        Args:
            df: Marketing DataFrame with campaign, channel, spend, ROI columns.
            **kwargs: Optional 'include_trends' (bool).

        Returns:
            LLM-generated marketing analysis report.
        """
        kpis, kpis_str, data_summary = self._prepare_context(df)

        # Optionally add ML trend signals
        trend_context = ""
        if kwargs.get("include_trends", True) and "date" in df.columns:
            try:
                trend_report = get_trend_report(
                    df, "date",
                    {
                        "Marketing Spend": "spend",
                        "Conversions": "conversions",
                        "Revenue Generated": "revenue_generated"
                    },
                    freq="M"
                )
                trend_context = f"\n## ML Trend Signals\n{trend_report}"
                self._metadata["ml_trends_included"] = True
            except Exception as e:
                self._metadata["ml_trends_error"] = str(e)

        prompt = self.get_prompt(kpis_str, data_summary + trend_context)
        return self._generate_report(prompt)

    def get_prompt(self, kpis_str: str, data_summary: str, **kwargs) -> str:
        """Build the marketing analysis prompt."""
        return PromptTemplates.marketing_analysis(kpis_str, data_summary)
