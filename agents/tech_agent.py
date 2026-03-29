"""
Tech/GitHub Analyst Agent
==========================
Analyzes GitHub repository data for technology landscape insights,
quality assessment, and ecosystem trends.
Integrates ML anomaly detection to flag unusual repositories.
"""

import pandas as pd
from typing import Optional

from agents.base_agent import BaseAgent
from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from ml.anomaly_detection import detect_anomalies, get_anomaly_report


class TechAnalystAgent(BaseAgent):
    """
    Tech Analyst Agent — Analyzes GitHub/tech data for language trends,
    code quality assessment, and community health. Uses ML anomaly
    detection to identify unusual repositories.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(
            name="Tech Analyst",
            role="Analyzes technology landscape, GitHub repos, and code quality trends",
            llm_client=llm_client,
            dataset_type="github"
        )
        self._anomaly_metadata = None

    def analyze(self, df: pd.DataFrame, **kwargs) -> str:
        """
        Run tech analysis with optional anomaly detection.

        Args:
            df: GitHub repos DataFrame.
            **kwargs:
                include_anomalies (bool): Run anomaly detection (default True).
                contamination (float): Anomaly contamination rate (default 0.1).

        Returns:
            LLM-generated tech analysis report.
        """
        kpis, kpis_str, data_summary = self._prepare_context(df)

        # ML Anomaly Detection integration
        anomaly_info = ""
        if kwargs.get("include_anomalies", True):
            try:
                features = ["stars", "forks", "open_issues", "contributors",
                            "code_quality_score"]
                contamination = kwargs.get("contamination", 0.1)

                anomaly_df, anomaly_meta = detect_anomalies(
                    df, features=features, contamination=contamination
                )
                anomaly_info = get_anomaly_report(anomaly_df, features)
                self._anomaly_metadata = anomaly_meta
                self._metadata["anomaly_detection"] = anomaly_meta
                self._metadata["ml_anomalies_included"] = True
            except Exception as e:
                self._metadata["ml_anomalies_error"] = str(e)

        prompt = self.get_prompt(kpis_str, data_summary, anomaly_info=anomaly_info)
        return self._generate_report(prompt)

    def get_prompt(self, kpis_str: str, data_summary: str, **kwargs) -> str:
        """Build the tech analysis prompt with optional anomaly context."""
        anomaly_info = kwargs.get("anomaly_info", "")
        return PromptTemplates.tech_analysis(kpis_str, data_summary, anomaly_info)

    def get_anomaly_metadata(self) -> Optional[dict]:
        """Get anomaly detection metadata."""
        return self._anomaly_metadata
