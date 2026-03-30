"""
Base Agent Module
=================
Abstract base class for all analyst agents in the multi-agent system.
Each agent analyzes a specific domain, uses LLM for reasoning, and
produces structured reports.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import pandas as pd

from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from llm.response_parser import ResponseParser
from utils.analysis import compute_kpis, get_summary_statistics
from utils.helpers import (
    compact_dataframe_profile,
    dataframe_to_summary_string,
    kpis_to_string,
)


class BaseAgent(ABC):
    """
    Abstract base class for AI analyst agents.

    Each agent:
    1. Receives structured data
    2. Computes domain-specific KPIs and statistics
    3. Uses LLM to generate insights and recommendations
    4. Produces a structured report
    """

    def __init__(self, name: str, role: str, llm_client: LLMClient, dataset_type: str):
        """
        Initialize the base agent.

        Args:
            name: Human-readable agent name.
            role: Description of the agent's role.
            llm_client: LLM client for generating insights.
            dataset_type: Type of dataset this agent analyzes.
        """
        self.name = name
        self.role = role
        self.llm_client = llm_client
        self.dataset_type = dataset_type
        self.parser = ResponseParser()

        # State
        self._report: Optional[str] = None
        self._kpis: Optional[Dict[str, Any]] = None
        self._metadata: Dict[str, Any] = {}
        self._execution_time: float = 0

    @abstractmethod
    def analyze(self, df: pd.DataFrame, **kwargs) -> str:
        """
        Run the agent's analysis on the provided data.

        Args:
            df: Input DataFrame for this agent's domain.

        Returns:
            Generated analysis report as a string.
        """
        pass

    @abstractmethod
    def get_prompt(self, kpis_str: str, data_summary: str, **kwargs) -> str:
        """
        Construct the analysis prompt for the LLM.

        Args:
            kpis_str: Formatted KPI string.
            data_summary: Data summary string.

        Returns:
            Formatted prompt string.
        """
        pass

    def _prepare_context(self, df: pd.DataFrame) -> tuple:
        """
        Prepare KPIs and data summary for LLM context.

        Returns:
            Tuple of (kpis_dict, kpis_string, data_summary_string).
        """
        kpis = compute_kpis(df, self.dataset_type)
        self._kpis = kpis

        kpis_str = kpis_to_string(kpis)
        data_summary = dataframe_to_summary_string(df, max_rows=8)
        compact_profile = compact_dataframe_profile(df)
        data_summary = f"{compact_profile}\n\n{data_summary}"

        return kpis, kpis_str, data_summary

    def _generate_report(self, prompt: str) -> str:
        """
        Generate a report using the LLM.

        Args:
            prompt: The formatted analysis prompt.

        Returns:
            LLM-generated report text.
        """
        start_time = time.time()

        report = self.llm_client.generate(
            prompt=prompt, system_prompt=PromptTemplates.SYSTEM_ANALYST, temperature=0.3
        )

        report = self.parser.clean_response(report)
        if self.parser.is_low_quality_response(report):
            repair_prompt = PromptTemplates.repair_insight(
                data_summary=prompt,
                question="Rewrite this analyst report to remove unsupported claims and improve clarity.",
                previous_output=report,
            )
            repaired = self.llm_client.generate(
                prompt=repair_prompt,
                system_prompt=PromptTemplates.SYSTEM_ANALYST_STRICT,
                temperature=0.1,
            )
            cleaned_repaired = self.parser.clean_response(repaired)
            if not self.parser.is_low_quality_response(cleaned_repaired):
                report = cleaned_repaired

        self._execution_time = round(time.time() - start_time, 2)
        self._report = report

        return report

    def report(self) -> str:
        """Get the latest generated report."""
        if self._report is None:
            return f"[{self.name}] No analysis has been run yet."
        return self._report

    def get_kpis(self) -> Optional[Dict[str, Any]]:
        """Get computed KPIs."""
        return self._kpis

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent execution metadata."""
        return {
            "agent_name": self.name,
            "role": self.role,
            "dataset_type": self.dataset_type,
            "execution_time_seconds": self._execution_time,
            "has_report": self._report is not None,
            "llm_model": self.llm_client.model,
            **self._metadata,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', type='{self.dataset_type}')>"
