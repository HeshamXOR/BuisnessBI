"""
Strategy/Decision Agent
========================
The chief strategist agent that synthesizes all analyst reports
into a unified strategic recommendation. This is the final
decision-maker in the multi-agent pipeline.
"""

import time
from typing import Dict, Any, Optional

from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from llm.response_parser import ResponseParser


class StrategyAgent:
    """
    Strategy Agent — Synthesizes insights from all analyst agents
    into executive-level strategic recommendations.
    """

    def __init__(self, llm_client: LLMClient):
        self.name = "Strategy & Decision Agent"
        self.role = (
            "Chief strategist that synthesizes all analyst insights "
            "into unified strategic recommendations"
        )
        self.llm_client = llm_client
        self.parser = ResponseParser()

        self._report: Optional[str] = None
        self._execution_time: float = 0

    def synthesize(
        self,
        sales_report: str,
        marketing_report: str,
        customer_report: str,
        tech_report: str,
    ) -> str:
        """
        Synthesize all analyst reports into a strategic recommendation.

        Args:
            sales_report: Output from SalesAnalystAgent.
            marketing_report: Output from MarketingAnalystAgent.
            customer_report: Output from CustomerAnalystAgent.
            tech_report: Output from TechAnalystAgent.

        Returns:
            Unified strategic recommendation report.
        """
        start_time = time.time()

        prompt = PromptTemplates.strategic_synthesis(
            sales_report=sales_report,
            marketing_report=marketing_report,
            customer_report=customer_report,
            tech_report=tech_report,
        )

        self._report = self.llm_client.generate(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_STRATEGIST,
            temperature=0.3,
            max_tokens=6144,  # Longer output for comprehensive strategy
        )

        self._report = self.parser.clean_response(self._report)
        if self.parser.is_low_quality_response(self._report):
            repaired = self.llm_client.generate(
                prompt=PromptTemplates.repair_insight(
                    data_summary=prompt,
                    question="Rewrite this strategic synthesis to remove weak claims and improve structure.",
                    previous_output=self._report,
                ),
                system_prompt=PromptTemplates.SYSTEM_ANALYST_STRICT,
                temperature=0.1,
                max_tokens=6144,
            )
            repaired = self.parser.clean_response(repaired)
            if not self.parser.is_low_quality_response(repaired):
                self._report = repaired

        self._execution_time = round(time.time() - start_time, 2)

        return self._report

    def generate_recommendations(self, all_insights: str) -> str:
        """
        Generate prioritized recommendations from all insights.

        Args:
            all_insights: Combined insights text from all agents.

        Returns:
            Prioritized action items with impact assessment.
        """
        prompt = PromptTemplates.recommendation_prompt(all_insights)

        raw = self.llm_client.generate(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_STRATEGIST,
            temperature=0.2,
        )
        cleaned = self.parser.clean_response(raw)
        return cleaned

    def report(self) -> str:
        """Get the latest strategic report."""
        if self._report is None:
            return "[Strategy Agent] No synthesis has been run yet."
        return self._report

    def get_metadata(self) -> Dict[str, Any]:
        """Get execution metadata."""
        return {
            "agent_name": self.name,
            "role": self.role,
            "execution_time_seconds": self._execution_time,
            "has_report": self._report is not None,
            "llm_model": self.llm_client.model,
        }

    def __repr__(self) -> str:
        return f"<StrategyAgent(name='{self.name}')>"
