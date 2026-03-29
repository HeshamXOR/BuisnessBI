"""
Agent Orchestrator
==================
Orchestrates the multi-agent analysis pipeline.
Runs all analyst agents, manages collaboration, and coordinates
the Strategy Agent for final synthesis.
"""

import time
from typing import Dict, Any, Optional

import pandas as pd

from llm.llm_client import LLMClient
from agents.sales_agent import SalesAnalystAgent
from agents.marketing_agent import MarketingAnalystAgent
from agents.customer_agent import CustomerAnalystAgent
from agents.tech_agent import TechAnalystAgent
from agents.strategy_agent import StrategyAgent


class AgentOrchestrator:
    """
    Orchestrates the full multi-agent analysis pipeline.

    Workflow:
    1. Initialize all specialist agents
    2. Run each agent's analysis on their respective datasets
    3. Collect all reports
    4. Feed reports to the Strategy Agent
    5. Generate unified recommendations
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the orchestrator with all agents.

        Args:
            llm_client: Shared LLM client for all agents.
        """
        self.llm_client = llm_client

        # Initialize all agents
        self.sales_agent = SalesAnalystAgent(llm_client)
        self.marketing_agent = MarketingAnalystAgent(llm_client)
        self.customer_agent = CustomerAnalystAgent(llm_client)
        self.tech_agent = TechAnalystAgent(llm_client)
        self.strategy_agent = StrategyAgent(llm_client)

        # State
        self._reports: Dict[str, str] = {}
        self._strategic_report: Optional[str] = None
        self._recommendations: Optional[str] = None
        self._total_time: float = 0
        self._agent_status: Dict[str, str] = {}

    def run_full_analysis(
        self,
        datasets: Dict[str, pd.DataFrame],
        include_ml: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run the complete multi-agent analysis pipeline.

        Args:
            datasets: Dict with keys 'sales', 'marketing', 'customers', 'github'.
            include_ml: Whether to include ML signals (clustering, anomalies, trends).
            progress_callback: Optional callback(agent_name, status) for progress updates.

        Returns:
            Dictionary with all reports, recommendations, and metadata.
        """
        start_time = time.time()

        def update_progress(agent_name: str, status: str):
            self._agent_status[agent_name] = status
            if progress_callback:
                progress_callback(agent_name, status)

        # ─── Phase 1: Run Analyst Agents ──────────────────────────

        # Sales Agent
        if "sales" in datasets:
            update_progress("Sales Analyst", "Analyzing sales data...")
            try:
                self._reports["sales"] = self.sales_agent.analyze(
                    datasets["sales"], include_trends=include_ml
                )
                update_progress("Sales Analyst", "✅ Complete")
            except Exception as e:
                self._reports["sales"] = f"[Error in Sales Analysis: {str(e)}]"
                update_progress("Sales Analyst", f"❌ Error: {str(e)}")

        # Marketing Agent
        if "marketing" in datasets:
            update_progress("Marketing Analyst", "Analyzing marketing data...")
            try:
                self._reports["marketing"] = self.marketing_agent.analyze(
                    datasets["marketing"], include_trends=include_ml
                )
                update_progress("Marketing Analyst", "✅ Complete")
            except Exception as e:
                self._reports["marketing"] = f"[Error in Marketing Analysis: {str(e)}]"
                update_progress("Marketing Analyst", f"❌ Error: {str(e)}")

        # Customer Agent
        if "customers" in datasets:
            update_progress("Customer Analyst", "Analyzing customer data...")
            try:
                self._reports["customers"] = self.customer_agent.analyze(
                    datasets["customers"],
                    include_clustering=include_ml,
                    n_clusters=4
                )
                update_progress("Customer Analyst", "✅ Complete")
            except Exception as e:
                self._reports["customers"] = f"[Error in Customer Analysis: {str(e)}]"
                update_progress("Customer Analyst", f"❌ Error: {str(e)}")

        # Tech Agent
        if "github" in datasets:
            update_progress("Tech Analyst", "Analyzing GitHub/tech data...")
            try:
                self._reports["tech"] = self.tech_agent.analyze(
                    datasets["github"],
                    include_anomalies=include_ml
                )
                update_progress("Tech Analyst", "✅ Complete")
            except Exception as e:
                self._reports["tech"] = f"[Error in Tech Analysis: {str(e)}]"
                update_progress("Tech Analyst", f"❌ Error: {str(e)}")

        # ─── Phase 2: Strategic Synthesis ─────────────────────────

        update_progress("Strategy Agent", "Synthesizing all insights...")

        try:
            self._strategic_report = self.strategy_agent.synthesize(
                sales_report=self._reports.get("sales", "No sales data available."),
                marketing_report=self._reports.get("marketing", "No marketing data available."),
                customer_report=self._reports.get("customers", "No customer data available."),
                tech_report=self._reports.get("tech", "No tech data available.")
            )
            update_progress("Strategy Agent", "✅ Complete")
        except Exception as e:
            self._strategic_report = f"[Error in Strategic Synthesis: {str(e)}]"
            update_progress("Strategy Agent", f"❌ Error: {str(e)}")

        # ─── Phase 3: Generate Recommendations ───────────────────

        update_progress("Recommendations", "Generating final recommendations...")

        try:
            all_insights = "\n\n---\n\n".join([
                f"## {k.title()} Analysis\n{v}"
                for k, v in self._reports.items()
            ])
            self._recommendations = self.strategy_agent.generate_recommendations(
                all_insights
            )
            update_progress("Recommendations", "✅ Complete")
        except Exception as e:
            self._recommendations = f"[Error generating recommendations: {str(e)}]"
            update_progress("Recommendations", f"❌ Error: {str(e)}")

        self._total_time = round(time.time() - start_time, 2)

        return self.get_results()

    def run_single_agent(
        self,
        agent_type: str,
        df: pd.DataFrame,
        include_ml: bool = True
    ) -> str:
        """
        Run a single agent's analysis.

        Args:
            agent_type: One of 'sales', 'marketing', 'customers', 'tech'.
            df: Input DataFrame.
            include_ml: Whether to include ML signals.

        Returns:
            Agent's analysis report.
        """
        agent_map = {
            "sales": (self.sales_agent, {"include_trends": include_ml}),
            "marketing": (self.marketing_agent, {"include_trends": include_ml}),
            "customers": (self.customer_agent, {"include_clustering": include_ml}),
            "tech": (self.tech_agent, {"include_anomalies": include_ml})
        }

        if agent_type not in agent_map:
            return f"Unknown agent type: {agent_type}"

        agent, kwargs = agent_map[agent_type]
        report = agent.analyze(df, **kwargs)
        self._reports[agent_type] = report
        return report

    def get_results(self) -> Dict[str, Any]:
        """Get all results from the analysis pipeline."""
        return {
            "reports": self._reports.copy(),
            "strategic_report": self._strategic_report,
            "recommendations": self._recommendations,
            "agent_status": self._agent_status.copy(),
            "metadata": {
                "total_execution_time": self._total_time,
                "agents_run": list(self._reports.keys()),
                "llm_model": self.llm_client.model,
                "llm_stats": self.llm_client.get_stats(),
                "agent_metadata": {
                    "sales": self.sales_agent.get_metadata(),
                    "marketing": self.marketing_agent.get_metadata(),
                    "customers": self.customer_agent.get_metadata(),
                    "tech": self.tech_agent.get_metadata(),
                    "strategy": self.strategy_agent.get_metadata()
                }
            }
        }

    def get_agent_reports(self) -> Dict[str, str]:
        """Get all individual agent reports."""
        return self._reports.copy()

    def get_strategic_report(self) -> Optional[str]:
        """Get the strategic synthesis report."""
        return self._strategic_report

    def get_recommendations(self) -> Optional[str]:
        """Get the final recommendations."""
        return self._recommendations
