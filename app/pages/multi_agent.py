"""
Multi-Agent Analysis Page
==========================
Run all agents collaboratively, show pipeline progress,
and display the combined strategic analysis.
"""

import os
import sys
import streamlit as st
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm.llm_client import LLMClient
from agents.orchestrator import AgentOrchestrator
from components.ui_elements import agent_report_panel, insight_card
from components.theme import apply_dark_page_style


def render():
    """Render the Multi-Agent Analysis page."""
    apply_dark_page_style()
    st.markdown("# 🔗 Multi-Agent Collaborative Analysis")
    st.markdown(
        "Run the full AI analyst team: **Sales → Marketing → Customer → Tech → Strategy**. "
        "Each agent uses LLM reasoning enhanced with optional ML signals."
    )
    st.markdown("---")

    if "datasets" not in st.session_state or not st.session_state["datasets"]:
        st.warning("⚠️ No datasets loaded. Go to **Data Upload** to load data.")
        return

    datasets = st.session_state["datasets"]

    # ─── Configuration ─────────────────────────────────────────

    st.markdown("### ⚙️ Analysis Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        include_ml = st.checkbox("Enable ML Signals", value=True,
                                 help="Clustering, anomaly detection, trend analysis")
    with col2:
        n_clusters = st.slider("Customer Clusters", 2, 8, 4)
    with col3:
        contamination = st.slider("Anomaly Sensitivity", 0.05, 0.3, 0.1, 0.05)

    st.markdown("---")

    # ─── Agent Pipeline Visualization ──────────────────────────

    st.markdown("### 🤖 Agent Team")
    agent_cols = st.columns(5)
    agents_info = [
        ("📊", "Sales Analyst", "Revenue & product trends"),
        ("🎯", "Marketing Analyst", "Campaign ROI & channels"),
        ("👥", "Customer Analyst", "Segments & churn risk"),
        ("💻", "Tech Analyst", "GitHub & code quality"),
        ("🧠", "Strategy Agent", "Unified recommendations")
    ]
    for i, (icon, name, desc) in enumerate(agents_info):
        with agent_cols[i]:
            st.markdown(f"""
            <div style="
                background: #1A1D23;
                border-radius: 12px;
                padding: 15px;
                text-align: center;
                border: 1px solid #2D3139;
            ">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="color: #00D4FF; font-weight: 600; font-size: 0.85rem;">{name}</div>
                <div style="color: #8B949E; font-size: 0.75rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Run Analysis ──────────────────────────────────────────

    if st.button("🚀 Run Full Multi-Agent Analysis", type="primary"):
        # Initialize LLM and orchestrator
        if "llm_client" not in st.session_state:
            st.session_state["llm_client"] = LLMClient()

        orchestrator = AgentOrchestrator(st.session_state["llm_client"])

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        step = 0
        total_steps = 6  # 4 agents + strategy + recommendations

        def progress_callback(agent_name: str, status: str):
            nonlocal step
            step += 1
            progress_bar.progress(min(step / total_steps, 1.0))
            status_text.markdown(f"**{agent_name}**: {status}")

        # Run the pipeline
        with st.spinner("Running multi-agent analysis pipeline..."):
            results = orchestrator.run_full_analysis(
                datasets=datasets,
                include_ml=include_ml,
                progress_callback=progress_callback
            )

        progress_bar.progress(1.0)
        status_text.markdown("**✅ All agents complete!**")

        # Store results
        st.session_state["multi_agent_results"] = results
        st.session_state["orchestrator"] = orchestrator

    # ─── Display Results ───────────────────────────────────────

    if "multi_agent_results" in st.session_state:
        results = st.session_state["multi_agent_results"]

        st.markdown("---")
        st.markdown("## 📊 Agent Reports")

        # Execution metadata
        meta = results.get("metadata", {})
        st.caption(
            f"⏱️ Total time: {meta.get('total_execution_time', 0)}s | "
            f"🤖 Model: {meta.get('llm_model', 'N/A')} | "
            f"📊 Agents: {len(meta.get('agents_run', []))}"
        )

        # Individual agent reports
        reports = results.get("reports", {})

        agent_names = {
            "sales": "📊 Sales Analyst",
            "marketing": "🎯 Marketing Analyst",
            "customers": "👥 Customer Analyst",
            "tech": "💻 Tech Analyst"
        }

        for key, display_name in agent_names.items():
            if key in reports:
                agent_meta = meta.get("agent_metadata", {}).get(key, {})
                exec_time = agent_meta.get("execution_time_seconds", 0)
                agent_report_panel(
                    agent_name=display_name,
                    report=reports[key],
                    execution_time=exec_time
                )

        # ─── Strategic Report ──────────────────────────────────

        st.markdown("---")
        st.markdown("## 🧠 Strategic Synthesis")

        strategic_report = results.get("strategic_report")
        if strategic_report:
            insight_card(
                title="Unified Strategic Analysis",
                content="",
                icon="🧠"
            )
            st.markdown(strategic_report)

        # ─── Recommendations ──────────────────────────────────

        st.markdown("---")
        st.markdown("## 🎯 Actionable Recommendations")

        recommendations = results.get("recommendations")
        if recommendations:
            st.markdown(recommendations)

            # Download full report
            full_report = "# Multi-Agent Analysis Report\n\n"
            for key, report in reports.items():
                full_report += f"## {key.title()} Analysis\n\n{report}\n\n---\n\n"
            full_report += f"## Strategic Synthesis\n\n{strategic_report}\n\n---\n\n"
            full_report += f"## Recommendations\n\n{recommendations}\n"

            st.download_button(
                "📥 Download Full Report",
                full_report,
                file_name="multi_agent_analysis_report.md",
                mime="text/markdown"
            )


if __name__ == "__main__":
    render()
