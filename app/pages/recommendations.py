"""
Recommendations Page
=====================
Display final strategic recommendations with priority,
impact assessment, and action items.
"""

import os
import sys
import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llm.llm_client import LLMClient
from llm.prompts import PromptTemplates
from llm.response_parser import ResponseParser
from components.ui_elements import insight_card
from components.theme import apply_dark_page_style


def render():
    """Render the Recommendations page."""
    apply_dark_page_style()
    st.markdown("# 🎯 Strategic Recommendations")
    st.markdown(
        "AI-generated prioritized action items based on all available analysis. "
        "Run the multi-agent analysis first for best results."
    )
    st.markdown("---")

    # ─── Check for existing analysis ───────────────────────────

    has_multi_agent = "multi_agent_results" in st.session_state
    has_insights = any(
        k.startswith("insight_") for k in st.session_state.keys()
    )

    if not has_multi_agent and not has_insights:
        st.info(
            "💡 **Tip**: Run the Multi-Agent Analysis first for comprehensive "
            "recommendations, or generate individual AI Insights."
        )

    # ─── Display Multi-Agent Recommendations ───────────────────

    if has_multi_agent:
        results = st.session_state["multi_agent_results"]

        st.markdown("### 📋 From Multi-Agent Analysis")

        # Strategic report
        strategic = results.get("strategic_report")
        if strategic:
            with st.expander("🧠 Strategic Synthesis", expanded=True):
                st.markdown(strategic)

        # Recommendations
        recs = results.get("recommendations")
        if recs:
            st.markdown("---")
            st.markdown("### 🎯 Prioritized Action Items")
            st.markdown(recs)

            # Parse into structured format
            parser = ResponseParser()
            parsed_recs = parser.parse_recommendations(recs)

            if parsed_recs:
                st.markdown("---")
                st.markdown("### 📊 Recommendations Summary")

                for i, rec in enumerate(parsed_recs, 1):
                    priority = rec.get("priority", "Medium")
                    priority_color = {
                        "High": "🔴",
                        "Medium": "🟡",
                        "Low": "🟢"
                    }.get(priority, "⚪")

                    st.markdown(
                        f"""
                        <div style="
                            background: #1A1D23;
                            border-left: 4px solid {'#FF6B6B' if priority == 'High' else '#FFE66D' if priority == 'Medium' else '#4ECDC4'};
                            border-radius: 8px;
                            padding: 15px 20px;
                            margin: 8px 0;
                        ">
                            <strong style="color: #FAFAFA;">
                                {priority_color} #{i}: {rec.get('title', 'Recommendation')}
                            </strong>
                            <br>
                            <span style="color: #8B949E; font-size: 0.85rem;">
                                Priority: {priority}
                                {' | Impact: ' + rec.get('impact', '') if rec.get('impact') else ''}
                            </span>
                            <br>
                            <span style="color: #E0E0E0;">{rec.get('details', '')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # ─── Generate New Recommendations ──────────────────────────

    st.markdown("---")
    st.markdown("### 🔮 Generate Custom Recommendations")

    context_source = st.radio(
        "Context source",
        ["All available insights", "Custom context"],
        horizontal=True
    )

    custom_context = ""
    if context_source == "Custom context":
        custom_context = st.text_area(
            "Enter your analysis context",
            height=150,
            placeholder="Paste data summaries, KPIs, or previous insights..."
        )
    elif context_source == "All available insights":
        # Gather all available insights
        parts = []
        if has_multi_agent:
            for key, report in st.session_state["multi_agent_results"].get("reports", {}).items():
                parts.append(f"## {key.title()} Analysis\n{report}")
        for key in st.session_state.keys():
            if key.startswith("insight_"):
                dataset = key.replace("insight_", "")
                parts.append(f"## {dataset.title()} Insight\n{st.session_state[key]}")

        custom_context = "\n\n---\n\n".join(parts) if parts else ""

    if st.button("🎯 Generate Recommendations", type="primary"):
        if not custom_context:
            st.warning("No analysis context available. Run AI Insights or Multi-Agent Analysis first.")
            return

        with st.spinner("🧠 Generating strategic recommendations..."):
            if "llm_client" not in st.session_state:
                st.session_state["llm_client"] = LLMClient()

            llm = st.session_state["llm_client"]
            prompt = PromptTemplates.recommendation_prompt(custom_context)

            response = llm.generate(
                prompt=prompt,
                system_prompt=PromptTemplates.SYSTEM_STRATEGIST,
                temperature=0.2
            )

            st.session_state["custom_recommendations"] = response

    # Show custom recommendations
    if "custom_recommendations" in st.session_state:
        st.markdown("---")
        insight_card("Custom Recommendations", "", "🎯")
        st.markdown(st.session_state["custom_recommendations"])

        st.download_button(
            "📥 Download Recommendations",
            st.session_state["custom_recommendations"],
            file_name="ai_recommendations.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    render()
