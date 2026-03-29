"""
AI Insights Page
=================
Single-agent LLM insight generation per dataset.
Uses dynamic DatasetDetector for any CSV schema.
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
from utils.dataset_detector import DatasetDetector
from components.ui_elements import insight_card
from components.theme import apply_dark_page_style


def _deterministic_fallback(dataset_name: str, detected_type: str, kpis: dict) -> str:
    """Fallback insight when model output quality is unstable."""
    lines = [
        f"## {dataset_name.title()} Analysis ({detected_type.title()})",
        "",
        "## Direct Answer",
        "Model output was unstable, so this answer is generated from computed KPIs.",
        "",
        "## Evidence from Data",
    ]

    filtered_kpis = [
        (k, v)
        for k, v in kpis.items()
        if k not in {"total_records", "total_columns", "detected_type", "detection_confidence"}
    ]

    for key, value in filtered_kpis[:8]:
        label = key.replace("_", " ").title()
        lines.append(f"- **{label}**: {value}")

    lines.extend(
        [
            "",
            "## Risks or Gaps",
            "- Detailed narrative interpretation is limited until the model returns stable output.",
            "",
            "## Recommended Next Checks",
            "- Re-run analysis with a narrower question.",
            "- Verify the active model is `business-analyst` and Ollama is healthy.",
        ]
    )

    return "\n".join(lines)


def render():
    """Render the AI Insights page."""
    apply_dark_page_style()
    st.markdown("# 🤖 AI-Powered Insights")
    st.markdown("Select a dataset and let the LLM generate deep analytical insights.")
    st.markdown("---")

    if "datasets" not in st.session_state or not st.session_state["datasets"]:
        st.warning("⚠️ No datasets loaded. Go to **Data Upload** to load data.")
        return

    datasets = st.session_state["datasets"]

    # ─── Initialize LLM Client ─────────────────────────────────

    if "llm_client" not in st.session_state:
        st.session_state["llm_client"] = LLMClient()

    llm_client = st.session_state["llm_client"]
    parser = ResponseParser()

    # ─── Dataset & Analysis Config ─────────────────────────────

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        dataset_name = st.selectbox(
            "Select Dataset for Analysis",
            list(datasets.keys()),
            format_func=lambda x: f"📊 {x.title()}"
        )
    with col2:
        include_ml = st.checkbox("Include ML Signals", value=True,
                                 help="Add trend/clustering/anomaly signals")
    with col3:
        detail_level = st.selectbox(
            "Detail Level",
            ["Concise", "Standard", "Deep Dive"],
            index=1,
            help="Controls response depth and length.",
        )

    analysis_focus = st.selectbox(
        "Analysis Focus",
        [
            "General insights",
            "Growth opportunities",
            "Risk detection",
            "Customer experience",
            "Operational efficiency",
        ],
        index=0,
    )

    df = datasets[dataset_name]

    # ─── Auto-Detect Dataset Type ──────────────────────────────

    detector = DatasetDetector(df, dataset_name)
    kpis = detector.compute_auto_kpis()

    # Show detection info
    st.info(
        f"🔍 **Auto-detected type:** {detector.detected_type.title()} "
        f"(confidence: {detector.confidence:.0%}) — "
        f"{len(detector.monetary_columns)} monetary, "
        f"{len(detector.numeric_columns)} numeric, "
        f"{len(detector.categorical_columns)} categorical columns"
    )

    # ─── Show KPIs ─────────────────────────────────────────────

    st.markdown("### 📈 Auto-Detected KPIs")
    kpi_items = [(k, v) for k, v in kpis.items()
                 if k not in ["total_records", "total_columns", "detected_type", "detection_confidence"]]

    cols = st.columns(4)
    for i, (key, val) in enumerate(kpi_items[:8]):
        with cols[i % 4]:
            label = key.replace("_", " ").title()
            if isinstance(val, float) and val > 1000:
                display_val = f"{val:,.2f}"
            elif isinstance(val, float):
                display_val = f"{val:.2f}"
            else:
                display_val = str(val)
            st.metric(label, display_val)

    st.markdown("---")

    # ─── Generate Insights ─────────────────────────────────────

    st.markdown("### 🧠 LLM Analysis")

    custom_question = st.text_area(
        "Custom question (optional)",
        placeholder="e.g., What are the top growth opportunities in this data?",
        height=80
    )

    if st.button("🔮 Generate AI Insights", type="primary"):
        with st.spinner("🧠 AI is analyzing your data..."):
            try:
                context = detector.get_analysis_context()

                if custom_question:
                    prompt = PromptTemplates.quick_insight(
                        data_summary=context,
                        question=custom_question
                    )
                else:
                    # Use universal dynamic prompt
                    prompt = PromptTemplates.dynamic_analysis(
                        detected_type=detector.detected_type,
                        context=context
                    )

                detail_instruction = {
                    "Concise": "Keep total output under 220 words.",
                    "Standard": "Keep total output under 380 words.",
                    "Deep Dive": "Provide a thorough analysis up to 650 words.",
                }[detail_level]

                focus_instruction = (
                    f"\n\n## Focus Requirement\nPrioritize insights about: {analysis_focus}."
                )

                if include_ml:
                    chart_titles = [spec.get("title", "") for spec in detector.get_chart_recommendations()[:5]]
                    ml_hint = (
                        "\n\n## ML/Visualization Hints\n"
                        f"Recommended chart intents: {', '.join([t for t in chart_titles if t])}."
                    )
                else:
                    ml_hint = ""

                prompt = (
                    prompt
                    + focus_instruction
                    + ml_hint
                    + "\n\n## Length Constraint\n"
                    + detail_instruction
                )

                response = llm_client.generate_structured(
                    prompt=prompt,
                    system_prompt=PromptTemplates.SYSTEM_ANALYST_STRICT,
                    temperature=0.15
                )
                response = parser.clean_response(response)

                if parser.is_low_quality_response(response):
                    retry_prompt = PromptTemplates.repair_insight(
                        data_summary=context,
                        question=custom_question or "Provide a concise analysis of this dataset.",
                        previous_output=response,
                    )
                    response = llm_client.generate_structured(
                        prompt=retry_prompt,
                        system_prompt=PromptTemplates.SYSTEM_ANALYST_STRICT,
                        temperature=0.1,
                    )
                    response = parser.clean_response(response)

                if parser.is_low_quality_response(response):
                    response = _deterministic_fallback(
                        dataset_name=dataset_name,
                        detected_type=detector.detected_type,
                        kpis=kpis,
                    )
                    st.warning(
                        "Model output looked unstable, so a KPI-based fallback summary was shown."
                    )

                st.session_state[f"insight_{dataset_name}"] = response

            except Exception as e:
                st.error(f"❌ Error generating insights: {str(e)}")

    # ─── Display Results ───────────────────────────────────────

    insight_key = f"insight_{dataset_name}"
    if insight_key in st.session_state:
        insight_card(
            title=f"{dataset_name.title()} Analysis ({detector.detected_type.title()})",
            content="",
            icon="🧠"
        )
        st.markdown(st.session_state[insight_key])

        st.download_button(
            "📥 Download Report",
            st.session_state[insight_key],
            file_name=f"{dataset_name}_ai_insight.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    render()
