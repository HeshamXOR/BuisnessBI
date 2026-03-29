"""
UI Elements Module
==================
Reusable Streamlit UI components for the dashboard.
Provides styled metric cards, insight panels, and report displays.
"""

import streamlit as st
from typing import Optional


def metric_card(label: str, value: str, delta: Optional[str] = None,
                delta_color: str = "normal"):
    """
    Render a styled metric card using Streamlit's metric component.

    Args:
        label: Metric name.
        value: Metric value.
        delta: Optional change/delta value.
        delta_color: Color for delta ('normal', 'inverse', 'off').
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def insight_card(title: str, content: str, icon: str = "💡"):
    """
    Render an insight card with styled container.

    Args:
        title: Card title.
        content: Insight text (supports markdown).
        icon: Emoji icon for the title.
    """
    with st.container():
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1d23 0%, #2d3139 100%);
                border-left: 4px solid #00D4FF;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
            ">
                <h4 style="color: #00D4FF; margin-top: 0;">{icon} {title}</h4>
                <div style="color: #E0E0E0; line-height: 1.6;">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def agent_report_panel(agent_name: str, report: str, status: str = "✅ Complete",
                       execution_time: float = 0):
    """
    Render an agent's report in a styled expandable panel.

    Args:
        agent_name: Name of the agent.
        report: Agent's report text (markdown).
        status: Agent status indicator.
        execution_time: Time taken in seconds.
    """
    with st.expander(f"🤖 {agent_name} — {status}", expanded=False):
        if execution_time > 0:
            st.caption(f"⏱️ Execution time: {execution_time:.1f}s")
        st.markdown(report)


def status_indicator(agent_name: str, status: str):
    """
    Render a small status indicator for an agent.

    Args:
        agent_name: Name of the agent.
        status: Status text with emoji.
    """
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"**{agent_name}**")
    with col2:
        st.markdown(status)


def section_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled section header."""
    st.markdown(f"## {icon} {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


def kpi_row(kpis: dict, columns: int = 4):
    """
    Render a row of KPI metrics.

    Args:
        kpis: Dict of label → (value, delta) or label → value.
        columns: Number of columns.
    """
    cols = st.columns(columns)
    for i, (label, value) in enumerate(kpis.items()):
        with cols[i % columns]:
            if isinstance(value, tuple):
                metric_card(label, str(value[0]), str(value[1]))
            else:
                metric_card(label, str(value))


def llm_status_badge(connected: bool, model: str = ""):
    """Render LLM connection status badge."""
    if connected:
        st.sidebar.success(f"🟢 LLM Connected: {model}")
    else:
        st.sidebar.warning("🔴 LLM Disconnected — Using fallback mode")


def render_markdown_report(report: str, container_color: str = "#1A1D23"):
    """Render a full markdown report in a styled container."""
    st.markdown(
        f"""
        <div style="
            background-color: {container_color};
            border-radius: 12px;
            padding: 24px;
            margin: 10px 0;
            border: 1px solid #2D3139;
        ">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(report)
