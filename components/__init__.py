# Components Package
from components.charts import (
    revenue_trend_chart, top_products_chart, campaign_performance_chart,
    customer_segments_chart, github_stats_chart
)


def _missing_streamlit(*_args, **_kwargs):
    raise ModuleNotFoundError(
        "streamlit is required for UI element helpers. Install it with 'pip install streamlit'."
    )


try:
    from components.ui_elements import metric_card, insight_card, agent_report_panel
except ModuleNotFoundError as exc:
    if exc.name == "streamlit":
        metric_card = _missing_streamlit
        insight_card = _missing_streamlit
        agent_report_panel = _missing_streamlit
    else:
        raise
