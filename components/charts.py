"""
Visualization Charts Module
============================
Plotly chart builders for the Streamlit dashboard.
All charts use a consistent dark theme with vibrant accent colors.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


# ─── Theme Configuration ──────────────────────────────────────────

THEME = {
    "bg_color": "#0E1117",
    "paper_color": "#1A1D23",
    "text_color": "#FAFAFA",
    "grid_color": "#2D3139",
    "accent_colors": [
        "#00D4FF",  # Cyan
        "#FF6B6B",  # Coral
        "#4ECDC4",  # Teal
        "#FFE66D",  # Yellow
        "#A78BFA",  # Purple
        "#F472B6",  # Pink
        "#34D399",  # Green
        "#FB923C",  # Orange
    ],
    "font_family": "Inter, sans-serif"
}


def _empty_figure(title: str, message: str = "Not enough valid data for this chart") -> go.Figure:
    """Return a themed placeholder figure instead of a blank chart."""
    fig = go.Figure()
    _apply_theme(fig, title)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=THEME["text_color"], size=14),
    )
    return fig


def _as_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric with NaN for invalid values."""
    return pd.to_numeric(series, errors="coerce")


def _positive_log_ok(series: pd.Series) -> bool:
    """Check whether a numeric series is valid for logarithmic scaling."""
    if series is None or series.empty:
        return False
    s = _as_numeric(series).dropna()
    return (not s.empty) and bool((s > 0).all())


def _truncate_categories(series: pd.Series, top_n: int = 10):
    """Keep top categories and collapse the rest into 'Other'."""
    counts = series.value_counts(dropna=True)
    top = counts.head(top_n)
    if len(counts) <= top_n:
        return series
    top_idx = set(top.index)
    return series.apply(lambda v: v if v in top_idx else "Other")


def _valid_count(values) -> int:
    """Count non-null values for Plotly trace arrays."""
    if values is None:
        return 0
    s = pd.Series(list(values))
    return int(s.notna().sum())


def _valid_pair_count(x_values, y_values) -> int:
    """Count non-null x/y pairs for point-based traces."""
    if x_values is None or y_values is None:
        return 0
    x = pd.Series(list(x_values))
    y = pd.Series(list(y_values))
    n = min(len(x), len(y))
    if n == 0:
        return 0
    x = x.iloc[:n]
    y = y.iloc[:n]
    return int((x.notna() & y.notna()).sum())


def _figure_has_data(fig: go.Figure) -> bool:
    """Detect whether a figure has at least one meaningful, visible trace."""
    if not fig or not fig.data:
        return False

    for trace in fig.data:
        trace_type = getattr(trace, "type", "")

        if trace_type == "pie":
            values = getattr(trace, "values", None)
            labels = getattr(trace, "labels", None)
            if values is not None:
                v = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
                if len(v) >= 2 and float(v.sum()) > 0:
                    return True
            if labels is not None and _valid_count(labels) >= 2:
                return True
            continue

        if trace_type in {"scatter", "scattergl", "scatterpolar"}:
            if _valid_pair_count(getattr(trace, "x", None), getattr(trace, "y", None)) >= 3:
                return True
            continue

        if trace_type == "histogram":
            x_values = getattr(trace, "x", None)
            if x_values is not None:
                x = pd.to_numeric(pd.Series(list(x_values)), errors="coerce").dropna()
                if len(x) >= 5 and x.nunique() >= 3:
                    return True
            continue

        if trace_type == "bar":
            y_values = getattr(trace, "y", None)
            if y_values is not None:
                y = pd.to_numeric(pd.Series(list(y_values)), errors="coerce").dropna()
                if len(y) >= 2 and float(y.abs().sum()) > 0:
                    return True
            if _valid_count(getattr(trace, "x", None)) >= 2:
                return True
            continue

        if trace_type in {"box", "violin"}:
            if _valid_count(getattr(trace, "y", None)) >= 5:
                return True
            continue

        if _valid_count(getattr(trace, "x", None)) > 0 or _valid_count(getattr(trace, "y", None)) > 0:
            return True

    return False


def chart_has_meaningful_data(fig: go.Figure) -> bool:
    """Public helper to check if a chart has enough data to be useful."""
    return _figure_has_data(fig)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent dark theme to a plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=THEME["text_color"], family=THEME["font_family"]),
            x=0.02
        ),
        paper_bgcolor=THEME["paper_color"],
        plot_bgcolor=THEME["bg_color"],
        font=dict(color=THEME["text_color"], family=THEME["font_family"], size=12),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(26, 29, 35, 0.8)",
            bordercolor=THEME["grid_color"],
            borderwidth=1
        ),
        xaxis=dict(gridcolor=THEME["grid_color"], zeroline=False),
        yaxis=dict(gridcolor=THEME["grid_color"], zeroline=False),
        hovermode="closest"
    )
    return fig


# ─── Sales Charts ──────────────────────────────────────────────────

def revenue_trend_chart(df: pd.DataFrame, date_col: str = "date",
                        value_col: str = "revenue") -> go.Figure:
    """Create an interactive revenue trend chart with moving average."""
    if date_col not in df.columns or value_col not in df.columns:
        return _empty_figure("📈 Revenue Trend", "Required columns are missing.")

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    df_copy[value_col] = _as_numeric(df_copy[value_col])
    df_copy = df_copy.dropna(subset=[date_col, value_col])
    if df_copy.empty:
        return _empty_figure("📈 Revenue Trend", "No valid date/value rows found.")

    # Monthly aggregation
    monthly = df_copy.groupby(pd.Grouper(key=date_col, freq="ME"))[value_col].sum().reset_index()
    if monthly.empty:
        return _empty_figure("📈 Revenue Trend", "No monthly points available.")

    fig = go.Figure()

    # Main revenue line
    fig.add_trace(go.Scatter(
        x=monthly[date_col],
        y=monthly[value_col],
        mode="lines+markers",
        name="Monthly Revenue",
        line=dict(color=THEME["accent_colors"][0], width=2.5),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.1)"
    ))

    # 3-month moving average
    if len(monthly) >= 3:
        monthly["ma"] = monthly[value_col].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=monthly[date_col],
            y=monthly["ma"],
            mode="lines",
            name="3-Month Avg",
            line=dict(color=THEME["accent_colors"][3], width=2, dash="dash")
        ))

    _apply_theme(fig, "📈 Revenue Trend")
    fig.update_yaxes(title="Revenue ($)")
    fig.update_xaxes(title="Month")
    return fig


def top_products_chart(df: pd.DataFrame, n: int = 8) -> go.Figure:
    """Create a horizontal bar chart of top products by revenue."""
    top = df.groupby("product")["revenue"].sum().nlargest(n).sort_values()

    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index,
        orientation="h",
        marker=dict(
            color=top.values,
            colorscale=[[0, THEME["accent_colors"][2]], [1, THEME["accent_colors"][0]]],
            line=dict(width=0)
        ),
        text=[f"${v:,.0f}" for v in top.values],
        textposition="auto"
    ))

    _apply_theme(fig, f"🏆 Top {n} Products by Revenue")
    fig.update_xaxes(title="Total Revenue ($)")
    return fig


def revenue_by_region_chart(df: pd.DataFrame) -> go.Figure:
    """Create a pie/donut chart for revenue by region."""
    region_rev = df.groupby("region")["revenue"].sum().reset_index()

    fig = go.Figure(go.Pie(
        labels=region_rev["region"],
        values=region_rev["revenue"],
        hole=0.45,
        marker=dict(colors=THEME["accent_colors"]),
        textinfo="label+percent",
        textfont=dict(size=12)
    ))

    _apply_theme(fig, "🌍 Revenue Distribution by Region")
    return fig


def revenue_by_category_chart(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart for revenue by category."""
    cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()

    fig = go.Figure(go.Bar(
        x=cat_rev["category"],
        y=cat_rev["revenue"],
        marker=dict(
            color=THEME["accent_colors"][:len(cat_rev)],
            line=dict(width=0)
        ),
        text=[f"${v:,.0f}" for v in cat_rev["revenue"]],
        textposition="auto"
    ))

    _apply_theme(fig, "📊 Revenue by Category")
    fig.update_yaxes(title="Revenue ($)")
    return fig


# ─── Marketing Charts ─────────────────────────────────────────────

def campaign_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Create a grouped bar chart comparing campaign types by ROI."""
    campaign_stats = df.groupby("campaign_type").agg({
        "roi": "mean",
        "conversions": "sum",
        "spend": "sum"
    }).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=campaign_stats["campaign_type"],
        y=campaign_stats["roi"],
        name="Avg ROI (%)",
        marker_color=THEME["accent_colors"][0],
        text=[f"{v:.1f}%" for v in campaign_stats["roi"]],
        textposition="auto"
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=campaign_stats["campaign_type"],
        y=campaign_stats["conversions"],
        name="Total Conversions",
        mode="lines+markers",
        line=dict(color=THEME["accent_colors"][1], width=2.5),
        marker=dict(size=8)
    ), secondary_y=True)

    _apply_theme(fig, "🎯 Campaign Performance")
    fig.update_yaxes(title="Average ROI (%)", secondary_y=False)
    fig.update_yaxes(title="Total Conversions", secondary_y=True)
    return fig


def channel_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create a radar chart comparing marketing channels."""
    channel_stats = df.groupby("channel").agg({
        "ctr": "mean",
        "conversion_rate": "mean",
        "roi": "mean",
        "cpc": "mean"
    }).reset_index()

    # Normalize metrics for radar chart
    metrics = ["ctr", "conversion_rate", "roi"]
    for m in metrics:
        max_val = channel_stats[m].max()
        if max_val > 0:
            channel_stats[f"{m}_norm"] = channel_stats[m] / max_val * 100
        else:
            channel_stats[f"{m}_norm"] = 0

    fig = go.Figure()
    categories = ["CTR", "Conversion Rate", "ROI"]

    for i, (_, row) in enumerate(channel_stats.iterrows()):
        values = [row["ctr_norm"], row["conversion_rate_norm"], row["roi_norm"]]
        values.append(values[0])  # Close the polygon
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            name=row["channel"],
            line=dict(color=THEME["accent_colors"][i % len(THEME["accent_colors"])])
        ))

    _apply_theme(fig, "📡 Channel Comparison")
    fig.update_layout(polar=dict(
        bgcolor=THEME["bg_color"],
        radialaxis=dict(gridcolor=THEME["grid_color"], showticklabels=False),
        angularaxis=dict(gridcolor=THEME["grid_color"])
    ))
    return fig


# ─── Customer Charts ───────────────────────────────────────────────

def customer_segments_chart(df: pd.DataFrame) -> go.Figure:
    """Create a scatter plot of customer segments by LTV vs satisfaction."""
    required = ["satisfaction_score", "lifetime_value", "segment", "engagement_score"]
    if any(c not in df.columns for c in required):
        return _empty_figure("👥 Customer Segments: Value vs Satisfaction", "Required columns are missing.")

    df_copy = df.copy()
    df_copy["satisfaction_score"] = _as_numeric(df_copy["satisfaction_score"])
    df_copy["lifetime_value"] = _as_numeric(df_copy["lifetime_value"])
    df_copy["engagement_score"] = _as_numeric(df_copy["engagement_score"])
    df_copy = df_copy.dropna(subset=["satisfaction_score", "lifetime_value", "engagement_score", "segment"])
    if df_copy.empty:
        return _empty_figure("👥 Customer Segments: Value vs Satisfaction", "No valid segment points found.")

    use_log_y = _positive_log_ok(df_copy["lifetime_value"])

    fig = px.scatter(
        df_copy,
        x="satisfaction_score",
        y="lifetime_value",
        color="segment",
        size="engagement_score",
        hover_data=[c for c in ["churn_risk", "purchase_frequency"] if c in df_copy.columns],
        color_discrete_sequence=THEME["accent_colors"],
        log_y=use_log_y
    )

    _apply_theme(fig, "👥 Customer Segments: Value vs Satisfaction")
    fig.update_xaxes(title="Satisfaction Score")
    fig.update_yaxes(title="Lifetime Value ($, log scale)" if use_log_y else "Lifetime Value ($)")
    return fig


def churn_risk_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create a histogram of churn risk distribution."""
    if "churn_risk" not in df.columns:
        return _empty_figure("⚠️ Churn Risk Distribution", "Column churn_risk is missing.")

    x = _as_numeric(df["churn_risk"]).dropna()
    if x.empty:
        return _empty_figure("⚠️ Churn Risk Distribution", "No valid churn risk values found.")

    fig = go.Figure(go.Histogram(
        x=x,
        nbinsx=30,
        marker_color=THEME["accent_colors"][1],
        marker_line_color=THEME["accent_colors"][0],
        marker_line_width=1,
        opacity=0.8
    ))

    # Add threshold line
    fig.add_vline(x=0.5, line_dash="dash", line_color=THEME["accent_colors"][3],
                  annotation_text="High Risk Threshold")

    _apply_theme(fig, "⚠️ Churn Risk Distribution")
    fig.update_xaxes(title="Churn Risk Score")
    fig.update_yaxes(title="Number of Customers")
    return fig


def segment_ltv_chart(df: pd.DataFrame) -> go.Figure:
    """Create a box plot of LTV by customer segment."""
    if "segment" not in df.columns or "lifetime_value" not in df.columns:
        return _empty_figure("💰 Lifetime Value by Segment", "Required columns are missing.")

    df_copy = df.copy()
    df_copy["lifetime_value"] = _as_numeric(df_copy["lifetime_value"])
    df_copy = df_copy.dropna(subset=["segment", "lifetime_value"])
    if df_copy.empty:
        return _empty_figure("💰 Lifetime Value by Segment", "No valid LTV values found.")

    use_log_y = _positive_log_ok(df_copy["lifetime_value"])

    fig = px.box(
        df_copy,
        x="segment",
        y="lifetime_value",
        color="segment",
        color_discrete_sequence=THEME["accent_colors"],
        log_y=use_log_y
    )

    _apply_theme(fig, "💰 Lifetime Value by Segment")
    fig.update_xaxes(title="Customer Segment")
    fig.update_yaxes(title="Lifetime Value ($, log scale)" if use_log_y else "Lifetime Value ($)")
    return fig


# ─── GitHub/Tech Charts ───────────────────────────────────────────

def github_stats_chart(df: pd.DataFrame) -> go.Figure:
    """Create a scatter plot of GitHub repos: stars vs forks colored by language."""
    required = ["stars", "forks"]
    if any(c not in df.columns for c in required):
        return _empty_figure("🐙 GitHub Repos: Stars vs Forks", "Required columns are missing.")

    df_copy = df.copy()
    for col in ["stars", "forks", "contributors", "code_quality_score", "open_issues"]:
        if col in df_copy.columns:
            df_copy[col] = _as_numeric(df_copy[col])

    base_cols = ["stars", "forks"]
    if "language" in df_copy.columns:
        base_cols.append("language")
    df_copy = df_copy.dropna(subset=base_cols)
    if df_copy.empty:
        return _empty_figure("🐙 GitHub Repos: Stars vs Forks", "No valid stars/forks pairs found.")

    use_log_x = _positive_log_ok(df_copy["stars"])
    use_log_y = _positive_log_ok(df_copy["forks"])

    fig = px.scatter(
        df_copy,
        x="stars",
        y="forks",
        color="language" if "language" in df_copy.columns else None,
        size="contributors" if "contributors" in df_copy.columns else None,
        hover_data=[c for c in ["repo_name", "code_quality_score", "open_issues"] if c in df_copy.columns],
        color_discrete_sequence=THEME["accent_colors"],
        log_x=use_log_x,
        log_y=use_log_y
    )

    _apply_theme(fig, "🐙 GitHub Repos: Stars vs Forks")
    fig.update_xaxes(title="Stars (log scale)" if use_log_x else "Stars")
    fig.update_yaxes(title="Forks (log scale)" if use_log_y else "Forks")
    return fig


def language_popularity_chart(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart of programming language popularity."""
    lang_stats = df.groupby("language").agg({
        "stars": "sum",
        "repo_name": "count"
    }).reset_index()
    lang_stats.columns = ["language", "total_stars", "repo_count"]
    lang_stats = lang_stats.sort_values("total_stars", ascending=False)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=lang_stats["language"],
        y=lang_stats["total_stars"],
        name="Total Stars",
        marker_color=THEME["accent_colors"][4],
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=lang_stats["language"],
        y=lang_stats["repo_count"],
        name="Repo Count",
        mode="lines+markers",
        line=dict(color=THEME["accent_colors"][3], width=2.5),
        marker=dict(size=8)
    ), secondary_y=True)

    _apply_theme(fig, "💻 Programming Language Popularity")
    fig.update_yaxes(title="Total Stars", secondary_y=False)
    fig.update_yaxes(title="Number of Repos", secondary_y=True)
    return fig


def code_quality_chart(df: pd.DataFrame) -> go.Figure:
    """Create a violin plot of code quality by language."""
    if "language" not in df.columns or "code_quality_score" not in df.columns:
        return _empty_figure("🔍 Code Quality Distribution by Language", "Required columns are missing.")

    df_copy = df.copy()
    df_copy["code_quality_score"] = _as_numeric(df_copy["code_quality_score"])
    df_copy = df_copy.dropna(subset=["language", "code_quality_score"])
    if df_copy.empty:
        return _empty_figure("🔍 Code Quality Distribution by Language", "No valid quality scores found.")

    # Filter to top 8 languages
    top_langs = df_copy["language"].value_counts().head(8).index.tolist()
    df_filtered = df_copy[df_copy["language"].isin(top_langs)]
    if df_filtered.empty:
        return _empty_figure("🔍 Code Quality Distribution by Language", "No language groups available.")

    fig = px.violin(
        df_filtered,
        x="language",
        y="code_quality_score",
        color="language",
        box=True,
        color_discrete_sequence=THEME["accent_colors"]
    )

    _apply_theme(fig, "🔍 Code Quality Distribution by Language")
    fig.update_xaxes(title="Language")
    fig.update_yaxes(title="Code Quality Score")
    return fig


# ─── Dynamic / Auto Chart Generation ─────────────────────────────

def auto_chart(df: pd.DataFrame, chart_spec: dict) -> Optional[go.Figure]:
    """
    Generate a Plotly chart from a chart specification dict.
    Works with any dataset — the DatasetDetector produces chart specs.

    Args:
        df: DataFrame to visualize.
        chart_spec: Dict with 'type', 'x', 'y', 'title' keys.

    Returns:
        Plotly Figure or None if the chart can't be created.
    """
    chart_type = chart_spec.get("type", "bar")
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    title = chart_spec.get("title", "Chart")

    if x_col and x_col not in df.columns:
        return None
    if y_col and y_col != "count" and y_col not in df.columns:
        return None

    try:
        if chart_type == "line":
            return _auto_line_chart(df, x_col, y_col, title)
        elif chart_type == "bar":
            return _auto_bar_chart(df, x_col, y_col, title)
        elif chart_type == "histogram":
            return _auto_histogram(df, x_col, title)
        elif chart_type == "scatter":
            return _auto_scatter(df, x_col, y_col, title)
        elif chart_type == "pie":
            return _auto_pie_chart(df, x_col, y_col, title)
        elif chart_type == "box":
            return _auto_box_chart(df, x_col, y_col, title)
        elif chart_type == "heatmap":
            return _auto_heatmap(df, chart_spec.get("columns", []), title)
        elif chart_type == "stacked_area":
            return _auto_stacked_area(df, x_col, y_col,
                                      chart_spec.get("color"), title)
        elif chart_type == "treemap":
            return _auto_treemap(df, chart_spec.get("path", []),
                                 y_col, title)
        elif chart_type == "funnel":
            return _auto_funnel(df, chart_spec.get("columns", []), title)
        elif chart_type == "waterfall":
            return _auto_waterfall(df, x_col, y_col, title)
        else:
            return _auto_bar_chart(df, x_col, y_col, title)
    except Exception:
        return None


def _auto_line_chart(df, x_col, y_col, title):
    if x_col not in df.columns:
        return _empty_figure(f"📈 {title}", "X-axis column is missing.")

    df_copy = df.copy()

    # Numeric or count fallback for y-axis.
    if y_col and y_col in df_copy.columns:
        y_num = _as_numeric(df_copy[y_col])
        use_count = y_num.notna().sum() == 0
    else:
        use_count = True

    parsed_dates = pd.to_datetime(df_copy[x_col], errors="coerce")
    can_use_dates = parsed_dates.notna().sum() >= max(3, int(len(df_copy) * 0.6))

    if can_use_dates:
        df_copy[x_col] = parsed_dates
        if use_count:
            monthly = (
                df_copy.dropna(subset=[x_col])
                .groupby(pd.Grouper(key=x_col, freq="ME"))
                .size()
                .reset_index(name="value")
            )
        else:
            df_copy[y_col] = y_num
            monthly = (
                df_copy.dropna(subset=[x_col, y_col])
                .groupby(pd.Grouper(key=x_col, freq="ME"))[y_col]
                .sum()
                .reset_index(name="value")
            )
    else:
        if use_count:
            grouped = (
                df_copy[x_col].astype(str).value_counts().head(20).sort_index()
            )
            monthly = grouped.reset_index()
            monthly.columns = [x_col, "value"]
        else:
            df_copy[y_col] = y_num
            monthly = df_copy[[x_col, y_col]].dropna().sort_values(by=x_col).head(300)
            monthly = monthly.rename(columns={y_col: "value"})

    if monthly.empty:
        return _empty_figure(f"📈 {title}", "No usable points for line chart.")
    if len(monthly) < 3:
        return _empty_figure(f"📈 {title}", "Need at least 3 points for a trend chart.")

    value_num = pd.to_numeric(monthly["value"], errors="coerce").dropna()
    if value_num.nunique() < 2:
        return _empty_figure(f"📈 {title}", "Not enough variation for a trend chart.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly[x_col], y=monthly["value"],
        mode="lines+markers",
        name=(y_col.replace("_", " ").title() if y_col else "Count"),
        line=dict(color=THEME["accent_colors"][0], width=2.5),
        fill="tozeroy", fillcolor="rgba(0, 212, 255, 0.1)"
    ))
    _apply_theme(fig, f"📈 {title}")
    return fig


def _auto_bar_chart(df, x_col, y_col, title):
    if x_col not in df.columns:
        return _empty_figure(f"📊 {title}", "X-axis column is missing.")

    if y_col and y_col in df.columns:
        y_num = _as_numeric(df[y_col])
        if y_num.notna().sum() > 0:
            df_copy = df[[x_col]].copy()
            df_copy[y_col] = y_num
            agg = (
                df_copy.dropna(subset=[x_col, y_col])
                .groupby(x_col)[y_col]
                .sum()
                .sort_values(ascending=False)
                .head(15)
            )
        else:
            agg = df[x_col].astype(str).value_counts().head(15)
    else:
        agg = df[x_col].astype(str).value_counts().head(15)

    if agg.empty:
        return _empty_figure(f"📊 {title}", "No data available for bar chart.")
    if len(agg) < 2:
        return _empty_figure(f"📊 {title}", "Need at least 2 categories for bar chart.")

    fig = go.Figure(go.Bar(
        x=agg.index, y=agg.values,
        marker=dict(
            color=agg.values,
            colorscale=[[0, THEME["accent_colors"][2]], [1, THEME["accent_colors"][0]]]
        ),
        text=[f"{v:,.0f}" for v in agg.values], textposition="auto"
    ))
    _apply_theme(fig, f"📊 {title}")
    return fig


def _auto_histogram(df, x_col, title):
    if x_col not in df.columns:
        return _empty_figure(f"📉 {title}", "Column is missing.")

    x_num = _as_numeric(df[x_col])
    if x_num.notna().sum() > 1:
        x_values = x_num.dropna()
        fig = go.Figure(go.Histogram(
            x=x_values, nbinsx=30,
            marker_color=THEME["accent_colors"][0],
            marker_line_color=THEME["accent_colors"][3],
            marker_line_width=1, opacity=0.8
        ))
        _apply_theme(fig, f"📉 {title}")
        return fig

    counts = df[x_col].astype(str).value_counts().head(20)
    if counts.empty:
        return _empty_figure(f"📉 {title}", "No data available for distribution chart.")
    if len(counts) < 2:
        return _empty_figure(f"📉 {title}", "Need at least 2 categories for distribution chart.")

    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=THEME["accent_colors"][0],
        marker_line_color=THEME["accent_colors"][3],
        marker_line_width=1,
        opacity=0.8
    ))
    _apply_theme(fig, f"📉 {title}")
    return fig


def _auto_scatter(df, x_col, y_col, title):
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_figure(f"🔵 {title}", "Required columns are missing.")

    df_copy = df[[x_col, y_col]].copy()
    df_copy[x_col] = _as_numeric(df_copy[x_col])
    df_copy[y_col] = _as_numeric(df_copy[y_col])
    df_copy = df_copy.dropna(subset=[x_col, y_col])
    if len(df_copy) < 3:
        return _empty_figure(f"🔵 {title}", "Not enough numeric points for scatter plot.")
    if df_copy[x_col].nunique() < 2 or df_copy[y_col].nunique() < 2:
        return _empty_figure(f"🔵 {title}", "Not enough variation for scatter plot.")

    fig = px.scatter(
        df_copy, x=x_col, y=y_col,
        color_discrete_sequence=THEME["accent_colors"],
        opacity=0.7
    )
    _apply_theme(fig, f"🔵 {title}")
    return fig


def _auto_pie_chart(df, x_col, y_col, title):
    if x_col not in df.columns:
        return _empty_figure(f"🍩 {title}", "Category column is missing.")

    x_series = _truncate_categories(df[x_col].astype(str), top_n=10)

    if y_col == "count" or y_col is None or y_col not in df.columns:
        values = x_series.value_counts()
    else:
        y_num = _as_numeric(df[y_col])
        if y_num.notna().sum() == 0:
            values = x_series.value_counts()
        else:
            df_copy = pd.DataFrame({x_col: x_series, y_col: y_num}).dropna(subset=[y_col])
            values = df_copy.groupby(x_col)[y_col].sum().sort_values(ascending=False)

    if values.empty:
        return _empty_figure(f"🍩 {title}", "No values available for pie chart.")
    if len(values) < 2:
        return _empty_figure(f"🍩 {title}", "Need at least 2 categories for pie chart.")

    fig = go.Figure(go.Pie(
        labels=values.index, values=values.values,
        hole=0.45, marker=dict(colors=THEME["accent_colors"]),
        textinfo="label+percent"
    ))
    _apply_theme(fig, f"🍩 {title}")
    return fig


def _auto_box_chart(df, x_col, y_col, title):
    if x_col not in df.columns or y_col not in df.columns:
        return _empty_figure(f"📦 {title}", "Required columns are missing.")

    df_copy = df[[x_col, y_col]].copy()
    df_copy[y_col] = _as_numeric(df_copy[y_col])
    df_copy = df_copy.dropna(subset=[x_col, y_col])
    if df_copy.empty:
        return _empty_figure(f"📦 {title}", "No valid values available for box chart.")

    # Avoid unreadable plots with extremely high-cardinality categories.
    top_categories = df_copy[x_col].value_counts().head(20).index
    df_copy = df_copy[df_copy[x_col].isin(top_categories)]
    if df_copy[x_col].nunique() < 2:
        return _empty_figure(f"📦 {title}", "Need at least 2 categories for box chart.")

    fig = px.box(
        df_copy, x=x_col, y=y_col,
        color=x_col,
        color_discrete_sequence=THEME["accent_colors"]
    )
    _apply_theme(fig, f"📦 {title}")
    return fig


# ─── New Auto-Chart Types ─────────────────────────────────────────

def _auto_heatmap(df, columns, title):
    """Correlation heatmap for numeric columns."""
    available = [c for c in columns if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c])]
    if len(available) < 3:
        return _empty_figure(f"🔥 {title}", "Need at least 3 numeric columns.")

    corr = df[available].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0, "#FF6B6B"],
            [0.5, "#1A1D23"],
            [1.0, "#00D4FF"]
        ],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False
    ))
    _apply_theme(fig, f"🔥 {title}")
    fig.update_layout(height=max(400, 50 * len(available)))
    return fig


def _auto_stacked_area(df, x_col, y_col, color_col, title):
    """Stacked area chart showing composition over time."""
    if not x_col or x_col not in df.columns:
        return _empty_figure(f"📊 {title}", "Date column is missing.")
    if not y_col or y_col not in df.columns:
        return _empty_figure(f"📊 {title}", "Metric column is missing.")
    if not color_col or color_col not in df.columns:
        return _empty_figure(f"📊 {title}", "Category column is missing.")

    df_copy = df.copy()
    df_copy[x_col] = pd.to_datetime(df_copy[x_col], errors="coerce")
    df_copy[y_col] = _as_numeric(df_copy[y_col])
    df_copy = df_copy.dropna(subset=[x_col, y_col, color_col])

    if len(df_copy) < 5:
        return _empty_figure(f"📊 {title}", "Not enough data for stacked area.")

    # Truncate categories to top 6
    top_cats = df_copy[color_col].value_counts().head(6).index
    df_copy[color_col] = df_copy[color_col].apply(
        lambda v: v if v in top_cats else "Other")

    # Group by month + category
    df_copy["_period"] = df_copy[x_col].dt.to_period("M").dt.to_timestamp()
    grouped = (df_copy.groupby(["_period", color_col])[y_col]
               .sum().reset_index())

    if grouped.empty or grouped["_period"].nunique() < 3:
        return _empty_figure(f"📊 {title}", "Not enough time periods.")

    fig = px.area(
        grouped, x="_period", y=y_col, color=color_col,
        color_discrete_sequence=THEME["accent_colors"],
        groupnorm=None
    )
    _apply_theme(fig, f"📊 {title}")
    fig.update_xaxes(title="Period")
    fig.update_yaxes(title=y_col.replace("_", " ").title())
    return fig


def _auto_treemap(df, path_cols, value_col, title):
    """Hierarchical treemap chart."""
    available_paths = [c for c in path_cols if c in df.columns]
    if len(available_paths) < 1:
        return _empty_figure(f"🌳 {title}", "Need at least 1 category column.")

    df_copy = df.copy()
    if value_col and value_col in df_copy.columns:
        df_copy[value_col] = _as_numeric(df_copy[value_col])
        df_copy = df_copy.dropna(subset=available_paths + [value_col])
    else:
        df_copy = df_copy.dropna(subset=available_paths)
        value_col = None

    if len(df_copy) < 5:
        return _empty_figure(f"🌳 {title}", "Not enough data for treemap.")

    # Truncate high-cardinality categories
    for col in available_paths:
        if df_copy[col].nunique() > 20:
            top = df_copy[col].value_counts().head(15).index
            df_copy = df_copy[df_copy[col].isin(top)]

    try:
        fig = px.treemap(
            df_copy,
            path=[px.Constant("All")] + available_paths,
            values=value_col,
            color_discrete_sequence=THEME["accent_colors"],
        )
        _apply_theme(fig, f"🌳 {title}")
        fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
        return fig
    except Exception:
        return _empty_figure(f"🌳 {title}", "Could not build treemap.")


def _auto_funnel(df, columns, title):
    """Conversion funnel chart."""
    available = [c for c in columns if c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c])]
    if len(available) < 2:
        return _empty_figure(f"🔻 {title}", "Need at least 2 funnel stages.")

    stage_values = []
    for col in available:
        total = _as_numeric(df[col]).sum()
        if total > 0:
            stage_values.append((col.replace("_", " ").title(), float(total)))

    if len(stage_values) < 2:
        return _empty_figure(f"🔻 {title}", "Not enough data for funnel.")

    # Sort descending (largest stage first — typical funnel shape)
    stage_values.sort(key=lambda x: x[1], reverse=True)

    fig = go.Figure(go.Funnel(
        y=[s[0] for s in stage_values],
        x=[s[1] for s in stage_values],
        textinfo="value+percent initial+percent previous",
        marker=dict(
            color=THEME["accent_colors"][:len(stage_values)],
            line=dict(width=1, color=THEME["grid_color"])
        ),
        connector=dict(line=dict(color=THEME["grid_color"], width=1))
    ))
    _apply_theme(fig, f"🔻 {title}")
    return fig


def _auto_waterfall(df, x_col, y_col, title):
    """Waterfall chart showing cumulative contributions."""
    if not x_col or x_col not in df.columns:
        return _empty_figure(f"💧 {title}", "Category column is missing.")
    if not y_col or y_col not in df.columns:
        return _empty_figure(f"💧 {title}", "Value column is missing.")

    df_copy = df[[x_col, y_col]].copy()
    df_copy[y_col] = _as_numeric(df_copy[y_col])
    df_copy = df_copy.dropna(subset=[x_col, y_col])

    if len(df_copy) < 2:
        return _empty_figure(f"💧 {title}", "Not enough data for waterfall.")

    agg = df_copy.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(12)

    measures = ["relative"] * len(agg) + ["total"]
    x_labels = list(agg.index.astype(str)) + ["Total"]
    y_values = list(agg.values) + [agg.sum()]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        textposition="outside",
        text=[f"{v:,.0f}" for v in y_values],
        connector=dict(line=dict(color=THEME["grid_color"])),
        increasing=dict(marker=dict(color=THEME["accent_colors"][2])),
        decreasing=dict(marker=dict(color=THEME["accent_colors"][1])),
        totals=dict(marker=dict(color=THEME["accent_colors"][0]))
    ))
    _apply_theme(fig, f"💧 {title}")
    return fig


# ─── ML Visualization Charts ─────────────────────────────────────

def anomaly_overlay_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    anomaly_col: str = "is_anomaly",
    score_col: str = "anomaly_score"
) -> go.Figure:
    """Scatter plot with anomalies highlighted in contrasting color."""
    required = [x_col, y_col, anomaly_col]
    if any(c not in df.columns for c in required):
        return _empty_figure("🚨 Anomaly Detection",
                             "Required columns are missing.")

    df_copy = df.copy()
    df_copy[x_col] = _as_numeric(df_copy[x_col])
    df_copy[y_col] = _as_numeric(df_copy[y_col])
    df_copy = df_copy.dropna(subset=[x_col, y_col])
    if len(df_copy) < 5:
        return _empty_figure("🚨 Anomaly Detection", "Not enough data.")

    df_copy["_label"] = df_copy[anomaly_col].map(
        {True: "⚠ Anomaly", False: "Normal"})

    hover_data = [score_col] if score_col in df_copy.columns else None

    fig = px.scatter(
        df_copy, x=x_col, y=y_col, color="_label",
        color_discrete_map={"⚠ Anomaly": "#FF6B6B", "Normal": "#00D4FF"},
        opacity=0.7,
        hover_data=hover_data,
        size=score_col if score_col in df_copy.columns else None,
        size_max=14
    )
    _apply_theme(fig, f"🚨 Anomalies: {x_col.replace('_', ' ').title()} vs "
                      f"{y_col.replace('_', ' ').title()}")
    return fig


def cluster_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = "cluster"
) -> go.Figure:
    """2-D scatter colored by cluster assignment."""
    required = [x_col, y_col, cluster_col]
    if any(c not in df.columns for c in required):
        return _empty_figure("🎯 Customer Clusters",
                             "Required columns are missing.")

    df_copy = df.copy()
    df_copy[x_col] = _as_numeric(df_copy[x_col])
    df_copy[y_col] = _as_numeric(df_copy[y_col])
    df_copy = df_copy.dropna(subset=[x_col, y_col])
    if len(df_copy) < 5:
        return _empty_figure("🎯 Customer Clusters", "Not enough data.")

    df_copy["_cluster"] = "Cluster " + df_copy[cluster_col].astype(str)

    fig = px.scatter(
        df_copy, x=x_col, y=y_col, color="_cluster",
        color_discrete_sequence=THEME["accent_colors"],
        opacity=0.75
    )
    _apply_theme(fig, f"🎯 Clusters: {x_col.replace('_', ' ').title()} vs "
                      f"{y_col.replace('_', ' ').title()}")
    return fig


def trend_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame = None,
    date_col: str = "date",
    value_col: str = "value",
    title: str = "Trend & Forecast"
) -> go.Figure:
    """Line chart with historical trend + optional linear forecast overlay."""
    if date_col not in historical_df.columns or value_col not in historical_df.columns:
        return _empty_figure(f"📈 {title}", "Required columns are missing.")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_df[date_col],
        y=historical_df[value_col],
        mode="lines+markers",
        name="Actual",
        line=dict(color=THEME["accent_colors"][0], width=2.5),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.08)"
    ))

    # Forecast overlay
    if forecast_df is not None and not forecast_df.empty:
        fc = forecast_df[forecast_df.get("type", pd.Series(dtype=str)) == "forecast"]
        if fc.empty and "type" not in forecast_df.columns:
            fc = forecast_df
        if not fc.empty:
            fig.add_trace(go.Scatter(
                x=fc[date_col],
                y=fc[value_col],
                mode="lines+markers",
                name="Forecast",
                line=dict(color=THEME["accent_colors"][3], width=2.5,
                          dash="dash"),
                marker=dict(size=7, symbol="diamond")
            ))

    _apply_theme(fig, f"📈 {title}")
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title=value_col.replace("_", " ").title())
    return fig


def auto_generate_charts(df: pd.DataFrame, chart_specs: list,
                         max_charts: int = 12) -> list:
    """
    Generate all recommended charts for a dataset.
    Returns list of (title, figure) tuples.
    """
    charts = []
    seen_titles = set()
    seen_signatures = set()

    priority = {
        "line": 0,
        "stacked_area": 1,
        "bar": 2,
        "scatter": 3,
        "histogram": 4,
        "heatmap": 5,
        "funnel": 6,
        "treemap": 7,
        "waterfall": 8,
        "box": 9,
        "pie": 10,
    }
    ranked_specs = sorted(
        chart_specs,
        key=lambda s: priority.get(s.get("type", "bar"), 99),
    )

    for spec in ranked_specs:
        fig = auto_chart(df, spec)
        title = spec.get("title", "Chart")
        signature = (spec.get("type"), spec.get("x"), spec.get("y"))

        if (
            fig is not None
            and _figure_has_data(fig)
            and title not in seen_titles
            and signature not in seen_signatures
        ):
            charts.append((title, fig))
            seen_titles.add(title)
            seen_signatures.add(signature)
            if len(charts) >= max_charts:
                break

    return charts
