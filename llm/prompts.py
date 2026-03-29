"""
Prompt Templates Module
=======================
Centralized, production-quality prompt templates for each agent and use-case.
Uses structured prompt engineering for consistent, high-quality LLM outputs.
"""


class PromptTemplates:
    """Repository of prompt templates for the multi-agent system."""

    # ─── System Prompts ────────────────────────────────────────────

    SYSTEM_ANALYST = (
        "You are a senior data analyst AI agent working in a multi-agent decision "
        "intelligence platform. You analyze structured business and technology data "
        "to generate actionable insights. Be specific, cite data points, and provide "
        "concrete recommendations. Use clear markdown formatting with headers, "
        "bullet points, and bold for emphasis. Always ground your analysis in the "
        "actual data provided."
    )

    SYSTEM_ANALYST_STRICT = (
        "You are a senior data analyst. You MUST use only the provided dataset context. "
        "Do not invent facts, names, studies, documents, laws, or external references. "
        "If a question cannot be answered from the provided context, state: "
        "'Insufficient evidence in provided dataset context.' Then list exactly which "
        "additional columns or data are needed. "
        "Output strict markdown with short, clear sections and data-backed bullets."
    )

    SYSTEM_STRATEGIST = (
        "You are a chief strategy officer AI agent. You synthesize insights from "
        "multiple analyst agents (Sales, Marketing, Customer, and Tech) into a "
        "unified strategic recommendation. Focus on cross-functional synergies, "
        "risk assessment, and prioritized action items. Your output should be "
        "executive-ready with clear sections and data-backed conclusions."
    )

    # ─── Sales Agent Prompts ───────────────────────────────────────

    @staticmethod
    def sales_analysis(kpis: str, data_summary: str) -> str:
        return f"""Analyze the following sales performance data and provide actionable insights.

## Key Performance Indicators
{kpis}

## Data Summary
{data_summary}

## Required Analysis
Please provide a comprehensive sales analysis covering:

1. **Revenue Performance**: Overall revenue health, trends, and notable patterns.
2. **Product Analysis**: Which products/categories are driving growth? Which are underperforming?
3. **Regional Breakdown**: Compare performance across regions. Identify growth opportunities.
4. **Profit Margins**: Analyze margin distribution. Flag any concerning trends.
5. **Seasonal Patterns**: Identify any seasonality in the data.
6. **Key Risks**: What risks or warning signs do you see?
7. **Recommendations**: Provide 3-5 specific, prioritized action items with expected impact.

Be specific with numbers and percentages. Reference actual data points."""

    # ─── Marketing Agent Prompts ───────────────────────────────────

    @staticmethod
    def marketing_analysis(kpis: str, data_summary: str) -> str:
        return f"""Analyze the following marketing campaign data and provide strategic insights.

## Key Performance Indicators
{kpis}

## Data Summary
{data_summary}

## Required Analysis
Please provide a comprehensive marketing analysis covering:

1. **Campaign ROI**: Which campaign types deliver the best return on investment?
2. **Channel Effectiveness**: Compare channels by CTR, conversion rate, and ROI.
3. **Cost Analysis**: Evaluate CPC and cost efficiency across channels.
4. **Conversion Funnel**: Analyze the impression → click → conversion pipeline.
5. **Budget Optimization**: Where should budget be reallocated for maximum impact?
6. **Underperformers**: Identify campaigns or channels that should be scaled back.
7. **Recommendations**: Provide 3-5 specific, data-backed marketing recommendations.

Use specific metrics and comparisons in your analysis."""

    # ─── Customer Agent Prompts ────────────────────────────────────

    @staticmethod
    def customer_analysis(kpis: str, data_summary: str, cluster_info: str = "") -> str:
        cluster_section = ""
        if cluster_info:
            cluster_section = f"""
## ML Clustering Results
{cluster_info}

Interpret the clustering results and explain what each customer cluster represents.
"""
        return f"""Analyze the following customer data and provide segmentation insights.

## Key Performance Indicators
{kpis}

## Data Summary
{data_summary}
{cluster_section}
## Required Analysis
Please provide a comprehensive customer analysis covering:

1. **Customer Segments**: Characterize each segment by value, behavior, and risk profile.
2. **Churn Risk Assessment**: Which segments are at highest risk? What are the warning signs?
3. **Lifetime Value Distribution**: Analyze LTV across segments and industries.
4. **Satisfaction & Engagement**: How do satisfaction and engagement scores correlate with retention?
5. **High-Value Customers**: What defines your most valuable customers?
6. **At-Risk Customers**: Profile the at-risk segment and suggest intervention strategies.
7. **Recommendations**: Provide 3-5 specific customer retention and growth strategies.

Ground your analysis in the actual data metrics provided."""

    # ─── Tech/GitHub Agent Prompts ─────────────────────────────────

    @staticmethod
    def tech_analysis(kpis: str, data_summary: str, anomaly_info: str = "") -> str:
        anomaly_section = ""
        if anomaly_info:
            anomaly_section = f"""
## Anomaly Detection Results
{anomaly_info}

Interpret the detected anomalies and explain their significance.
"""
        return f"""Analyze the following GitHub repository / technology data and provide insights.

## Key Performance Indicators
{kpis}

## Data Summary
{data_summary}
{anomaly_section}
## Required Analysis
Please provide a comprehensive technology landscape analysis covering:

1. **Language Trends**: Which programming languages are most popular? Any emerging trends?
2. **Repository Quality**: What characterizes high-quality repos (stars, CI/CD, docs)?
3. **Community Health**: Analyze contributor activity and issue management.
4. **Open Source Ecosystem**: What patterns emerge in the open-source landscape?
5. **Technology Adoption**: Which topics/areas show the most activity?
6. **Quality Correlation**: How do CI/CD and documentation relate to repo success?
7. **Recommendations**: Provide 3-5 insights for technology strategy and investment.

Use specific data points and comparisons in your analysis."""

    # ─── Strategy Agent Prompts ────────────────────────────────────

    @staticmethod
    def strategic_synthesis(
        sales_report: str,
        marketing_report: str,
        customer_report: str,
        tech_report: str
    ) -> str:
        return f"""You are the Chief Strategy Agent. Synthesize the following analyst reports
into a unified strategic recommendation.

## Sales Analyst Report
{sales_report}

## Marketing Analyst Report
{marketing_report}

## Customer Analyst Report
{customer_report}

## Tech Analyst Report
{tech_report}

## Required Strategic Output

Create an executive-level strategic recommendation covering:

1. **Executive Summary**: 2-3 sentence overview of the organization's position.
2. **Cross-Functional Insights**: Identify synergies and conflicts between departments.
3. **Top Opportunities** (ranked by impact):
   - What are the 3 biggest growth opportunities?
   - How should resources be prioritized?
4. **Critical Risks**:
   - What are the top risks across all areas?
   - What mitigation strategies are recommended?
5. **Strategic Recommendations** (prioritized action plan):
   - Immediate actions (0-30 days)
   - Short-term initiatives (30-90 days)
   - Long-term strategy (90-365 days)
6. **Resource Allocation**: How should budget and talent be distributed?
7. **Success Metrics**: What KPIs should be tracked to measure progress?

Make this actionable and specific. Reference data from all analyst reports."""

    # ─── Single Insight Prompts ────────────────────────────────────

    @staticmethod
    def quick_insight(data_summary: str, question: str) -> str:
        return f"""Answer the business question using only the dataset context.

## Dataset Context
{data_summary}

## Question
{question}

## Output Requirements
Return markdown with exactly these sections in order:
1. ## Direct Answer
2. ## Evidence from Data
3. ## Risks or Gaps
4. ## Recommended Next Checks

Rules:
- Cite concrete values from the context (numbers, percentages, counts, or column names).
- Do not use external references or unrelated examples.
- If context is insufficient, say: Insufficient evidence in provided dataset context.
- Keep total length under 350 words."""

    @staticmethod
    def repair_insight(
        data_summary: str,
        question: str,
        previous_output: str
    ) -> str:
        return f"""Rewrite the prior answer to remove hallucinations and formatting issues.

## Dataset Context
{data_summary}

## Question
{question}

## Previous Output To Repair
{previous_output}

## Repair Requirements
- Keep only claims directly supported by Dataset Context.
- Remove apologies, meta-text, and unrelated narrative.
- Return markdown with sections:
  - ## Direct Answer
  - ## Evidence from Data
  - ## Risks or Gaps
  - ## Recommended Next Checks
- If evidence is missing, explicitly write: Insufficient evidence in provided dataset context."""

    @staticmethod
    def data_summary_prompt(data_summary: str) -> str:
        return f"""Provide a concise executive summary of the following dataset.

## Data Overview
{data_summary}

Summarize:
1. What this data represents
2. Key observations (3-5 bullet points)
3. Data quality assessment
4. Potential areas for deeper analysis

Keep it concise and actionable."""

    @staticmethod
    def recommendation_prompt(all_insights: str) -> str:
        return f"""Based on all the following analysis insights, generate a final set of
prioritized recommendations.

## All Analysis Insights
{all_insights}

## Output Format
Generate exactly 5-7 recommendations, each with:
- **Title**: Clear action item
- **Priority**: High / Medium / Low
- **Impact**: Expected business impact
- **Effort**: Implementation effort (Low/Medium/High)
- **Details**: 2-3 sentences explaining the recommendation

Rank by priority and expected impact."""

    # ─── Dynamic / Universal Analysis Prompt ───────────────────

    @staticmethod
    def dynamic_analysis(
        detected_type: str,
        context: str,
        ml_signals: str = ""
    ) -> str:
        """
        Universal analysis prompt that works with any auto-detected dataset.
        This is the primary entrypoint for analyzing unknown datasets.
        """
        ml_section = ""
        if ml_signals:
            ml_section = f"\n## ML Signal Analysis\n{ml_signals}\nInterpret these ML signals in context.\n"

        return f"""Analyze the following {detected_type} dataset and provide expert insights.

## Dataset Context
{context}
{ml_section}
## Required Analysis
Provide a comprehensive analysis covering:

1. **Overview**: What does this data tell us? Summarize the key story.
2. **Top Findings**: 3-5 most important patterns or insights found in the data.
3. **Performance Metrics**: Analyze the key metrics — what's performing well vs poorly?
4. **Trends & Patterns**: Any notable trends, correlations, or seasonal patterns?
5. **Risks & Concerns**: What warning signs or risks are visible in the data?
6. **Recommendations**: Provide 3-5 specific, data-backed action items.

Be specific with numbers. Reference actual data points from the context above.
Keep the response focused and concise.

Do not use external facts. If context is insufficient, state exactly what is missing."""
