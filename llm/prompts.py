"""
Prompt Templates Module
=======================
Centralized, production-quality prompt templates for each agent and use-case.
Uses structured prompt engineering for consistent, high-quality LLM outputs.
"""


class PromptTemplates:
    """Repository of prompt templates for the multi-agent system."""

    MARKDOWN_OUTPUT_RULES = (
        "Output markdown only. Keep headings short. Use bullet points instead of long paragraphs. "
        "Every key claim must reference at least one concrete value, rate, count, column name, or ranking from the provided context. "
        "If evidence is weak or missing, say 'Insufficient evidence in provided dataset context.'"
    )

    ANALYST_SECTION_SCHEMA = (
        "Return markdown with these sections in order:\n"
        "## Executive Summary\n"
        "## Evidence Snapshot\n"
        "## Key Findings\n"
        "## Risks and Watchouts\n"
        "## Recommended Actions\n"
        "## Additional Data That Would Help"
    )

    # ─── System Prompts ────────────────────────────────────────────

    SYSTEM_ANALYST = (
        "You are a senior data analyst AI agent working in a multi-agent decision "
        "intelligence platform. You analyze structured business and technology data "
        "to generate actionable insights. Be specific, cite data points, and provide "
        "concrete recommendations. Use clean markdown formatting and concise sections. "
        "Always ground your analysis in the actual data provided. "
        + MARKDOWN_OUTPUT_RULES
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
        "executive-ready with clear sections and data-backed conclusions. "
        + MARKDOWN_OUTPUT_RULES
    )

    @staticmethod
    def _analyst_prompt(
        title: str,
        kpis: str,
        data_summary: str,
        required_analysis: str,
        extra_context: str = "",
    ) -> str:
        extra = f"\n## Additional Context\n{extra_context}\n" if extra_context else ""
        return f"""{title}

## KPI Snapshot
{kpis}

## Dataset Summary
{data_summary}
{extra}
## Required Analysis
{required_analysis}

## Output Format
{PromptTemplates.ANALYST_SECTION_SCHEMA}

Rules:
- Keep the Executive Summary to 2-4 bullets.
- In Evidence Snapshot, list 4-6 data-backed bullets only.
- In Recommended Actions, provide 3-5 actions with priority tags such as High / Medium / Low.
- Avoid generic phrases like 'optimize operations' unless you explain exactly what to change.
- Do not use external facts, benchmarks, or assumptions beyond the provided context.
"""

    # ─── Sales Agent Prompts ───────────────────────────────────────

    @staticmethod
    def sales_analysis(kpis: str, data_summary: str) -> str:
        required = (
            "1. Revenue performance: explain overall revenue health, growth, concentration, and volatility.\n"
            "2. Product and category mix: identify top and underperforming products/categories.\n"
            "3. Regional or channel breakdown: compare performance across major groupings.\n"
            "4. Margin and discount behavior: flag any concerning margin compression or over-discounting.\n"
            "5. Trend signals: explain seasonality, acceleration, slowdown, or instability if present.\n"
            "6. Risks and action plan: include specific operational or commercial next steps."
        )
        return PromptTemplates._analyst_prompt(
            "Analyze the sales dataset and produce an operator-ready revenue report.",
            kpis,
            data_summary,
            required,
        )

    # ─── Marketing Agent Prompts ───────────────────────────────────

    @staticmethod
    def marketing_analysis(kpis: str, data_summary: str) -> str:
        required = (
            "1. ROI and efficiency: identify the best and worst campaign/channel outcomes.\n"
            "2. Funnel quality: explain where the biggest drop-offs likely occur.\n"
            "3. Cost structure: compare spend, CPC, CPM, and conversion efficiency.\n"
            "4. Reallocation opportunities: be explicit about what to scale up or down.\n"
            "5. Risks and action plan: include 3-5 tactical moves with expected upside."
        )
        return PromptTemplates._analyst_prompt(
            "Analyze the marketing dataset and produce a performance allocation report.",
            kpis,
            data_summary,
            required,
        )

    # ─── Customer Agent Prompts ────────────────────────────────────

    @staticmethod
    def customer_analysis(kpis: str, data_summary: str, cluster_info: str = "") -> str:
        required = (
            "1. Customer segmentation: characterize segments or clusters by value, behavior, and risk.\n"
            "2. Churn and retention: identify at-risk cohorts and the most useful intervention levers.\n"
            "3. Lifetime value and engagement: explain which variables appear tied to stronger customer value.\n"
            "4. High-value vs at-risk profiles: contrast them clearly.\n"
            "5. Action plan: provide 3-5 retention or growth actions with target cohorts."
        )
        return PromptTemplates._analyst_prompt(
            "Analyze the customer dataset and produce a segmentation and retention report.",
            kpis,
            data_summary,
            required,
            extra_context=cluster_info,
        )

    # ─── Tech/GitHub Agent Prompts ─────────────────────────────────

    @staticmethod
    def tech_analysis(kpis: str, data_summary: str, anomaly_info: str = "") -> str:
        required = (
            "1. Technology landscape: explain the main language/framework/topic patterns.\n"
            "2. Repository quality: identify what seems associated with repo success or health.\n"
            "3. Community and maintenance: comment on contributors, issues, documentation, and CI/CD.\n"
            "4. Outliers/anomalies: explain whether any unusual repositories matter strategically.\n"
            "5. Action plan: provide 3-5 recommendations for tech investment or engineering focus."
        )
        return PromptTemplates._analyst_prompt(
            "Analyze the technology dataset and produce a platform health and ecosystem report.",
            kpis,
            data_summary,
            required,
            extra_context=anomaly_info,
        )

    # ─── Strategy Agent Prompts ────────────────────────────────────

    @staticmethod
    def strategic_synthesis(
        sales_report: str, marketing_report: str, customer_report: str, tech_report: str
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

## Output Format
Return markdown with these sections in order:
## Executive Summary
## Cross-Functional Signals
## Top Opportunities
## Critical Risks
## Prioritized Action Plan
## Resource Allocation Guidance
## Success Metrics

Rules:
- Top Opportunities: exactly 3 ranked items.
- Critical Risks: exactly 3 ranked items.
- Prioritized Action Plan: split into Immediate (0-30 days), Short Term (30-90 days), Longer Term (90-365 days).
- Success Metrics: 5-8 KPI bullets max.
- Reference evidence from at least 2 analyst reports in each major section.
- Keep it specific, board-ready, and under 900 words."""

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
    def repair_insight(data_summary: str, question: str, previous_output: str) -> str:
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
Return markdown with one top-level section:
## Prioritized Recommendations

Generate exactly 5 recommendations. For each one use this template:
### Recommendation <number>: <short title>
- Priority: High / Medium / Low
- Impact: <specific business outcome>
- Effort: Low / Medium / High
- Why this matters: <1-2 bullets grounded in the provided insights>
- What to do next: <1-2 bullets with concrete execution steps>

Rules:
- Rank by priority and expected business impact.
- Avoid repeating the same recommendation with different wording.
- Keep the whole output under 700 words."""

    # ─── Dynamic / Universal Analysis Prompt ───────────────────

    @staticmethod
    def dynamic_analysis(detected_type: str, context: str, ml_signals: str = "") -> str:
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
## Output Format
{PromptTemplates.ANALYST_SECTION_SCHEMA}

## Required Analysis
1. Explain the business story of the dataset in plain language.
2. List the most important patterns, rankings, changes, or anomalies.
3. Separate what looks healthy from what looks weak or risky.
4. Turn the evidence into 3-5 specific actions.
5. State what additional columns or history would improve confidence.

Rules:
- Use the detected dataset type only as context, not as a license to invent missing fields.
- Be specific with numbers and column names from the context above.
- Keep the response focused and concise.
- Do not use external facts. If context is insufficient, state exactly what is missing."""
