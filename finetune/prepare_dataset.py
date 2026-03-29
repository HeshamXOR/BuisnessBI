"""
Dataset Preparation Module
============================
Downloads and formats HuggingFace datasets for fine-tuning
a business data analysis LLM.

Datasets:
  1. Sujet-Finance-Instruct-177k  — Financial instruction pairs
  2. FinGPT sentiment             — Financial sentiment analysis
  3. Financial QA                 — Business Q&A pairs
  4. Custom Business Analysis     — Auto-generated from our own data

Output: JSONL files in Ollama/Unsloth training format.
"""

import json
import os
from typing import Dict, List, Optional


def _clean_text(value) -> str:
    """Normalize values pulled from dataset rows."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def _first_non_empty(item: dict, keys: List[str]) -> str:
    """Return the first non-empty field from a dataset item."""
    for key in keys:
        value = _clean_text(item.get(key, ""))
        if value:
            return value
    return ""


def _validate_dataset_output(name: str, ds, output_path: str, count: int) -> None:
    """Fail fast when a dataset converts to zero training rows."""
    if count > 0:
        return

    columns = getattr(ds, "column_names", [])
    raise ValueError(
        f"{name} produced 0 training samples. "
        f"Detected columns: {columns}. Output file: {output_path}"
    )


def download_and_prepare_all(output_dir: str = "finetune/data") -> Dict[str, str]:
    """
    Download all fine-tuning datasets and convert to training JSONL.

    Args:
        output_dir: Directory to save prepared datasets.

    Returns:
        Dict mapping dataset name to output file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install datasets: pip install datasets\n"
            "This is required for downloading HuggingFace datasets."
        )

    results = {}
    failures = {}

    # ─── 1. Sujet Finance Instruct ─────────────────────────────
    print("📥 Downloading Sujet-Finance-Instruct-177k...")
    try:
        ds = load_dataset("Sujet-AI/Sujet-Finance-Instruct-177k", split="train")
        output_path = os.path.join(output_dir, "finance_instruct.jsonl")
        count = _convert_instruct_dataset(ds, output_path, max_samples=50000)
        _validate_dataset_output("finance_instruct", ds, output_path, count)
        results["finance_instruct"] = output_path
        print(f"   ✅ Saved {count} samples → {output_path}")
    except Exception as e:
        failures["finance_instruct"] = str(e)
        print(f"   ⚠️ Failed: {e}")

    # ─── 2. FinGPT Sentiment ───────────────────────────────────
    print("📥 Downloading FinGPT Sentiment...")
    try:
        ds = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
        output_path = os.path.join(output_dir, "finance_sentiment.jsonl")
        count = _convert_sentiment_dataset(ds, output_path, max_samples=30000)
        _validate_dataset_output("finance_sentiment", ds, output_path, count)
        results["finance_sentiment"] = output_path
        print(f"   ✅ Saved {count} samples → {output_path}")
    except Exception as e:
        failures["finance_sentiment"] = str(e)
        print(f"   ⚠️ Failed: {e}")

    # ─── 3. Financial QA ───────────────────────────────────────
    print("📥 Downloading Financial QA...")
    try:
        ds = load_dataset("virattt/financial-qa-10K", split="train")
        output_path = os.path.join(output_dir, "financial_qa.jsonl")
        count = _convert_qa_dataset(ds, output_path, max_samples=10000)
        _validate_dataset_output("financial_qa", ds, output_path, count)
        results["financial_qa"] = output_path
        print(f"   ✅ Saved {count} samples → {output_path}")
    except Exception as e:
        failures["financial_qa"] = str(e)
        print(f"   ⚠️ Failed: {e}")

    # ─── 4. Custom Business Analysis Pairs ─────────────────────
    print("🔧 Generating custom business analysis training data...")
    output_path = os.path.join(output_dir, "business_analysis_custom.jsonl")
    count = _generate_business_analysis_pairs(output_path)
    results["business_analysis_custom"] = output_path
    print(f"   ✅ Generated {count} samples → {output_path}")

    print(f"\n📊 Total datasets prepared: {len(results)}")

    if failures:
        failure_lines = [f"- {name}: {message}" for name, message in failures.items()]
        raise RuntimeError(
            "One or more datasets failed during preparation:\n"
            + "\n".join(failure_lines)
        )

    return results


def _convert_instruct_dataset(ds, output_path: str, max_samples: int = 50000) -> int:
    """Convert instruction-style dataset to chat JSONL format."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_samples:
                break

            system_prompt = _first_non_empty(
                item, ["system_prompt", "system", "context"]
            )
            instruction = _first_non_empty(
                item, ["user_prompt", "instruction", "input", "inputs", "question"]
            )
            output = _first_non_empty(
                item, ["answer", "output", "response", "assistant", "target"]
            )

            if instruction and output:
                entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                            or (
                                "You are a senior business data analyst. Analyze data, "
                                "compute metrics, and provide actionable insights. "
                                "Be specific with numbers and recommendations."
                            ),
                        },
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": output},
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    return count


def _convert_sentiment_dataset(ds, output_path: str, max_samples: int = 30000) -> int:
    """Convert sentiment dataset to analysis-style JSONL."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_samples:
                break

            text = item.get("input", item.get("text", ""))
            label = item.get("output", item.get("label", ""))

            if text and label:
                entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a financial sentiment analyst. Analyze the sentiment "
                                "of financial text and explain your reasoning."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Analyze the sentiment of this financial text:\n\n{text}",
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"**Sentiment: {label}**\n\n"
                                f"The text expresses a {label.lower()} sentiment based on "
                                f"the financial indicators and language used."
                            ),
                        },
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    return count


def _convert_qa_dataset(ds, output_path: str, max_samples: int = 10000) -> int:
    """Convert Q&A dataset to chat JSONL format."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_samples:
                break

            question = item.get("question", item.get("input", ""))
            answer = item.get("answer", item.get("output", ""))
            context = item.get("context", "")

            if question and answer:
                user_msg = question
                if context:
                    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

                entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a business data analyst assistant. Answer questions "
                                "about financial and business data accurately and concisely."
                            ),
                        },
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": answer},
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
    return count


def _generate_business_analysis_pairs(output_path: str) -> int:
    """
    Generate custom instruction pairs for business data analysis.
    These teach the model the specific output format we want.
    """
    pairs = [
        {
            "instruction": "Analyze this sales KPI data: Total Revenue: $4,521,890, Avg Order: $452, Top Region: North America (42%), Growth Rate: 12.3% MoM",
            "response": "## Sales Analysis\n\n**Revenue Health**: Total revenue of $4.52M is strong. The 12.3% MoM growth indicates healthy trajectory.\n\n**Key Findings**:\n- Average order value of $452 suggests mid-market positioning\n- North America drives 42% of revenue — significant concentration risk\n- Month-over-month growth of 12.3% is excellent but needs monitoring for sustainability\n\n**Recommendations**:\n1. **Diversify geographically** — reduce NA dependency below 40%\n2. **Increase AOV** — target $500+ through bundling and upsells\n3. **Sustain growth** — identify the specific drivers behind 12.3% MoM",
        },
        {
            "instruction": "A dataset has columns: customer_id, segment, lifetime_value, churn_risk, satisfaction_score. What type of analysis should I perform?",
            "response": "## Recommended Analysis for Customer Dataset\n\n**Dataset Type**: Customer Analytics / CRM Data\n\n**Key Analyses**:\n1. **Segmentation Profiling** — characterize each segment by LTV, churn risk, satisfaction\n2. **Churn Prediction** — identify churn risk drivers (satisfaction < threshold?)\n3. **LTV Distribution** — analyze value concentration across segments\n4. **Retention Strategy** — cross-reference satisfaction × churn risk for intervention targeting\n5. **Cohort Analysis** — if temporal data available, track segment evolution\n\n**ML Suggestions**: K-Means clustering on numeric features, anomaly detection on LTV",
        },
        {
            "instruction": "Marketing data shows: Email ROI: 340%, Social ROI: 89%, PPC ROI: 156%, Content ROI: 210%. Budget is $100K/month. Recommend budget allocation.",
            "response": "## Budget Allocation Recommendation\n\n**Current Performance Ranking**: Email (340%) > Content (210%) > PPC (156%) > Social (89%)\n\n**Recommended Monthly Allocation** ($100K):\n\n| Channel | Current | Recommended | Rationale |\n|---------|---------|-------------|----------|\n| Email | $25K | **$35K (+40%)** | Highest ROI — scale aggressively |\n| Content | $25K | **$30K (+20%)** | Strong ROI, long-term compounding |\n| PPC | $25K | **$25K (flat)** | Solid performer, maintain |\n| Social | $25K | **$10K (-60%)** | Lowest ROI — reduce, focus on brand only |\n\n**Expected Impact**: Shifting $15K from Social to Email/Content should increase total return by ~$30K/month based on current ROI rates.",
        },
        {
            "instruction": "Given a CSV with columns: date, product, category, region, revenue, units_sold, cost, profit_margin — compute the most important KPIs.",
            "response": "## Key Performance Indicators\n\n**Revenue KPIs**:\n- Total Revenue (sum of revenue)\n- Average Revenue per Transaction (mean revenue)\n- Revenue by Region (groupby region → sum revenue)\n- Revenue Growth Rate (compare periods)\n\n**Profitability KPIs**:\n-  Average Profit Margin (mean profit_margin)\n- Total Profit (revenue - cost, summed)\n- Margin by Category (groupby category → mean profit_margin)\n\n**Volume KPIs**:\n- Total Units Sold (sum units_sold)\n- Average Order Size (mean units_sold)\n- Top Products by Volume (groupby product → sum units_sold → top N)\n\n**Efficiency KPIs**:\n- Cost-to-Revenue Ratio (sum cost / sum revenue)\n- Revenue per Unit (sum revenue / sum units_sold)",
        },
        {
            "instruction": "I uploaded an HR dataset with employee_id, department, salary, hire_date, performance_score, tenure_years, turnover. Analyze it.",
            "response": "## HR Dataset Analysis\n\n**Dataset Type**: Human Resources / Workforce Analytics\n\n**Auto-Detected Columns**:\n- 📅 Date: hire_date\n- 💰 Monetary: salary\n- 📊 Numeric: performance_score, tenure_years, turnover\n- 📁 Categorical: department, employee_id\n\n**Key Analysis Areas**:\n1. **Compensation Analysis**: Salary distribution by department, pay equity assessment\n2. **Performance Patterns**: Correlation between tenure and performance scores\n3. **Turnover Risk**: Identify departments with highest turnover rates\n4. **Tenure Analysis**: Average tenure by department, retention trends\n5. **Hiring Trends**: Hire date patterns, seasonal hiring cycles\n\n**Immediate Insights to Look For**:\n- Departments with high salary but low performance (overpaying?)\n- Departments with high turnover + low salary (underpaying?)\n- Performance score vs tenure correlation (do long-tenured employees perform better?)",
        },
    ]

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a senior data analyst AI. You analyze business datasets, "
                            "compute KPIs, identify patterns, and provide actionable "
                            "recommendations. Use markdown formatting with headers, tables, "
                            "and bullet points. Be specific with numbers."
                        ),
                    },
                    {"role": "user", "content": pair["instruction"]},
                    {"role": "assistant", "content": pair["response"]},
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    return count


if __name__ == "__main__":
    download_and_prepare_all()
