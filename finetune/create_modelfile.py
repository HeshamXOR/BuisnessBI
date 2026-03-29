"""
Create Ollama Modelfile
========================
Generates an Ollama Modelfile for the fine-tuned model
so it can be used directly with `ollama create`.

Two approaches supported:
  1. GGUF model file (from converted fine-tuned weights)
  2. System prompt + parameter tuning on existing model
"""

import os
import argparse


def create_modelfile(
    approach: str = "prompt",
    base_model: str = "phi3:mini",
    gguf_path: str = "finetune/output/model.gguf",
    output_path: str = "finetune/Modelfile"
):
    """
    Create an Ollama Modelfile.

    Args:
        approach: 'prompt' (system prompt tuning) or 'gguf' (custom weights).
        base_model: Base Ollama model name (for prompt approach).
        gguf_path: Path to GGUF file (for gguf approach).
        output_path: Where to save the Modelfile.
    """
    if approach == "gguf":
        modelfile = _create_gguf_modelfile(gguf_path)
    else:
        modelfile = _create_prompt_modelfile(base_model)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modelfile)

    print(f"✅ Modelfile created: {output_path}")
    print(f"\nTo create the Ollama model, run:")
    print(f"  ollama create business-analyst -f {output_path}")
    print(f"\nThen update .env:")
    print(f"  OLLAMA_MODEL=business-analyst")

    return output_path


def _create_prompt_modelfile(base_model: str) -> str:
    """Create a Modelfile with optimized system prompt and parameters."""
    return f"""# Business Data Analysis Model
# Based on {base_model} with specialized business analysis prompt
# Optimized for fast inference on RTX A6000

FROM {base_model}

# System prompt specialized for business data analysis
SYSTEM \"\"\"You are a senior business data analyst AI agent. Your role is to:

1. Analyze structured business datasets (sales, marketing, customer, financial, HR, inventory)
2. Compute and interpret KPIs with precision
3. Identify trends, patterns, anomalies, and correlations
4. Provide specific, data-backed recommendations with expected impact
5. Use clear markdown formatting: headers, tables, bullet points, bold metrics

Rules:
- Always cite specific numbers from the data
- Rank recommendations by impact and effort
- Flag risks and warning signs
- Keep analysis concise and actionable
- Use markdown tables for comparisons
- Bold key metrics and findings\"\"\"

# Optimized parameters for fast, focused analysis
PARAMETER temperature 0.2
PARAMETER top_p 0.85
PARAMETER top_k 40
PARAMETER num_predict 2048
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
PARAMETER stop "---"
PARAMETER stop "## END"
"""


def _create_gguf_modelfile(gguf_path: str) -> str:
    """Create a Modelfile from a custom GGUF model file."""
    return f"""# Fine-Tuned Business Data Analysis Model
# Custom trained on financial/business datasets

FROM {gguf_path}

SYSTEM \"\"\"You are a senior business data analyst AI agent specialized in:
- Sales, marketing, customer, and financial data analysis
- KPI computation and trend identification
- Actionable business recommendations
- Structured markdown output with tables and metrics

Always be specific with numbers and rank recommendations by impact.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.85
PARAMETER num_predict 2048
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Ollama Modelfile")
    parser.add_argument(
        "--approach", choices=["prompt", "gguf"], default="prompt",
        help="'prompt' for system prompt tuning, 'gguf' for custom weights"
    )
    parser.add_argument("--base-model", default="phi3:mini")
    parser.add_argument("--gguf-path", default="finetune/output/model.gguf")
    parser.add_argument("--output", default="finetune/Modelfile")
    args = parser.parse_args()

    create_modelfile(
        approach=args.approach,
        base_model=args.base_model,
        gguf_path=args.gguf_path,
        output_path=args.output
    )
