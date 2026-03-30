import pandas as pd
from typing import Dict, Any

try:
    from unsloth import FastLanguageModel
    import torch
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from llm.llm_client import LLMClient

# Global references for persistent VRAM caching
_model = None
_tokenizer = None

def load_unsloth_model(model_name="unsloth/phi-3-mini-4k-instruct", max_seq_length=2048):
    global _model, _tokenizer
    if not UNSLOTH_AVAILABLE:
        return False
    try:
        if _model is None:
            _model, _tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(_model)
        return True
    except Exception as e:
        print(f"Failed to load Unsloth model: {e}")
        return False

def generate_insights(dataset_name: str, kpis: Dict[str, Any], use_ollama: bool = False) -> str:
    """Generate 3-bullet executive summary from dataset KPIs."""
    
    prompt = f"Analyze the following dataset metadata for '{dataset_name}':\n"
    for k, v in kpis.items():
        if isinstance(v, float):
            prompt += f"- {k}: {v:.2f}\n"
        else:
            prompt += f"- {k}: {v}\n"
    prompt += "\nProvide exactly 3 bullet points with executive insights derived directly from the metrics. Do not invent numbers."

    if not use_ollama and load_unsloth_model():
        try:
            inputs = _tokenizer(
                [
                    f"<|system|>\nYou are an expert Business Intelligence AI. Keep it concise and insightful.<|end|>\n"
                    f"<|user|>\n{prompt}<|end|>\n"
                    f"<|assistant|>\n"
                ], return_tensors="pt").to("cuda")
            
            outputs = _model.generate(**inputs, max_new_tokens=200, use_cache=True)
            response = _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract just the assistant response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            return response
        except Exception as e:
            print(f"Unsloth generation failed: {e}")
            # Fall back to Ollama
            
    # Fallback to Ollama or Default text
    client = LLMClient()
    return client.generate_structured(
        prompt=prompt,
        system_prompt="You are an expert Data Analyst AI. Generate exactly 3 bullet points summarizing the most critical executive insights. No fluff.",
    )
