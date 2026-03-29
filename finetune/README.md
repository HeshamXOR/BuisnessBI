# Fine-Tuning: Business Data Analysis LLM

## Quick Start (No Training Required)

### Option A: Load Pre-Tuned Weights From Hugging Face (Skip Training)

Uploaded model artifacts:

- `HeshamXOR/business-analyst-phi3-mini-lora`
- `HeshamXOR/business-analyst-phi3-mini-merged`

Download directly into this project (no re-training required):

```bash
export HF_TOKEN=your_hf_token

# Download merged model (recommended)
python finetune/hf_hub.py quick-download --which merged

# Or download LoRA adapter only
python finetune/hf_hub.py quick-download --which lora
```

You can also download any Hub repo explicitly:

```bash
python finetune/hf_hub.py download \
    --repo-id HeshamXOR/business-analyst-phi3-mini-merged \
    --local-dir finetune/output/merged_model
```

Then continue from deployment/load steps in the notebook (skip Step 2 training cells).

---

The fastest way to get a specialized model — create a custom Ollama model
with an optimized system prompt:

```bash
# Generate the Modelfile
python finetune/create_modelfile.py --approach prompt --base-model phi3:mini

# Create the model in Ollama
ollama create business-analyst -f finetune/Modelfile

# Update .env
# OLLAMA_MODEL=business-analyst
```

This takes 30 seconds and gives you a tuned model immediately.

---

## Full Fine-Tuning (QLoRA on ThunderCompute RTX A6000)

### Step 1: Install Dependencies

```bash
# 1. Install core fine-tuning packages
pip install datasets>=2.14.0 trl>=0.12.0 peft>=0.13.0 \
    transformers>=4.46.0 accelerate>=1.2.0 bitsandbytes>=0.45.0

# 2. Install Unsloth (pick ONE):

# Option A — Simple install (auto-detects your torch + CUDA):
pip install unsloth

# Option B — Ampere-optimized for ThunderCompute A6000 (recommended):
pip install "unsloth[cu128-ampere-torch290] @ git+https://github.com/unslothai/unsloth.git"

# Option C — Using uv (Unsloth's recommended package manager):
uv pip install unsloth --torch-backend=auto
```

> **⚠️ If Unsloth shows no output / hangs**, do a clean reinstall:
> ```bash
> pip uninstall -y unsloth unsloth_zoo
> pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth
> pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth_zoo
> ```
>
> Also make sure to run with unbuffered Python:
> ```bash
> python -u finetune/train.py ...
> ```

### Step 2: Prepare Training Data

Downloads 3 HuggingFace datasets + generates custom pairs:

```bash
python finetune/prepare_dataset.py
```

**Datasets used:**
| Dataset | Samples | Source |
|---------|---------|--------|
| Sujet-Finance-Instruct-177k | 50K | `Sujet-AI/Sujet-Finance-Instruct-177k` |
| FinGPT Sentiment | 30K | `FinGPT/fingpt-sentiment-train` |
| Financial QA | 10K | `virattt/financial-qa-10K` |
| Custom Business Analysis | 5 | Hand-crafted demonstrations |

### Step 3: Train

```bash
# RTX A6000 — auto-detects GPU and optimizes batch size + dtype
python -u finetune/train.py \
    --model unsloth/Phi-3-mini-4k-instruct \
    --epochs 3 \
    --batch-size 8 \
    --lora-rank 64 \
    --max-samples 20000
```

**Expected on ThunderCompute RTX A6000 (48 GB VRAM):**
- ~1.5-2 hours for 20K samples (Unsloth backend)
- ~2-3 hours for 20K samples (Transformers fallback)
- Uses **bfloat16** automatically (Ampere GPU native support)
- Auto-optimizes batch size to 8 and grad_accum to 2 (effective batch = 16)
- Uses gradient checkpointing for memory efficiency

**Backend selection:**
```bash
# Auto-detect (tries Unsloth first, falls back to Transformers)
python -u finetune/train.py --backend auto

# Force Unsloth only
python -u finetune/train.py --backend unsloth

# Force Transformers + PEFT fallback (no Unsloth needed)
python -u finetune/train.py --backend transformers
```

### Step 4: Convert & Deploy to Ollama

```bash
# Create Modelfile from trained GGUF
python finetune/create_modelfile.py --approach gguf --gguf-path finetune/output/model.gguf

# Import to Ollama
ollama create business-analyst -f finetune/Modelfile

# Update .env
# OLLAMA_MODEL=business-analyst
```

### Step 5: Use

```python
from llm.llm_client import LLMClient
client = LLMClient(model="business-analyst")
response = client.generate("Analyze revenue trends from Q1 2024 data...")
```

---

## Troubleshooting

### Unsloth loads with no output
1. Run with `python -u` (unbuffered mode)
2. The updated `train.py` forces `flush=True` on all prints and imports Unsloth **before** transformers/trl/peft
3. Clean reinstall Unsloth (see Step 1 above)

### Training hangs / OOM
- Check VRAM with `nvidia-smi -l 1` in another terminal
- Reduce `--batch-size` to 4 or `--max-samples` to 10000
- Try `--backend transformers` if Unsloth is unstable

### ThunderCompute-specific
- **Do NOT reinstall CUDA** — ThunderCompute pre-configures CUDA 12.8+
- **Use `pip`**, not `conda`, for all training dependencies
- Set `CUDA_VISIBLE_DEVICES=0` if you have issues with GPU detection

---

## Why Phi-3 Mini?

| Feature | Phi-3 Mini (3.8B) | Llama 3.1 (8B) |
|---------|-------------------|----------------|
| Parameters | 3.8B | 8B |
| Speed | ★★★★★ | ★★★☆☆ |
| English Quality | ★★★★☆ | ★★★★☆ |
| Multilingual | ❌ English only | ✅ Many languages |
| VRAM Usage | ~3 GB | ~6 GB |
| Fine-tune Memory | ~12 GB (QLoRA) | ~20 GB (QLoRA) |
| Best For | Fast analysis | General purpose |

Since this project only needs **English** and business data analysis,
Phi-3 Mini gives better speed-to-quality ratio.
