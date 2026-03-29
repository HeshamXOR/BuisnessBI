"""
Fine-Tuning Training Script
=============================
QLoRA fine-tuning for the business data analysis model.

Primary backend:
    - Unsloth (fastest when it works cleanly)

Fallback backend:
    - Transformers + PEFT + bitsandbytes

This lets the project keep training even if Unsloth is unstable in the
current environment.

Optimized for:  ThunderCompute RTX A6000 (48 GB VRAM, Ampere)
CUDA:           12.8+
PyTorch:        2.9.0+
"""

# ── CRITICAL: Force unbuffered output so prints appear immediately ────
# This fixes the "unsloth loads with no output" problem.
import sys
import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Force stdout/stderr to be unbuffered (safe for both terminal and Jupyter)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass  # Jupyter kernels use custom stdout that may not support reconfigure

# Detect Unsloth without importing it eagerly.
# Importing Unsloth at module-load time can hang inside notebooks.
import importlib.util

_UNSLOTH_AVAILABLE = importlib.util.find_spec("unsloth") is not None

import argparse
import inspect
import json
from typing import Optional


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _print(msg: str = "") -> None:
    """Print with forced flush so output is always visible."""
    print(msg, flush=True)


def _detect_gpu_info() -> dict:
    """Detect GPU info for automatic optimization."""
    info = {
        "gpu_name": "unknown",
        "vram_gb": 0,
        "is_ampere_or_newer": False,
        "cuda_version": "unknown",
        "torch_version": "unknown",
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
            # Ampere = compute capability 8.x, Hopper = 9.x
            cc = torch.cuda.get_device_capability(0)
            info["is_ampere_or_newer"] = cc[0] >= 8
            info["cuda_version"] = torch.version.cuda or "unknown"
    except Exception:
        pass
    return info


def _resolve_base_model_name(base_model: str) -> str:
    """Map Unsloth aliases to upstream Hugging Face model names."""
    model_map = {
        "unsloth/Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    }
    return model_map.get(base_model, base_model)


def _load_model_with_unsloth(
    base_model: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: int,
    gpu_info: dict,
):
    """Load a QLoRA model with Unsloth."""
    _print("🦥 Importing Unsloth backend...")
    from unsloth import FastLanguageModel

    _print("📦 Loading base model with Unsloth...")
    _print(f"   Model: {base_model}")
    _print(f"   GPU: {gpu_info['gpu_name']} ({gpu_info['vram_gb']} GB)")

    # Ampere+ GPUs support bfloat16 natively — much better than float16
    dtype = None  # let Unsloth auto-detect (will pick bf16 on Ampere+)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    _print("   ✅ Base model loaded.")

    _print("🔧 Applying LoRA adapters with Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
    )
    _print("   ✅ LoRA adapters applied.")

    return model, tokenizer, "unsloth", base_model


def _load_model_with_transformers(
    base_model: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: int,
    gpu_info: dict,
):
    """Load a QLoRA model with Transformers + PEFT."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    resolved_model = _resolve_base_model_name(base_model)

    _print("📦 Loading base model with Transformers fallback...")
    _print(f"   Resolved model: {resolved_model}")
    _print(f"   GPU: {gpu_info['gpu_name']} ({gpu_info['vram_gb']} GB)")

    # Use bf16 on Ampere+ (A6000, A100, etc.), fp16 otherwise
    use_bf16 = gpu_info["is_ampere_or_newer"]
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype_name = "bfloat16" if use_bf16 else "float16"
    _print(f"   Compute dtype: {dtype_name} (Ampere+: {use_bf16})")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _print("   ✅ Tokenizer loaded.")

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    _print("   ✅ Base model loaded with 4-bit quantization.")

    _print("🔧 Applying LoRA adapters with PEFT...")
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    _print("   ✅ LoRA adapters applied.")

    return model, tokenizer, "transformers", resolved_model


def _load_training_model(
    base_model: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: int,
    backend: str,
    gpu_info: dict,
):
    """Load the training model using the requested backend."""
    last_error = None

    if backend in ("auto", "unsloth"):
        if not _UNSLOTH_AVAILABLE and backend == "auto":
            _print("⚠️  Unsloth not available, skipping to Transformers fallback...")
        elif not _UNSLOTH_AVAILABLE and backend == "unsloth":
            raise ImportError(
                "Unsloth backend was explicitly requested but is not installed.\n"
                "Install it with:  pip install unsloth\n"
                "Or for ThunderCompute A6000:\n"
                '  pip install "unsloth[cu128-ampere-torch290] @ '
                'git+https://github.com/unslothai/unsloth.git"'
            )
        else:
            try:
                return _load_model_with_unsloth(
                    base_model=base_model,
                    max_seq_length=max_seq_length,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    gpu_info=gpu_info,
                )
            except Exception as exc:
                last_error = exc
                if backend == "unsloth":
                    raise
                _print(
                    "⚠️  Unsloth backend failed, switching to Transformers fallback..."
                )
                _print(f"   Reason: {exc}")

    if backend in ("auto", "transformers"):
        return _load_model_with_transformers(
            base_model=base_model,
            max_seq_length=max_seq_length,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            gpu_info=gpu_info,
        )

    raise ValueError(
        f"Unsupported backend '{backend}'. Choose from: auto, unsloth, transformers."
    ) from last_error


def _save_merged_transformers_model(
    adapter_path: str, output_dir: str, tokenizer
) -> bool:
    """Merge PEFT adapters into a full model after fallback training."""
    import torch
    from peft import AutoPeftModelForCausalLM

    merged_path = os.path.join(output_dir, "merged_model")
    _print("🔄 Re-loading adapter in FP16 to create merged model...")

    merge_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    merge_model = merge_model.merge_and_unload()
    merge_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    return True


def train(
    base_model: str = "unsloth/Phi-3-mini-4k-instruct",
    dataset_path: str = "finetune/data",
    output_dir: str = "finetune/output",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    max_seq_length: int = 2048,
    max_samples: Optional[int] = None,
    backend: str = "auto",
    gradient_accumulation_steps: int = 4,
    val_size: float = 0.02,
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 0.001,
    eval_steps: int = 100,
):
    """
    Fine-tune a model using QLoRA.

    Args:
        base_model: HuggingFace model to fine-tune.
        dataset_path: Directory containing training JSONL files.
        output_dir: Where to save the fine-tuned model.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Learning rate for AdamW.
        lora_rank: LoRA rank (higher = more capacity, more memory).
        lora_alpha: LoRA alpha scaling factor.
        max_seq_length: Maximum sequence length.
        max_samples: Max training samples (None = use all).
        backend: auto, unsloth, or transformers.
        gradient_accumulation_steps: Gradient accumulation steps.
        val_size: Validation split ratio for early stopping.
        early_stopping_patience: Number of evals without improvement before stop.
        early_stopping_threshold: Minimum eval_loss improvement to reset patience.
        eval_steps: Evaluation frequency in steps.
    """
    _print("=" * 60)
    _print("🧠 Business Data Analysis LLM Fine-Tuning")
    _print("=" * 60)

    # ── GPU Detection ──────────────────────────────────────────────
    gpu_info = _detect_gpu_info()
    _print(f"  🖥️  GPU:        {gpu_info['gpu_name']}")
    _print(f"  💾 VRAM:       {gpu_info['vram_gb']} GB")
    _print(f"  🔧 Ampere+:    {gpu_info['is_ampere_or_newer']}")
    _print(f"  🔥 PyTorch:    {gpu_info['torch_version']}")
    _print(f"  ⚡ CUDA:       {gpu_info['cuda_version']}")
    _print()

    # ── Auto-optimize for RTX A6000 (48 GB VRAM) ──────────────────
    if gpu_info["vram_gb"] >= 40:
        # A6000 / A100 class — can handle larger batches
        if batch_size <= 4:
            batch_size = 8
            _print(
                f"  🚀 Auto-optimized batch_size → {batch_size} (48 GB VRAM detected)"
            )
        if gradient_accumulation_steps > 2:
            gradient_accumulation_steps = 2
            _print(
                f"  🚀 Auto-optimized grad_accum → {gradient_accumulation_steps} "
                f"(effective batch = {batch_size * gradient_accumulation_steps})"
            )

    _print(f"  Base model:  {base_model}")
    _print(f"  Dataset dir: {dataset_path}")
    _print(f"  Output dir:  {output_dir}")
    _print(f"  Epochs:      {epochs}")
    _print(f"  Batch size:  {batch_size}")
    _print(f"  Grad accum:  {gradient_accumulation_steps}")
    _print(f"  LoRA rank:   {lora_rank}")
    _print(f"  Max seq len: {max_seq_length}")
    _print(f"  Backend:     {backend}")
    _print(f"  Val split:   {val_size}")
    _print(f"  Eval steps:  {eval_steps}")
    _print()

    model, tokenizer, backend_used, resolved_base_model = _load_training_model(
        base_model=base_model,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        backend=backend,
        gpu_info=gpu_info,
    )
    _print(f"✅ Training backend in use: {backend_used}")
    _print(f"   Model source: {resolved_base_model}")

    _print("📂 Loading training data...")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}\n"
            f"Run first:  python finetune/prepare_dataset.py"
        )

    all_messages = []
    file_sample_counts = {}
    skipped_lines = 0

    for filename in sorted(os.listdir(dataset_path)):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(dataset_path, filename)
            file_count = 0
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        all_messages.append(item)
                        file_count += 1
                    except json.JSONDecodeError:
                        skipped_lines += 1
                        continue
            file_sample_counts[filename] = file_count

    if not file_sample_counts:
        raise ValueError(f"No JSONL training files found in {dataset_path}")

    _print("   Samples by file:")
    for filename, file_count in file_sample_counts.items():
        _print(f"     - {filename}: {file_count}")
    if skipped_lines:
        _print(f"   Skipped malformed lines: {skipped_lines}")

    if not all_messages:
        raise ValueError(
            f"No valid training samples were loaded from {dataset_path}. "
            "Check the prepared JSONL files."
        )

    if max_samples and len(all_messages) > max_samples:
        import random

        _print(
            f"   Sampling {max_samples} rows from {len(all_messages)} available samples..."
        )
        random.shuffle(all_messages)
        all_messages = all_messages[:max_samples]

    _print(f"   Total training samples: {len(all_messages)}")

    from datasets import Dataset

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(all_messages)

    use_validation = bool(val_size and len(dataset) >= 100)
    if use_validation:
        split_dataset = dataset.train_test_split(
            test_size=val_size, seed=42, shuffle=True
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        _print(
            f"   Validation enabled: {len(train_dataset)} train / {len(eval_dataset)} eval"
        )
    else:
        train_dataset = dataset
        eval_dataset = None
        _print("   Validation disabled: dataset too small or val_size=0")

    train_dataset = train_dataset.map(
        format_chat, remove_columns=train_dataset.column_names
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            format_chat, remove_columns=eval_dataset.column_names
        )

    _print(f"   Tokenization-ready train rows: {len(train_dataset)}")
    if eval_dataset is not None:
        _print(f"   Tokenization-ready eval rows: {len(eval_dataset)}")

    _print("🚀 Starting fine-tuning...")
    from transformers import TrainingArguments
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer

    os.makedirs(output_dir, exist_ok=True)

    # ── Use bf16 on Ampere+, fp16 on older GPUs ───────────────────
    use_bf16 = gpu_info["is_ampere_or_newer"]
    _print(f"   Training precision: {'bf16' if use_bf16 else 'fp16'}")

    training_args_signature = inspect.signature(TrainingArguments.__init__)
    training_args_params = set(training_args_signature.parameters.keys())

    training_args_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": 100,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "logging_steps": 5,
        "logging_first_step": True,
        "save_total_limit": 2,
        "optim": "adamw_8bit",
        "seed": 42,
        "report_to": "none",
        "ddp_find_unused_parameters": False,
    }

    callbacks = []
    if eval_dataset is not None:
        eval_key = None
        if "evaluation_strategy" in training_args_params:
            eval_key = "evaluation_strategy"
        elif "eval_strategy" in training_args_params:
            eval_key = "eval_strategy"

        if eval_key is not None:
            training_args_kwargs[eval_key] = "steps"

        if "eval_steps" in training_args_params:
            training_args_kwargs["eval_steps"] = eval_steps
        if "save_strategy" in training_args_params:
            training_args_kwargs["save_strategy"] = "steps"
        if "save_steps" in training_args_params:
            training_args_kwargs["save_steps"] = eval_steps
        if "load_best_model_at_end" in training_args_params:
            training_args_kwargs["load_best_model_at_end"] = True
        if "metric_for_best_model" in training_args_params:
            training_args_kwargs["metric_for_best_model"] = "eval_loss"
        if "greater_is_better" in training_args_params:
            training_args_kwargs["greater_is_better"] = False

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )
        _print(
            "   Early stopping enabled: "
            f"patience={early_stopping_patience}, threshold={early_stopping_threshold}"
        )
        _print(
            "   TrainingArguments compatibility mode: "
            f"eval_key={eval_key}, "
            f"eval_steps={'eval_steps' in training_args_params}, "
            f"load_best_model={'load_best_model_at_end' in training_args_params}"
        )
    else:
        if "save_strategy" in training_args_params:
            training_args_kwargs["save_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_signature = inspect.signature(SFTTrainer.__init__)
    trainer_params = set(trainer_signature.parameters.keys())

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": training_args,
    }

    if eval_dataset is not None and "eval_dataset" in trainer_params:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if callbacks and "callbacks" in trainer_params:
        trainer_kwargs["callbacks"] = callbacks

    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = max_seq_length

    _print(
        "   SFTTrainer compatibility mode: "
        f"tokenizer={'tokenizer' in trainer_params}, "
        f"processing_class={'processing_class' in trainer_params}, "
        f"eval_dataset={'eval_dataset' in trainer_params}, "
        f"callbacks={'callbacks' in trainer_params}, "
        f"dataset_text_field={'dataset_text_field' in trainer_params}, "
        f"max_seq_length={'max_seq_length' in trainer_params}"
    )

    trainer = SFTTrainer(**trainer_kwargs)

    _print("   Trainer initialized. Beginning training loop...")
    _print()

    train_result = trainer.train()
    metrics = getattr(train_result, "metrics", {})
    if metrics:
        _print("📈 Training metrics:")
        for key, value in metrics.items():
            _print(f"   {key}: {value}")

    _print("💾 Saving fine-tuned model...")
    adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    merged_saved = False
    try:
        if backend_used == "unsloth":
            _print("🔄 Saving merged model with Unsloth...")
            model.save_pretrained_merged(
                os.path.join(output_dir, "merged_model"),
                tokenizer,
                save_method="merged_16bit",
            )
            merged_saved = True
        else:
            merged_saved = _save_merged_transformers_model(
                adapter_path,
                output_dir,
                tokenizer,
            )
    except Exception as exc:
        _print(f"⚠️  Could not create merged model automatically: {exc}")

    _print()
    _print("=" * 60)
    _print("✅ Fine-tuning complete!")
    _print(f"   LoRA adapter: {output_dir}/lora_adapter")
    if merged_saved:
        _print(f"   Merged model: {output_dir}/merged_model")
    else:
        _print("   Merged model: not created automatically")
    _print()
    _print("Next steps:")
    if merged_saved:
        _print("  1. Convert to GGUF: python finetune/create_modelfile.py")
        _print(
            "  2. Import to Ollama: ollama create business-analyst -f finetune/Modelfile"
        )
    else:
        _print(
            "  1. Re-run with a backend that supports merge, or merge adapters manually"
        )
        _print("  2. Then create the Ollama model from the merged weights")
    _print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM for business analysis")
    parser.add_argument("--model", default="unsloth/Phi-3-mini-4k-instruct")
    parser.add_argument("--data", default="finetune/data")
    parser.add_argument("--output", default="finetune/output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--val-size", type=float, default=0.02)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.001)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument(
        "--backend",
        choices=["auto", "unsloth", "transformers"],
        default="auto",
    )
    args = parser.parse_args()

    train(
        base_model=args.model,
        dataset_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        max_samples=args.max_samples,
        backend=args.backend,
        gradient_accumulation_steps=args.grad_accum,
        val_size=args.val_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        eval_steps=args.eval_steps,
    )
