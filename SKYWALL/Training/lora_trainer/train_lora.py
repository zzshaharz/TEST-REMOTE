#!/usr/bin/env python3
"""
train_lora.py
SKYWALL - Autonomous Aerial Detection System

LoRA fine-tuning script for a vision-language model (Gemma-based or LLaVA-based)
to perform detailed drone classification from aerial images.

Pipeline:
  1. Load base VLM (or use SigLIP + Gemma 2B as proxy)
  2. Apply QLoRA (4-bit quantization + LoRA adapters)
  3. Prepare image+text pair dataset for supervised fine-tuning
  4. Train with gradient checkpointing
  5. Merge adapters and export
  6. Convert to GGUF for CoreML/llama.cpp deployment

Dataset format (JSON Lines):
  {"image": "path/to/img.jpg", "question": "What aerial object is in this image?",
   "answer": "This is a DJI Mavic 3 consumer quadcopter drone. It is hovering at
   approximately 50m altitude with a camera gimbal visible. The drone shows typical
   DJI design with foldable arms. Threat level: MEDIUM."}

Usage:
  python train_lora.py --data ./dataset --model google/gemma-2b-it --output ./output
  python train_lora.py --export_only --model ./output/merged --output ./export
"""

import argparse
import os
import sys
import json
import shutil
import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from datasets import load_dataset, Dataset as HFDataset
import coremltools as ct


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class LoraTrainingConfig:
    # Base model
    base_model: str = "google/gemma-2b-it"

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"

    # Quantization (QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True  # Double quantization

    # Training
    num_epochs: int = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_seq_length: int = 512
    fp16: bool = False
    bf16: bool = True     # Use bfloat16 if available

    # Evaluation
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    early_stopping_patience: int = 5

    # Data
    train_split: float = 0.85
    val_split: float = 0.10


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SKYWALL, an expert aerial threat assessment system.
When shown an image, analyze any aerial objects and provide a structured assessment:
- Object type and model identification
- Estimated size category (micro/small/medium/large)
- Payload observation (camera, thermal, cargo, none)
- Behavioral pattern (hovering, surveying, approaching, retreating, circling, racing)
- Threat level assessment (NONE/LOW/MEDIUM/HIGH)
- Confidence level (0.0-1.0)
Be concise and precise. Focus on observable characteristics."""


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DroneClassificationDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, processor, max_length: int = 512):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.processor  = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image_path = sample.get("image", "")
        question   = sample.get("question", "What aerial object is visible in this image?")
        answer     = sample.get("answer", "")

        # Load image
        try:
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
                # Resize to model input size
                image = image.resize((224, 224), Image.LANCZOS)
            else:
                # Synthetic sky image for missing files
                img_arr = np.random.randint(150, 220, (224, 224, 3), dtype=np.uint8)
                img_arr[:, :, 0] = img_arr[:, :, 0] * 70 // 100  # Less red → bluer sky
                image = Image.fromarray(img_arr)
        except Exception:
            img_arr = np.ones((224, 224, 3), dtype=np.uint8) * 180
            image = Image.fromarray(img_arr)

        # Format prompt
        prompt = f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
        prompt += f"<start_of_turn>user\n{question}<end_of_turn>\n"
        prompt += f"<start_of_turn>model\n{answer}<end_of_turn>"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Labels: mask the prompt (only compute loss on answer tokens)
        labels = input_ids.clone()
        # Find where answer starts (after "model\n")
        model_token = self.tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
        prompt_end_idx = self.find_answer_start(input_ids, model_token)
        labels[:prompt_end_idx] = -100  # Ignore prompt loss

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    def find_answer_start(self, input_ids: torch.Tensor, model_tokens: List[int]) -> int:
        """Find the token index where the model's response starts."""
        ids = input_ids.tolist()
        mt = model_tokens
        for i in range(len(ids) - len(mt)):
            if ids[i:i + len(mt)] == mt:
                return i + len(mt)
        return len(ids) // 2  # Fallback: mask first half


def load_dataset_from_json(data_dir: Path) -> List[Dict]:
    """Load samples from JSONL files in data_dir."""
    samples = []

    # Look for JSONL files
    for jsonl_file in data_dir.glob("*.jsonl"):
        with open(str(jsonl_file), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # Also look for JSON arrays
    for json_file in data_dir.glob("*.json"):
        with open(str(json_file), "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
            except json.JSONDecodeError:
                continue

    print(f"[Dataset] Loaded {len(samples)} samples from {data_dir}")
    return samples


def create_synthetic_dataset(output_dir: Path, n_samples: int = 200) -> Path:
    """Generate synthetic training data for pipeline testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    templates = [
        {
            "class": "fpv_racer", "size": "micro",
            "q": "What drone is visible in this image? Describe its type and threat assessment.",
            "a_template": "This is an FPV racing quadcopter, likely under 250g class. "
                          "Carbon fiber frame visible with exposed electronics. "
                          "Small FPV camera mounted on tilt bracket (no gimbal stabilization). "
                          "Behavioral pattern: {behavior}. "
                          "Threat level: HIGH - FPV drones can be piloted at high speed with direct video feedback. "
                          "Confidence: {conf:.2f}",
        },
        {
            "class": "dji_mavic", "size": "small",
            "q": "Identify the aerial object and assess the threat level.",
            "a_template": "This appears to be a DJI Mavic series consumer quadcopter. "
                          "Foldable arm design consistent with Mavic 3 or Mavic Air 2S. "
                          "Stabilized camera gimbal is visible beneath the frame. "
                          "Behavioral pattern: {behavior}. "
                          "Threat level: MEDIUM - Consumer drone with significant camera payload. "
                          "Confidence: {conf:.2f}",
        },
        {
            "class": "fixed_wing", "size": "medium",
            "q": "What type of aerial object is present? Provide full classification.",
            "a_template": "Fixed-wing UAV detected. High-aspect ratio wings with pusher or tractor propeller. "
                          "Design consistent with commercial survey or long-range reconnaissance platform. "
                          "Behavioral pattern: {behavior}. "
                          "Threat level: MEDIUM - Extended range capability. "
                          "Confidence: {conf:.2f}",
        },
        {
            "class": "bird", "size": "small",
            "q": "Classify the aerial object in this image.",
            "a_template": "Bird detected. Organic wing morphology with flapping motion characteristics. "
                          "Not a drone. Wingbeat pattern and feather structure clearly visible. "
                          "Threat level: NONE - Natural wildlife. "
                          "Confidence: {conf:.2f}",
        },
        {
            "class": "helicopter", "size": "large",
            "q": "What is the aerial object? Assess threat level.",
            "a_template": "Rotary-wing helicopter detected. Single main rotor with tail rotor. "
                          "Scale suggests manned aircraft. Landing skids visible. "
                          "Behavioral pattern: {behavior}. "
                          "Threat level: LOW - Likely authorized manned aircraft. "
                          "Confidence: {conf:.2f}",
        },
    ]

    behaviors = ["hovering", "surveying", "approaching", "retreating", "circling"]
    samples = []

    for i in range(n_samples):
        tmpl = random.choice(templates)

        # Generate synthetic image
        img_arr = np.random.randint(130, 210, (224, 224, 3), dtype=np.uint8)
        img_arr[:, :, 0] = img_arr[:, :, 0] * 65 // 100
        img = Image.fromarray(img_arr)
        img_path = output_dir / "images" / f"sample_{i:05d}.jpg"
        img.save(str(img_path))

        behavior = random.choice(behaviors)
        conf     = random.uniform(0.72, 0.97)
        answer   = tmpl["a_template"].format(behavior=behavior, conf=conf)

        samples.append({
            "image":    str(img_path),
            "question": tmpl["q"],
            "answer":   answer,
            "class":    tmpl["class"],
        })

    jsonl_path = output_dir / "training_data.jsonl"
    with open(str(jsonl_path), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"[Dataset] Created {n_samples} synthetic samples at {jsonl_path}")
    return output_dir


# ─── Model Setup ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, config: LoraTrainingConfig):
    """Load quantized base model and tokenizer."""
    print(f"[Model] Loading {model_name}...")

    # Quantization config (QLoRA)
    bnb_config = None
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[Model] Full load failed ({e}). Trying CPU load...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for k-bit training
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, tokenizer


def apply_lora(model, config: LoraTrainingConfig):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(
    model, tokenizer, train_samples: List[Dict], val_samples: List[Dict],
    config: LoraTrainingConfig, output_dir: Path
) -> Path:
    """Run LoRA fine-tuning."""

    train_ds = DroneClassificationDataset(train_samples, tokenizer, None, config.max_seq_length)
    val_ds   = DroneClassificationDataset(val_samples,   tokenizer, None, config.max_seq_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    print(f"\n[Training] Starting LoRA fine-tuning...")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    print(f"  Epochs: {config.num_epochs}, LR: {config.learning_rate}")

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapter"))

    print(f"\n[Training] Complete.")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Steps: {train_result.global_step}")

    # Save training metrics
    metrics = train_result.metrics
    with open(str(output_dir / "training_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    return output_dir / "lora_adapter"


# ─── Adapter Merge ────────────────────────────────────────────────────────────

def merge_and_save(base_model_name: str, adapter_path: Path, output_dir: Path) -> Path:
    """Merge LoRA adapter into base model and save full weights."""
    print(f"\n[Merge] Loading base model + adapter...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.merge_and_unload()

    merged_path = output_dir / "merged_model"
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    print(f"[Merge] Merged model saved to {merged_path}")
    return merged_path


# ─── GGUF Export ──────────────────────────────────────────────────────────────

def export_to_gguf(merged_model_path: Path, output_dir: Path) -> Optional[Path]:
    """Convert merged model to GGUF format (requires llama.cpp)."""
    gguf_path = output_dir / "SkyWALLClassifier_Q4_K_M.gguf"

    # Attempt to use llama.cpp convert script
    convert_script = shutil.which("llama-convert") or "/opt/llama.cpp/convert_hf_to_gguf.py"

    if not Path(convert_script).exists():
        print(f"[GGUF] llama.cpp convert script not found. Skipping GGUF export.")
        print(f"[GGUF] To export manually: python /path/to/llama.cpp/convert_hf_to_gguf.py "
              f"{merged_model_path} --outfile {gguf_path} --outtype q4_k_m")
        return None

    import subprocess
    result = subprocess.run([
        sys.executable, convert_script,
        str(merged_model_path),
        "--outfile", str(gguf_path),
        "--outtype", "q4_k_m",
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[GGUF] Exported to {gguf_path}")
        return gguf_path
    else:
        print(f"[GGUF] Export failed: {result.stderr}")
        return None


# ─── Sample Inference ─────────────────────────────────────────────────────────

def run_sample_inference(model, tokenizer, image_path: Optional[str] = None, device: str = "cpu"):
    """Test the fine-tuned model with a sample prompt."""
    question = "What aerial object is in this image? Provide threat assessment."
    prompt = (f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
              f"<start_of_turn>user\n{question}<end_of_turn>\n"
              f"<start_of_turn>model\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[Inference] Sample output:\n{response}")
    return response


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SKYWALL LoRA Classifier Training")
    parser.add_argument("--data",            type=Path, default=Path("./dataset"),
                        help="Directory with training JSONL files")
    parser.add_argument("--model",           type=str,  default="google/gemma-2b-it",
                        help="Base model name or path")
    parser.add_argument("--output",          type=Path, default=Path("./lora_output"),
                        help="Output directory")
    parser.add_argument("--epochs",          type=int,  default=5)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--lora_r",          type=int,  default=16)
    parser.add_argument("--lora_alpha",      type=int,  default=32)
    parser.add_argument("--merge_only",      action="store_true")
    parser.add_argument("--adapter_path",    type=Path, default=None)
    parser.add_argument("--create_sample",   action="store_true")
    parser.add_argument("--no_4bit",         action="store_true",
                        help="Disable 4-bit quantization (use full precision)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    config = LoraTrainingConfig(
        base_model=args.model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=not args.no_4bit,
    )

    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open(str(args.output / "lora_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # Create synthetic dataset
    if args.create_sample:
        args.data = create_synthetic_dataset(args.output / "synthetic_data", n_samples=500)
        print(f"Synthetic dataset created at: {args.data}")

    # Merge only mode
    if args.merge_only:
        if not args.adapter_path:
            print("[Error] --adapter_path required for --merge_only")
            sys.exit(1)
        merged = merge_and_save(args.model, args.adapter_path, args.output)
        export_to_gguf(merged, args.output)
        return

    # Load dataset
    samples = load_dataset_from_json(args.data)

    if len(samples) == 0:
        print("[Warning] No samples found. Creating synthetic dataset...")
        args.data = create_synthetic_dataset(args.output / "synthetic_data", n_samples=500)
        samples = load_dataset_from_json(args.data)

    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * config.train_split)
    n_val   = int(n * config.val_split)
    train_samples = samples[:n_train]
    val_samples   = samples[n_train:n_train + n_val]

    print(f"[Dataset] Train={len(train_samples)} Val={len(val_samples)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, config)
    model = apply_lora(model, config)

    # Train
    adapter_path = train_model(
        model, tokenizer, train_samples, val_samples, config, args.output
    )

    # Sample inference test
    run_sample_inference(model, tokenizer)

    # Merge and export
    print("\n[Merge] Merging adapter into base model...")
    merged_path = merge_and_save(args.model, adapter_path, args.output)

    # GGUF export (requires llama.cpp)
    gguf_path = export_to_gguf(merged_path, args.output)

    print(f"\n{'='*60}")
    print("[Done] LoRA training complete.")
    print(f"  Adapter:     {adapter_path}")
    print(f"  Merged:      {merged_path}")
    if gguf_path:
        print(f"  GGUF:        {gguf_path}")
    print(f"  Output:      {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
