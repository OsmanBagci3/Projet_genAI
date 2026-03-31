"""
Fine-tuning local de TinyLlama (1.1B) sur le dataset NLQ->SQL hospitalier.
Utilise LoRA pour minimiser la memoire (fonctionne sans GPU dedie).

Usage:
    python finetune/finetune.py
    python finetune/finetune.py --epochs 5 --output finetune/model_output
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from trl import SFTTrainer

# ── Modele: TinyLlama 1.1B (petit, rapide, fonctionne en local) ───────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR


def format_example(example: dict) -> str:
    """Formate un exemple en prompt Alpaca (instruction / input / response)."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def main(epochs: int, output_dir: str, max_seq_length: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()
    print(f"[INFO] Device: {device}  |  FP16: {use_fp16}")

    # ── Charger les datasets ──────────────────────────────────────────────────
    train_ds = load_dataset(
        "json",
        data_files=str(DATA_DIR / "dataset_train.jsonl"),
        split="train",
    )
    val_ds = load_dataset(
        "json",
        data_files=str(DATA_DIR / "dataset_val.jsonl"),
        split="train",
    )
    print(f"[INFO] Train: {len(train_ds)} exemples | Val: {len(val_ds)} exemples")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Modele en 4-bit si GPU sinon FP32 ───────────────────────────────────
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # CPU: charge en float32 (plus lent mais fonctionne)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    model.config.use_cache = False

    # ── LoRA: fine-tune seulement une partie des poids ───────────────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Arguments d'entrainement ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=use_fp16,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        formatting_func=format_example,
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("\n[INFO] Début de l'entraînement...")
    trainer.train()

    # ── Sauvegarder ──────────────────────────────────────────────────────────
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n[INFO] Modèle sauvegardé → {final_path}")
    print("[INFO] Fine-tuning terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=3,
                        help="Nombre d'epochs (defaut: 3)")
    parser.add_argument("--output",     type=str,
                        default=str(BASE_DIR / "model_output"),
                        help="Dossier de sortie du modele")
    parser.add_argument("--max-seq",    type=int, default=512,
                        help="Longueur max des sequences (defaut: 512)")
    args = parser.parse_args()

    main(epochs=args.epochs, output_dir=args.output, max_seq_length=args.max_seq)
