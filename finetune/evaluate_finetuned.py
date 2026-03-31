"""
Evalue le modele fine-tune sur le test set et compare avec le baseline Mistral.
Produit un rapport de comparaison: evaluation/comparison_report.json

Usage:
    python finetune/evaluate_finetuned.py
    python finetune/evaluate_finetuned.py --model-path finetune/model_output/final
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = str(BASE_DIR / "hospital.db")
TEST_FILE  = Path(__file__).resolve().parent / "dataset_test.jsonl"
REPORT_OUT = BASE_DIR / "evaluation" / "comparison_report.json"
BASELINE   = BASE_DIR / "evaluation" / "evaluation_report.json"

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ── Helpers SQL ───────────────────────────────────────────────────────────────

def execute_sql(sql: str) -> Tuple[List[str], List[tuple]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, rows
    finally:
        conn.close()


def clean_sql(raw: str) -> str:
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip("`").strip()
    match = re.search(r"(SELECT\b.*)", raw, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


def result_set_match(expected_sql: str, generated_sql: str) -> int:
    try:
        exp_cols, exp_rows = execute_sql(expected_sql)
        act_cols, act_rows = execute_sql(generated_sql)
        exp_n = [str(c).lower() for c in exp_cols]
        act_n = [str(c).lower() for c in act_cols]
        if exp_n != act_n:
            return 0
        exp_r = sorted([tuple(str(v).lower() for v in r) for r in exp_rows])
        act_r = sorted([tuple(str(v).lower() for v in r) for r in act_rows])
        return 1 if exp_r == act_r else 0
    except Exception:
        return 0


def is_sql_valid(sql: str) -> Tuple[bool, str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(f"EXPLAIN {sql}")
        return True, "valid"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


# ── Generateur fine-tune ─────────────────────────────────────────────────────

class FinetunedGenerator:
    def __init__(self, model_path: str) -> None:
        print(f"[INFO] Chargement du modele fine-tune: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        print("[INFO] Modele charge.")

    def generate(self, instruction: str, input_text: str) -> str:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraire seulement la partie apres ### Response:
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        return clean_sql(response)


# ── Evaluation principale ─────────────────────────────────────────────────────

def evaluate(model_path: str) -> None:
    # Charger le test set
    test_items = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_items.append(json.loads(line))

    print(f"[INFO] {len(test_items)} questions dans le test set")

    generator = FinetunedGenerator(model_path)

    results = []
    for item in test_items:
        question = item["input"].split("\n")[0].replace("Question: ", "")
        expected_sql = item["output"]
        instruction  = item["instruction"]
        input_text   = item["input"]

        generated_sql = generator.generate(instruction, input_text)
        valid, msg    = is_sql_valid(generated_sql)
        exec_ok       = 0
        set_match     = 0

        if valid:
            try:
                execute_sql(generated_sql)
                exec_ok   = 1
                set_match = result_set_match(expected_sql, generated_sql)
            except Exception:
                exec_ok = 0

        final_score = round(0.30 * valid + 0.30 * exec_ok + 0.40 * set_match, 4)

        results.append({
            "question":       question,
            "category":       item.get("category", ""),
            "expected_sql":   expected_sql,
            "generated_sql":  generated_sql,
            "sql_valid":      int(valid),
            "execution_ok":   exec_ok,
            "result_set_match": set_match,
            "final_score":    final_score,
        })
        print(f"  valid={int(valid)} exec={exec_ok} set={set_match} score={final_score:.2f}  |  {question[:60]}")

    n = len(results)
    summary_ft = {
        "model":              "TinyLlama-finetuned",
        "count":              n,
        "avg_sql_valid":      round(sum(r["sql_valid"]        for r in results) / n, 4),
        "avg_execution_ok":   round(sum(r["execution_ok"]     for r in results) / n, 4),
        "avg_result_set_match": round(sum(r["result_set_match"] for r in results) / n, 4),
        "avg_final_score":    round(sum(r["final_score"]      for r in results) / n, 4),
    }

    # Charger le baseline Mistral si disponible
    summary_baseline: Optional[dict] = None
    if BASELINE.exists():
        with open(BASELINE, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        b = baseline_data.get("summary", {})
        summary_baseline = {
            "model":               "Mistral-7B-baseline",
            "count":               b.get("count", 0),
            "avg_sql_valid":       b.get("avg_sql_valid", 0),
            "avg_execution_ok":    b.get("avg_execution_ok", 0),
            "avg_result_set_match": b.get("avg_result_set_match", 0),
            "avg_final_score":     b.get("avg_final_score", 0),
        }

    # Rapport final
    report = {
        "finetuned": summary_ft,
        "baseline":  summary_baseline,
        "comparison": {
            "delta_sql_valid":       round(summary_ft["avg_sql_valid"]       - (summary_baseline["avg_sql_valid"]       if summary_baseline else 0), 4),
            "delta_execution_ok":    round(summary_ft["avg_execution_ok"]    - (summary_baseline["avg_execution_ok"]    if summary_baseline else 0), 4),
            "delta_result_set_match": round(summary_ft["avg_result_set_match"] - (summary_baseline["avg_result_set_match"] if summary_baseline else 0), 4),
            "delta_final_score":     round(summary_ft["avg_final_score"]     - (summary_baseline["avg_final_score"]     if summary_baseline else 0), 4),
        },
        "test_results": results,
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("COMPARAISON BASELINE vs FINE-TUNED")
    print("=" * 60)
    header = f"{'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>10}"
    print(header)
    print("-" * 60)
    metrics = ["avg_sql_valid", "avg_execution_ok", "avg_result_set_match", "avg_final_score"]
    for m in metrics:
        base_val = summary_baseline[m] if summary_baseline else 0
        ft_val   = summary_ft[m]
        delta    = ft_val - base_val
        sign     = "+" if delta >= 0 else ""
        print(f"{m:<25} {base_val:>12.2%} {ft_val:>12.2%} {sign}{delta:>9.2%}")
    print("=" * 60)
    print(f"\nRapport sauvegardé → {REPORT_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "model_output" / "final"),
        help="Chemin vers le modele fine-tune",
    )
    args = parser.parse_args()
    evaluate(args.model_path)
