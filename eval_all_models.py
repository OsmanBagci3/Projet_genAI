"""
Evaluation unifiee des 3 modeles sur dataset_20.json
Meme pipeline, meme formule de score, tout dans Langfuse.

Usage:
    # Tous les modeles:
    python eval_all_models.py --models mistral claude tinyllama

    # Un seul modele:
    python eval_all_models.py --models mistral
    python eval_all_models.py --models claude
    python eval_all_models.py --models tinyllama

    # Chemin custom pour le modele fine-tune:
    python eval_all_models.py --models tinyllama --ft-model finetune/model_output/final
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(Path(__file__).resolve().parent / ".env")

try:
    from langfuse import Langfuse
    _langfuse: Optional[Langfuse] = Langfuse()
except Exception:
    _langfuse = None

from execute_sql import (
    DB_PATH,
    ChainOfThoughtPlanner,
    SchemaInspector,
    SQLExecutor,
    SQLGenerator,
    SQLValidator,
)
from hybrid_retrieval import HybridRetriever
from query_construction import QueryConstructor
from rerank_results import HeuristicReranker
from router import QueryRouter

BASE_DIR      = Path(__file__).resolve().parent
DATASET_PATH  = BASE_DIR / "evaluation" / "dataset_20.json"
OUTPUT_PATH   = BASE_DIR / "evaluation" / "eval_all_models_report.json"
FT_MODEL_PATH = BASE_DIR / "finetune" / "model_output" / "final"
FT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_SCHEMA_TEXT = (
    "patients(patient_id, first_name, last_name, gender, birth_date, city, phone)\n"
    "doctors(doctor_id, first_name, last_name, specialty, department_id, email)\n"
    "appointments(appointment_id, patient_id, doctor_id, department_id, "
    "appointment_date, reason, diagnosis, status)\n"
    "prescriptions(prescription_id, appointment_id, patient_id, "
    "medication_name, dosage, duration_days)\n"
    "departments(department_id, department_name, floor_number, building_name)"
)
_FT_INSTRUCTION = "Generate a valid SQLite SELECT query using only the provided schema."


# ── Generateurs ────────────────────────────────────────────────────────────────

class MistralGenerator:
    """Genere via Ollama local (Mistral 7B)."""
    def __init__(self) -> None:
        from execute_sql import SQLGenerator as _Gen
        self._gen = _Gen()

    def generate(self, prompt: str) -> str:
        return self._gen.generate(prompt)

    def regenerate_with_feedback(self, **kwargs) -> str:
        return self._gen.regenerate_with_feedback(**kwargs)


class ClaudeGenerator:
    """Genere via Anthropic API (Claude)."""
    def __init__(self, model: str = "claude-3-5-haiku-20241022") -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("Installe anthropic: pip install anthropic")
        import os
        self._model = model
        self._client = _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def _call(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    def generate(self, prompt: str) -> str:
        return self._call(prompt)

    def regenerate_with_feedback(
        self, base_prompt: str, failed_sql: str,
        error_message: str, schema_summary: str,
    ) -> str:
        repair = (
            f"{base_prompt}\n\n---\n\n"
            f"PREVIOUS SQL (FAILED):\n{failed_sql}\n\n"
            f"ERROR:\n{error_message}\n\n"
            f"SCHEMA:\n{schema_summary}\n\n"
            "Fix the SQL. Output ONLY SQL or NO_SQL."
        )
        return self._call(repair)


class TinyLlamaGenerator:
    """Genere via TinyLlama fine-tune (LoRA)."""
    def __init__(self, model_path: Path) -> None:
        print(f"[INFO] Chargement TinyLlama fine-tune: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(FT_BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            FT_BASE_MODEL, torch_dtype=torch.float32, device_map="cpu"
        )
        self.model = PeftModel.from_pretrained(base, str(model_path))
        self.model.eval()
        print("[INFO] TinyLlama charge.")

    def _generate_raw(self, question: str) -> str:
        input_text = f"Question: {question}\nSchema:\n{_SCHEMA_TEXT}"
        prompt = (
            f"### Instruction:\n{_FT_INSTRUCTION}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                temperature=1.0, pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        response = re.sub(r"```(?:sql)?", "", response, flags=re.IGNORECASE).strip("`").strip()
        m = re.search(r"(SELECT\b.*)", response, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else response

    def generate(self, prompt: str) -> str:
        # Pour TinyLlama on extrait la question depuis le prompt complet
        m = re.search(r"Question:\s*(.+?)(?:\n|Schema|$)", prompt, re.IGNORECASE)
        question = m.group(1).strip() if m else prompt
        return self._generate_raw(question)

    def regenerate_with_feedback(
        self, base_prompt: str, failed_sql: str,
        error_message: str, schema_summary: str,
    ) -> str:
        # TinyLlama est trop petit pour corriger — on regenere simplement
        return self.generate(base_prompt)


# ── Helpers scores ─────────────────────────────────────────────────────────────

def normalize_cols(cols: List) -> List[str]:
    return [str(c).strip().lower() for c in cols]

def normalize_rows(rows: List) -> List[Tuple]:
    return [tuple(str(v).strip().lower() for v in r) for r in rows]

def result_exact_match(exp_c, exp_r, act_c, act_r) -> int:
    return 1 if (normalize_cols(exp_c) == normalize_cols(act_c)
                 and normalize_rows(exp_r) == normalize_rows(act_r)) else 0

def result_set_match(exp_c, exp_r, act_c, act_r) -> int:
    if normalize_cols(exp_c) != normalize_cols(act_c):
        return 0
    return 1 if sorted(normalize_rows(exp_r)) == sorted(normalize_rows(act_r)) else 0

def semantic_score(sql: str, required: List[str], forbidden: List[str]) -> float:
    sql_l = sql.lower()
    if not required and not forbidden:
        return 1.0
    checks, passed = 0, 0
    for t in required:
        checks += 1
        if t.lower() in sql_l:
            passed += 1
    for t in forbidden:
        checks += 1
        if t.lower() not in sql_l:
            passed += 1
    return passed / checks if checks else 1.0


# ── Pipeline principal ─────────────────────────────────────────────────────────

def eval_model(
    model_name: str,
    generator: Any,
    dataset: List[Dict],
    inspector: SchemaInspector,
    validator: SQLValidator,
    executor: SQLExecutor,
    hybrid: HybridRetriever,
    reranker: HeuristicReranker,
    constructor: QueryConstructor,
    router: QueryRouter,
) -> Tuple[List[Dict], Dict]:
    """Evalue un modele sur dataset_20.json avec le pipeline complet."""

    planner = ChainOfThoughtPlanner(generator.generate)
    per_item: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name.upper()}")
    print(f"{'='*60}")

    for item in dataset:
        qid          = item["id"]
        question     = item["question"]
        expected_sql = item["expected_sql"]
        required     = item.get("required_sql_contains", [])
        forbidden    = item.get("forbidden_sql_contains", [])

        route = router.route(question)

        # Trace Langfuse par question
        trace = _langfuse.trace(
            name="eval-item",
            input={"id": qid, "question": question},
            metadata={"model": model_name, "dataset": "dataset_20.json"},
            tags=[model_name, "eval-all-models"],
        ) if _langfuse else None

        generated_sql = ""
        validation_msg = ""
        is_valid = False
        attempts = 0
        execution_ok = 0
        exact = 0
        set_match = 0
        sem = 0.0
        error = ""
        start = time.time()

        try:
            hybrid_results = hybrid.search(question)
            reranked = reranker.rerank(question, hybrid_results, top_k_final=8)
            cot_plan = planner.plan(question, inspector.summary())
            prompt = constructor.build_context(question, reranked, cot_plan=cot_plan)

            for attempt in range(1, 4):
                attempts = attempt
                if attempt == 1:
                    raw = generator.generate(prompt)
                else:
                    raw = generator.regenerate_with_feedback(
                        base_prompt=prompt,
                        failed_sql=generated_sql,
                        error_message=validation_msg,
                        schema_summary=inspector.summary(),
                    )
                generated_sql = validator.clean_sql(raw)
                is_valid, validation_msg = validator.validate(generated_sql)
                if is_valid:
                    break

            sql_valid = 1 if is_valid else 0

            if is_valid:
                act_c, act_r = executor.execute(generated_sql)
                execution_ok = 1
                exp_c, exp_r = executor.execute(expected_sql)
                exact     = result_exact_match(exp_c, exp_r, act_c, act_r)
                set_match = result_set_match(exp_c, exp_r, act_c, act_r)
            sem = semantic_score(generated_sql, required, forbidden)

        except Exception as exc:
            sql_valid = 1 if is_valid else 0
            error = str(exc)

        duration = round(time.time() - start, 2)

        final_score = (
            0.15 * sql_valid
            + 0.15 * execution_ok
            + 0.40 * exact
            + 0.20 * set_match
            + 0.10 * sem
        )

        row = {
            "id": qid,
            "question": question,
            "model": model_name,
            "route": route,
            "attempts": attempts,
            "duration_seconds": duration,
            "generated_sql": generated_sql,
            "expected_sql": expected_sql,
            "error": error,
            "scores": {
                "sql_valid":          sql_valid,
                "execution_ok":       execution_ok,
                "result_exact_match": exact,
                "result_set_match":   set_match,
                "semantic_score":     round(sem, 4),
                "final_score":        round(final_score, 4),
            },
        }
        per_item.append(row)

        # Scores Langfuse par question
        if trace:
            trace.score(name="sql_valid",           value=sql_valid)
            trace.score(name="execution_ok",        value=execution_ok)
            trace.score(name="result_exact_match",  value=exact)
            trace.score(name="result_set_match",    value=set_match)
            trace.score(name="semantic_score",      value=float(round(sem, 4)))
            trace.score(name="final_score",         value=float(round(final_score, 4)))
            trace.update(output={
                "generated_sql": generated_sql,
                "expected_sql":  expected_sql,
                "error":         error,
                "scores":        row["scores"],
            })
            _langfuse.flush()

        print(
            f"  [{qid}] valid={sql_valid} exec={execution_ok} "
            f"exact={exact} set={set_match} "
            f"sem={sem:.2f} final={final_score:.2f} ({duration}s)"
        )

    n = len(per_item)
    summary = {
        "model":                    model_name,
        "count":                    n,
        "avg_sql_valid":            round(sum(r["scores"]["sql_valid"]          for r in per_item) / n, 4),
        "avg_execution_ok":         round(sum(r["scores"]["execution_ok"]       for r in per_item) / n, 4),
        "avg_result_exact_match":   round(sum(r["scores"]["result_exact_match"] for r in per_item) / n, 4),
        "avg_result_set_match":     round(sum(r["scores"]["result_set_match"]   for r in per_item) / n, 4),
        "avg_semantic_score":       round(sum(r["scores"]["semantic_score"]     for r in per_item) / n, 4),
        "avg_final_score":          round(sum(r["scores"]["final_score"]        for r in per_item) / n, 4),
        "avg_duration":             round(sum(r["duration_seconds"]             for r in per_item) / n, 2),
    }

    # Trace de synthese Langfuse par modele
    if _langfuse:
        strace = _langfuse.trace(
            name="eval-summary",
            input={"model": model_name, "count": n},
            output=summary,
            metadata={"model": model_name},
            tags=[model_name, "eval-all-models", "summary"],
        )
        strace.score(name="avg_final_score",        value=float(summary["avg_final_score"]))
        strace.score(name="avg_result_exact_match", value=float(summary["avg_result_exact_match"]))
        strace.score(name="avg_result_set_match",   value=float(summary["avg_result_set_match"]))
        _langfuse.flush()

    return per_item, summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation unifiee Mistral / Claude / TinyLlama sur dataset_20.json"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["mistral", "claude", "tinyllama"],
        choices=["mistral", "claude", "tinyllama"],
        help="Modeles a evaluer",
    )
    parser.add_argument(
        "--ft-model", type=Path, default=FT_MODEL_PATH,
        help="Chemin vers le modele TinyLlama fine-tune",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Nombre de questions max (0 = toutes)",
    )
    args = parser.parse_args()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit > 0:
        dataset = dataset[: args.limit]

    # Ressources partagees
    inspector = SchemaInspector(DB_PATH)
    inspector.load()
    validator  = SQLValidator(inspector)
    executor   = SQLExecutor(DB_PATH)
    hybrid     = HybridRetriever()
    reranker   = HeuristicReranker()
    constructor = QueryConstructor()
    router     = QueryRouter()

    all_summaries: Dict[str, Dict] = {}
    all_results:   Dict[str, List] = {}

    try:
        for model_name in args.models:
            if model_name == "mistral":
                gen = MistralGenerator()
            elif model_name == "claude":
                gen = ClaudeGenerator()
            else:
                gen = TinyLlamaGenerator(args.ft_model)

            items, summary = eval_model(
                model_name, gen, dataset,
                inspector, validator, executor,
                hybrid, reranker, constructor, router,
            )
            all_summaries[model_name] = summary
            all_results[model_name]   = items

    finally:
        hybrid.close()

    # ── Tableau comparatif ───────────────────────────────────────────────────
    metrics = [
        "avg_sql_valid", "avg_execution_ok",
        "avg_result_exact_match", "avg_result_set_match",
        "avg_semantic_score", "avg_final_score", "avg_duration",
    ]
    models = list(all_summaries.keys())

    print("\n" + "=" * 75)
    print("COMPARAISON FINALE — meme pipeline, meme 20 questions")
    print("=" * 75)
    header = f"{'Metric':<28}" + "".join(f"{m:>15}" for m in models)
    print(header)
    print("-" * 75)
    for metric in metrics:
        line = f"{metric:<28}"
        for m in models:
            val = all_summaries[m].get(metric, 0)
            if metric == "avg_duration":
                line += f"{val:>15.2f}s"
            else:
                line += f"{val:>15.2%}"
        print(line)
    print("=" * 75)

    # ── Rapport JSON ─────────────────────────────────────────────────────────
    report = {
        "dataset":  str(DATASET_PATH),
        "models":   models,
        "summary":  all_summaries,
        "results":  all_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nRapport sauvegarde -> {OUTPUT_PATH}")

    # Trace de comparaison globale dans Langfuse
    if _langfuse:
        _langfuse.trace(
            name="eval-global-comparison",
            input={"models": models, "questions": len(dataset)},
            output=all_summaries,
            tags=["eval-all-models", "comparison"],
        )
        _langfuse.flush()
        print("Traces envoyees dans Langfuse ✓")


if __name__ == "__main__":
    main()
