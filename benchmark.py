"""Benchmark Mistral vs Claude on the evaluation dataset."""

import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from core.llm_provider import generate
from execute_sql import (
    DB_PATH,
    SchemaInspector,
    SQLExecutor,
    SQLValidator,
)
from hybrid_retrieval import HybridRetriever
from query_construction import QueryConstructor
from rerank_results import HeuristicReranker

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "evaluation" / "dataset_20.json"
OUTPUT_PATH = BASE_DIR / "evaluation" / "benchmark_report.json"


def normalize_cols(cols):
    return [str(c).strip().lower() for c in cols]


def normalize_rows(rows):
    return [tuple(str(v).strip().lower() for v in r) for r in rows]


def result_exact_match(exp_c, exp_r, act_c, act_r):
    return (
        1
        if (
            normalize_cols(exp_c) == normalize_cols(act_c)
            and normalize_rows(exp_r) == normalize_rows(act_r)
        )
        else 0
    )


def result_set_match(exp_c, exp_r, act_c, act_r):
    if normalize_cols(exp_c) != normalize_cols(act_c):
        return 0
    return 1 if (sorted(normalize_rows(exp_r)) == sorted(normalize_rows(act_r))) else 0


def semantic_score(sql, required, forbidden):
    sql_l = sql.lower()
    if not required and not forbidden:
        return 1.0
    checks = 0
    passed = 0
    for t in required:
        checks += 1
        if t.lower() in sql_l:
            passed += 1
    for t in forbidden:
        checks += 1
        if t.lower() not in sql_l:
            passed += 1
    return passed / checks if checks else 1.0


def run_benchmark(providers):
    """Run benchmark for given providers."""
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    inspector = SchemaInspector(DB_PATH)
    inspector.load()
    validator = SQLValidator(inspector)
    executor = SQLExecutor(DB_PATH)
    hybrid = HybridRetriever()
    reranker = HeuristicReranker()
    constructor = QueryConstructor()

    all_results = {}

    try:
        for prov in providers:
            print(f"\n{'='*60}")
            print(f"BENCHMARK: {prov.upper()}")
            print(f"{'='*60}")

            prov_results = []

            for item in dataset:
                qid = item["id"]
                question = item["question"]
                expected_sql = item["expected_sql"]
                required = item.get("required_sql_contains", [])
                forbidden = item.get("forbidden_sql_contains", [])

                print(f"\n[{qid}] {question}")

                start = time.time()

                # Pipeline
                hybrid_results = hybrid.search(question)
                reranked = reranker.rerank(question, hybrid_results, top_k_final=8)

                # CoT planning
                plan_prompt = (
                    "You are a SQL analyst. Analyze:\n\n"
                    f"SCHEMA:\n{inspector.summary()}\n\n"
                    f"QUESTION: {question}\n\n"
                    "1. TABLES NEEDED:\n"
                    "2. COLUMNS NEEDED:\n"
                    "3. FILTERS:\n"
                    "4. JOINS:\n"
                    "Use ONLY schema tables. Be concise."
                )
                cot_plan = generate(plan_prompt, prov)

                prompt = constructor.build_context(
                    question, reranked, cot_plan=cot_plan
                )

                # Generate with retry
                max_attempts = 3
                sql = ""
                is_valid = False
                message = ""

                for attempt in range(1, max_attempts + 1):
                    if attempt == 1:
                        raw = generate(prompt, prov)
                    else:
                        repair = (
                            f"{prompt}\n\n---\n\n"
                            f"PREVIOUS SQL (FAILED):\n{sql}\n\n"
                            f"ERROR:\n{message}\n\n"
                            f"SCHEMA:\n{inspector.summary()}\n\n"
                            "Fix. Output ONLY SQL or NO_SQL."
                        )
                        raw = generate(repair, prov)

                    sql = validator.clean_sql(raw)
                    is_valid, message = validator.validate(sql)
                    if is_valid:
                        break

                duration = time.time() - start

                # Evaluate
                sql_valid = 1 if is_valid else 0
                exec_ok = 0
                exact = 0
                set_m = 0
                sem = semantic_score(sql, required, forbidden)
                error = ""

                if is_valid:
                    try:
                        act_c, act_r = executor.execute(sql)
                        exp_c, exp_r = executor.execute(expected_sql)
                        exec_ok = 1
                        exact = result_exact_match(exp_c, exp_r, act_c, act_r)
                        set_m = result_set_match(exp_c, exp_r, act_c, act_r)
                    except Exception as e:
                        error = str(e)

                final = (
                    0.15 * sql_valid
                    + 0.15 * exec_ok
                    + 0.40 * exact
                    + 0.20 * set_m
                    + 0.10 * sem
                )

                row = {
                    "id": qid,
                    "question": question,
                    "provider": prov,
                    "generated_sql": sql,
                    "expected_sql": expected_sql,
                    "attempts": attempt,
                    "duration_seconds": round(duration, 2),
                    "error": error,
                    "scores": {
                        "sql_valid": sql_valid,
                        "execution_ok": exec_ok,
                        "exact_match": exact,
                        "set_match": set_m,
                        "semantic": round(sem, 4),
                        "final": round(final, 4),
                    },
                }
                prov_results.append(row)

                print(
                    f"  valid={sql_valid} exec={exec_ok} "
                    f"exact={exact} set={set_m} "
                    f"sem={sem:.2f} final={final:.2f} "
                    f"({duration:.1f}s)"
                )

            all_results[prov] = prov_results
    finally:
        hybrid.close()

    # Summary
    report = {"providers": {}}
    for prov, results in all_results.items():
        n = len(results)
        summary = {
            "count": n,
            "avg_sql_valid": round(
                sum(r["scores"]["sql_valid"] for r in results) / n, 4
            ),
            "avg_execution_ok": round(
                sum(r["scores"]["execution_ok"] for r in results) / n, 4
            ),
            "avg_exact_match": round(
                sum(r["scores"]["exact_match"] for r in results) / n, 4
            ),
            "avg_set_match": round(
                sum(r["scores"]["set_match"] for r in results) / n, 4
            ),
            "avg_semantic": round(sum(r["scores"]["semantic"] for r in results) / n, 4),
            "avg_final_score": round(sum(r["scores"]["final"] for r in results) / n, 4),
            "avg_duration": round(sum(r["duration_seconds"] for r in results) / n, 2),
            "total_duration": round(sum(r["duration_seconds"] for r in results), 2),
        }
        report["providers"][prov] = {
            "summary": summary,
            "results": results,
        }

    # Print comparison
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    header = f"{'Metric':<25}"
    for prov in all_results:
        header += f"{prov:>15}"
    print(header)
    print("-" * 70)

    metrics = [
        "avg_sql_valid",
        "avg_execution_ok",
        "avg_exact_match",
        "avg_set_match",
        "avg_semantic",
        "avg_final_score",
        "avg_duration",
    ]
    for m in metrics:
        line = f"{m:<25}"
        for prov in all_results:
            val = report["providers"][prov]["summary"][m]
            line += f"{val:>15}"
        print(line)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["claude"],
        help="Providers to benchmark (mistral claude)",
    )
    args = parser.parse_args()
    run_benchmark(args.providers)
