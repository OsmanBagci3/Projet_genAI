from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

try:
    from langfuse import Langfuse
    _langfuse: Optional[Langfuse] = Langfuse()
except Exception:
    _langfuse = None

from execute_sql import (
    ChainOfThoughtPlanner,
    DB_PATH,
    SQLExecutor,
    SQLGenerator,
    SQLValidator,
    SchemaInspector,
)
from hybrid_retrieval import HybridRetriever
from query_construction import QueryConstructor
from rerank_results import HeuristicReranker
from router import QueryRouter

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "evaluation" / "dataset_20.json"
DEFAULT_OUTPUT = BASE_DIR / "evaluation" / "evaluation_report.json"


def normalize_columns(columns: List[str]) -> List[str]:
    return [str(c).strip().lower() for c in columns]


def normalize_rows(rows: List[Tuple[Any, ...]]) -> List[Tuple[str, ...]]:
    return [tuple(str(v).strip().lower() for v in row) for row in rows]


def result_exact_match(
    expected_cols: List[str],
    expected_rows: List[Tuple[Any, ...]],
    actual_cols: List[str],
    actual_rows: List[Tuple[Any, ...]],
) -> int:
    exp_c = normalize_columns(expected_cols)
    act_c = normalize_columns(actual_cols)
    exp_r = normalize_rows(expected_rows)
    act_r = normalize_rows(actual_rows)
    return 1 if (exp_c == act_c and exp_r == act_r) else 0


def result_set_match(
    expected_cols: List[str],
    expected_rows: List[Tuple[Any, ...]],
    actual_cols: List[str],
    actual_rows: List[Tuple[Any, ...]],
) -> int:
    exp_c = normalize_columns(expected_cols)
    act_c = normalize_columns(actual_cols)
    if exp_c != act_c:
        return 0
    exp_r = sorted(normalize_rows(expected_rows))
    act_r = sorted(normalize_rows(actual_rows))
    return 1 if exp_r == act_r else 0


def semantic_score(sql: str, required: List[str], forbidden: List[str]) -> float:
    sql_l = sql.lower()

    if not required and not forbidden:
        return 1.0

    checks = 0
    passed = 0

    for token in required:
        checks += 1
        if token.lower() in sql_l:
            passed += 1

    for token in forbidden:
        checks += 1
        if token.lower() not in sql_l:
            passed += 1

    return passed / checks if checks else 1.0


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(dataset_path: Path, output_path: Path, limit: int = 0) -> None:
    dataset = load_dataset(dataset_path)
    if limit > 0:
        dataset = dataset[:limit]

    router = QueryRouter()
    hybrid = HybridRetriever()
    reranker = HeuristicReranker()
    constructor = QueryConstructor()
    generator = SQLGenerator()
    planner = ChainOfThoughtPlanner(generator.generate)

    inspector = SchemaInspector(DB_PATH)
    inspector.load()
    validator = SQLValidator(inspector)
    executor = SQLExecutor(DB_PATH)

    per_item: List[Dict[str, Any]] = []

    try:
        for item in dataset:
            qid = item["id"]
            question = item["question"]
            expected_sql = item["expected_sql"]
            required = item.get("required_sql_contains", [])
            forbidden = item.get("forbidden_sql_contains", [])

            route = router.route(question)

            trace = _langfuse.trace(
                name="evaluation-item",
                input={"id": qid, "question": question},
                metadata={"dataset": str(dataset_path)},
            ) if _langfuse else None

            if trace:
                trace.event(name="routing", output={"route": route})

            generated_sql = ""
            validation_msg = ""
            is_valid = False
            attempts = 0
            execution_ok = 0
            exact = 0
            set_match = 0
            sem = 0.0
            error = ""

            try:
                # Build prompt with same production pipeline.
                hybrid_results = hybrid.search(question)
                reranked = reranker.rerank(question, hybrid_results, top_k_final=8)
                if trace:
                    trace.event(
                        name="retrieval",
                        output={"top_chunks": [c.get("content_id") for c in reranked]},
                    )

                cot_plan = planner.plan(question, inspector.summary())
                prompt = constructor.build_context(question, reranked, cot_plan=cot_plan)
                if trace:
                    trace.event(name="prompt", output={"prompt": prompt})

                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    attempts = attempt
                    raw_sql = generator.generate(prompt) if attempt == 1 else generator.regenerate_with_feedback(
                        base_prompt=prompt,
                        failed_sql=generated_sql,
                        error_message=validation_msg,
                        schema_summary=inspector.summary(),
                    )
                    generated_sql = validator.clean_sql(raw_sql)
                    is_valid, validation_msg = validator.validate(generated_sql)

                    if trace:
                        trace.event(
                            name="sql_generation",
                            output={
                                "attempt": attempt,
                                "sql": generated_sql,
                                "valid": is_valid,
                                "validation_message": validation_msg,
                            },
                        )

                    if is_valid:
                        break

                sql_valid = 1 if is_valid else 0

                if is_valid:
                    actual_cols, actual_rows = executor.execute(generated_sql)
                    execution_ok = 1
                    expected_cols, expected_rows = executor.execute(expected_sql)
                    exact = result_exact_match(expected_cols, expected_rows, actual_cols, actual_rows)
                    set_match = result_set_match(expected_cols, expected_rows, actual_cols, actual_rows)
                    sem = semantic_score(generated_sql, required, forbidden)
                    if trace:
                        trace.event(
                            name="execution",
                            output={
                                "columns": actual_cols,
                                "rows": [list(r) for r in actual_rows[:5]],
                            },
                        )
                else:
                    sem = semantic_score(generated_sql, required, forbidden)

            except Exception as exc:
                sql_valid = 1 if is_valid else 0
                error = str(exc)
                if trace:
                    trace.event(name="error", output={"error": error})

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
                "route": route,
                "attempts": attempts,
                "generated_sql": generated_sql,
                "expected_sql": expected_sql,
                "validation_message": validation_msg,
                "error": error,
                "scores": {
                    "sql_valid": sql_valid,
                    "execution_ok": execution_ok,
                    "result_exact_match": exact,
                    "result_set_match": set_match,
                    "semantic_score": round(sem, 4),
                    "final_score": round(final_score, 4),
                },
            }
            per_item.append(row)

            if trace:
                trace.score(name="sql_valid", value=sql_valid)
                trace.score(name="execution_success", value=execution_ok)
                trace.score(name="no_error", value=0 if error else 1)
                trace.score(name="result_exact_match", value=exact)
                trace.score(name="result_set_match", value=set_match)
                trace.score(name="semantic_score", value=float(round(sem, 4)))
                trace.score(name="final_score", value=float(round(final_score, 4)))
                trace.update(
                    output={
                        "attempts": attempts,
                        "generated_sql": generated_sql,
                        "expected_sql": expected_sql,
                        "validation_message": validation_msg,
                        "error": error,
                        "scores": row["scores"],
                    }
                )
                _langfuse.flush()

            print(
                f"[{qid}] sql_valid={row['scores']['sql_valid']} "
                f"exec={row['scores']['execution_ok']} "
                f"exact={row['scores']['result_exact_match']} "
                f"set={row['scores']['result_set_match']} "
                f"sem={row['scores']['semantic_score']:.2f} "
                f"final={row['scores']['final_score']:.2f}"
            )

    finally:
        hybrid.close()

    n = len(per_item)
    summary = {
        "count": n,
        "avg_sql_valid": round(sum(x["scores"]["sql_valid"] for x in per_item) / n, 4) if n else 0.0,
        "avg_execution_ok": round(sum(x["scores"]["execution_ok"] for x in per_item) / n, 4) if n else 0.0,
        "avg_result_exact_match": round(sum(x["scores"]["result_exact_match"] for x in per_item) / n, 4) if n else 0.0,
        "avg_result_set_match": round(sum(x["scores"]["result_set_match"] for x in per_item) / n, 4) if n else 0.0,
        "avg_semantic_score": round(sum(x["scores"]["semantic_score"] for x in per_item) / n, 4) if n else 0.0,
        "avg_final_score": round(sum(x["scores"]["final_score"] for x in per_item) / n, 4) if n else 0.0,
    }

    report = {
        "dataset": str(dataset_path),
        "summary": summary,
        "results": per_item,
    }

    if _langfuse:
        summary_trace = _langfuse.trace(
            name="evaluation-summary",
            input={"dataset": str(dataset_path), "count": n},
            output=summary,
        )
        summary_trace.score(name="avg_final_score", value=float(summary["avg_final_score"]))
        summary_trace.score(name="avg_result_exact_match", value=float(summary["avg_result_exact_match"]))
        _langfuse.flush()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nReport saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NLQ-to-SQL pipeline with a labeled dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to dataset JSON file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output report JSON")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of questions (0 = all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args.dataset, args.output, args.limit)
