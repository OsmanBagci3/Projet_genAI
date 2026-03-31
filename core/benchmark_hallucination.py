"""Run hallucination analysis on benchmark results."""

import json
from pathlib import Path

from core.hallucination_detector import HallucinationDetector

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BASE_DIR / "evaluation" / "benchmark_report.json"
OUTPUT_PATH = BASE_DIR / "evaluation" / "hallucination_report.json"
DB_PATH = BASE_DIR / "hospital.db"


def analyze():
    """Analyze benchmark results for hallucinations."""
    with open(BENCHMARK_PATH, "r") as f:
        benchmark = json.load(f)

    detector = HallucinationDetector(str(DB_PATH))
    report = {}

    for provider, data in benchmark["providers"].items():
        separator = "=" * 60
        print("\n" + separator)
        print("HALLUCINATION ANALYSIS: " + provider.upper())
        print(separator)

        results = []
        total_hall = 0
        total_critical = 0
        total_warning = 0

        for item in data["results"]:
            sql = item.get("generated_sql", "")
            if not sql:
                continue

            analysis = detector.detect(sql)
            analysis["id"] = item["id"]
            analysis["question"] = item["question"]
            results.append(analysis)

            total_hall += analysis["total_hallucinations"]
            total_critical += analysis["critical_count"]
            total_warning += analysis["warning_count"]

            if analysis["total_hallucinations"] > 0:
                print("\n[" + item["id"] + "] " + item["question"])
                for h in analysis["hallucinations"]:
                    print(
                        "  ["
                        + h["severity"].upper()
                        + "] "
                        + h["type"]
                        + ": "
                        + h["detail"]
                    )

        n = len(results)
        avg_score = sum(r["hallucination_score"] for r in results) / n if n else 0

        summary = {
            "total_queries": n,
            "total_hallucinations": total_hall,
            "critical_hallucinations": total_critical,
            "warning_hallucinations": total_warning,
            "avg_hallucination_score": round(avg_score, 4),
            "hallucination_rate": (
                round(sum(1 for r in results if r["total_hallucinations"] > 0) / n, 4)
                if n
                else 0
            ),
        }

        report[provider] = {
            "summary": summary,
            "details": results,
        }

        print("\n--- Summary ---")
        for k, v in summary.items():
            print("  " + k + ": " + str(v))

    separator = "=" * 60
    print("\n" + separator)
    print("HALLUCINATION COMPARISON")
    print(separator)
    header = "{:<30}".format("Metric")
    for prov in report:
        header += "{:>15}".format(prov)
    print(header)
    print("-" * 60)

    metrics = [
        "total_hallucinations",
        "critical_hallucinations",
        "warning_hallucinations",
        "avg_hallucination_score",
        "hallucination_rate",
    ]
    for m in metrics:
        line = "{:<30}".format(m)
        for prov in report:
            val = report[prov]["summary"][m]
            line += "{:>15}".format(val)
        print(line)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nReport saved: " + str(OUTPUT_PATH))


if __name__ == "__main__":
    analyze()
