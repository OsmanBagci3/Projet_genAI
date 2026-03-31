"""MLflow tracking and tracing for NLQ-to-SQL pipeline."""

import time
from contextlib import contextmanager
from typing import Generator

import mlflow


class NLQTracker:
    """Track NLQ-to-SQL pipeline metrics with MLflow."""

    def __init__(
        self,
        experiment_name: str = "nlq-to-sql",
        tracking_uri: str = "mlruns",
    ):
        """Initialize the MLflow tracker."""
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.step_times: dict[str, float] = {}

    @contextmanager
    def track_run(self, query: str, provider: str) -> Generator:
        """Context manager to track a full pipeline run."""
        with mlflow.start_run() as run:
            mlflow.log_param("query", query[:250])
            mlflow.log_param("provider", provider)
            self._run_start = time.time()
            yield run
            total = time.time() - self._run_start
            mlflow.log_metric("total_duration_seconds", total)

    def log_step(self, step_name: str, duration: float) -> None:
        """Log metrics for a pipeline step."""
        self.step_times[step_name] = duration
        mlflow.log_metric(f"{step_name}_duration_seconds", duration)

    def log_results(
        self,
        route: str,
        is_valid: bool,
        num_rows: int,
        attempts: int,
        sql: str,
    ) -> None:
        """Log result metrics."""
        mlflow.log_param("route", route)
        mlflow.log_metric("sql_valid", 1 if is_valid else 0)
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("attempts", attempts)
        mlflow.log_metric("sql_length", len(sql))

    def log_trace(
        self,
        query: str,
        provider: str,
        route: str,
        sql: str,
        is_valid: bool,
        num_rows: int,
        duration: float,
    ) -> None:
        """Log a trace for the pipeline run."""
        with open("temp_trace.txt", "w") as f:
            f.write("Query: " + query + "\n")
            f.write("Provider: " + provider + "\n")
            f.write("Route: " + route + "\n")
            f.write("SQL Valid: " + str(is_valid) + "\n")
            f.write("Num Rows: " + str(num_rows) + "\n")
            f.write("Duration: " + str(round(duration, 2)) + "s\n")
            f.write("\nGenerated SQL:\n" + sql + "\n")
        mlflow.log_artifact("temp_trace.txt")
