"""Register the NLQ-to-SQL pipeline as an MLflow model."""

import mlflow


def register_model(run_id: str, model_name: str = "nlq-to-sql"):
    """Register the pipeline as an MLflow model.

    Args:
        run_id: The MLflow run ID to register.
        model_name: Name for the registered model.
    """
    try:
        model_uri = "runs:/" + run_id + "/model"
        mlflow.register_model(model_uri, model_name)
        print("Model registered: " + model_name)
    except Exception as e:
        print("Model registration skipped: " + str(e))


def log_pipeline_as_model():
    """Log pipeline info as a model artifact."""
    import json

    pipeline_info = {
        "name": "nlq-to-sql",
        "version": "1.0.0",
        "agents": [
            "router",
            "translator",
            "retriever",
            "reranker",
            "planner",
            "constructor",
            "generator",
            "executor",
        ],
        "supported_providers": ["mistral", "claude"],
        "orchestration": "LangGraph",
        "retrieval": "Hybrid (BM25 + Qdrant)",
    }
    with open("pipeline_model.json", "w") as f:
        json.dump(pipeline_info, f, indent=2)
    mlflow.log_artifact("pipeline_model.json")
