"""Shared fixtures for NLQ-to-SQL tests."""

import pytest


@pytest.fixture
def sample_query():
    """Sample SQL query."""
    return "Which patients received Paracetamol?"


@pytest.fixture
def sample_state():
    """Sample pipeline state."""
    return {
        "query": "Which patients received Paracetamol?",
        "route": "",
        "translated_queries": [],
        "retrieval_results": [],
        "reranked_results": [],
        "cot_plan": "",
        "prompt": "",
        "generated_sql": "",
        "is_valid": False,
        "validation_message": "",
        "columns": [],
        "rows": [],
        "attempts": 0,
        "provider": "claude",
        "error": "",
        "status": "started",
    }
