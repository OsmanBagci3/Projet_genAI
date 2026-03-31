"""Tests for the semantic router."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router import SemanticRouter


def test_router_sql_query():
    """Test that SQL queries are routed correctly."""
    router = SemanticRouter()
    route, score, _ = router.route("How many patients are there?")
    assert route == "SQL_QUERY"
    assert score > 0.5


def test_router_schema_help():
    """Test that schema questions are routed correctly."""
    router = SemanticRouter()
    route, score, _ = router.route("What columns exist in doctors?")
    assert route == "SCHEMA_HELP"


def test_router_out_of_scope():
    """Test that out-of-scope questions are detected."""
    router = SemanticRouter()
    route, score, _ = router.route("What is the weather today?")
    assert route == "OUT_OF_SCOPE"
