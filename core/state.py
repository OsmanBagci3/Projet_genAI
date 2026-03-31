"""State definition for the NLQ-to-SQL multi-agent graph."""

from typing import TypedDict


class NLQState(TypedDict):
    """Global state shared across all agents."""

    query: str
    route: str
    translated_queries: list
    retrieval_results: list
    reranked_results: list
    cot_plan: str
    prompt: str
    generated_sql: str
    is_valid: bool
    validation_message: str
    columns: list
    rows: list
    attempts: int
    provider: str
    error: str
    status: str
