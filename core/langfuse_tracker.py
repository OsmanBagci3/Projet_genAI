"""Langfuse tracing for NLQ-to-SQL pipeline."""

import os

from dotenv import load_dotenv

load_dotenv()

try:
    from langfuse import Langfuse

    _client = Langfuse()
except Exception:
    _client = None


def start_trace(query, provider=""):
    """Start a new trace."""
    if _client:
        return _client.trace(
            name="nlq-to-sql",
            input={"question": query, "provider": provider},
        )
    return None


def log_event(trace, name, output):
    """Log an event to a trace."""
    if trace:
        trace.event(name=name, output=output)


def end_trace(trace, output):
    """End a trace."""
    if trace:
        trace.update(output=output)
    if _client:
        _client.flush()
