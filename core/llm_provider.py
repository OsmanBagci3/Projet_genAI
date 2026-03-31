"""LLM provider abstraction: switch between Ollama/Mistral and Claude."""

import os
from typing import Literal

import requests

LLMProvider = Literal["mistral", "claude"]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


def generate(prompt: str, provider: str = "") -> str:
    """Generate text using the configured LLM provider.

    Args:
        prompt: The prompt to send to the LLM.
        provider: Override provider ('mistral' or 'claude').

    Returns:
        Generated text response.
    """
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "mistral")

    if provider == "claude":
        return _generate_claude(prompt)
    return _generate_mistral(prompt)


def _generate_mistral(prompt: str) -> str:
    """Generate using local Ollama/Mistral."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"[ERREUR MISTRAL] {e}"


def _generate_claude(prompt: str) -> str:
    """Generate using Claude API."""
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"[ERREUR CLAUDE] {e}"


def list_providers() -> list:
    """List available providers."""
    available = []
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            available.append("mistral")
    except Exception:
        pass
    if os.getenv("ANTHROPIC_API_KEY"):
        available.append("claude")
    return available
