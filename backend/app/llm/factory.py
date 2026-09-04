"""Provider selection.

The single place that knows which concrete provider exists. Agent code takes
an LLMProvider and never branches on which one it got — that branch living in
one function is what keeps provider quirks from spreading (CLAUDE.md §5).
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        # Presence is enforced by Settings validation; assert for the type
        # checker rather than re-checking at runtime.
        assert settings.openai_api_key is not None
        return OpenAIProvider(
            api_key=settings.openai_api_key, base_url=settings.openai_base_url
        )

    if settings.llm_provider == "ollama":
        from app.llm.ollama_provider import OllamaProvider

        return OllamaProvider(base_url=settings.ollama_base_url)

    raise ValueError(f"unknown provider: {settings.llm_provider}")
