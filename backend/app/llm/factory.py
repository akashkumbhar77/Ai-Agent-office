"""Provider construction.

The `LLMProvider` seam stays — it earned its place when this project switched
providers, and FakeProvider depends on it — but there is only one real
implementation. A second one arrives when a second one is actually needed.
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMProvider


def build_provider(settings: Settings) -> LLMProvider:
    from app.llm.openai_provider import OpenAIProvider

    # Presence is enforced by Settings validation; assert for the type checker.
    assert settings.openai_api_key is not None
    return OpenAIProvider(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url
    )
