"""Configuration. Environment only — no hardcoded ports, paths, or model IDs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---------------------------------------------------------------
    llm_provider: Literal["anthropic", "ollama"] = "anthropic"
    anthropic_api_key: str | None = None
    planning_model: str = "claude-opus-5"
    utility_model: str = "claude-haiku-4-5"
    ollama_base_url: str = "http://localhost:11434"

    # Thinking is on by default on claude-opus-5 and shares max_tokens with the
    # response text. Sized for streaming; see CLAUDE.md §4.
    max_tokens: int = 64000
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"

    # --- Agent safety ------------------------------------------------------
    workspace_root: Path
    max_steps_per_subtask: int = 10
    max_tool_retries: int = 3

    # --- Transport ---------------------------------------------------------
    tick_interval_ms: int = 100
    host: str = "127.0.0.1"
    port: int = 8000

    # --- World -------------------------------------------------------------
    map_id: str = "office_v1"

    @field_validator("workspace_root")
    @classmethod
    def _resolve_workspace(cls, v: Path) -> Path:
        """Resolve to canonical form up front so every later path check
        compares against a real absolute path (CLAUDE.md §8)."""
        resolved = v.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"WORKSPACE_ROOT is not a directory: {resolved}")
        return resolved

    def require_anthropic_key(self) -> str:
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is unset")
        return self.anthropic_api_key or ""


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
