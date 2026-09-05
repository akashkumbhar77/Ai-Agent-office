"""Configuration. Environment only — no hardcoded ports, paths, or model IDs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---------------------------------------------------------------
    openai_api_key: str | None = None
    # Optional override for gateways and compatible endpoints.
    openai_base_url: str | None = None

    # Deliberately no defaults, and min_length=1 so a blank value in .env is
    # rejected too. A guessed or empty model ID fails at request time with a
    # 404 buried in a retry loop; a missing one fails at startup with a clear
    # message. Set both in .env.
    planning_model: str = Field(min_length=1)
    utility_model: str = Field(min_length=1)

    max_tokens: int = 8192
    max_llm_retries: int = 4

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

    @model_validator(mode="after")
    def _require_provider_credentials(self) -> Settings:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is unset. Set it in backend/.env (gitignored) "
                "or export it."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
