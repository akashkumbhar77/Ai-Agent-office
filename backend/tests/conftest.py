"""Test environment.

Settings deliberately has no defaults for credentials or model IDs, so tests
must supply them. Real values are never needed: nothing in the suite reaches a
provider — the FakeProvider stands in at the LLMProvider seam.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def _test_environment(tmp_path_factory: pytest.TempPathFactory) -> None:
    workspace: Path = tmp_path_factory.mktemp("agent-workspace")

    # Real env vars take precedence over .env in pydantic-settings, so this
    # isolates the suite from whatever the developer has configured locally.
    os.environ.update(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key-not-real",
            "PLANNING_MODEL": "test-planning-model",
            "UTILITY_MODEL": "test-utility-model",
            "WORKSPACE_ROOT": str(workspace),
        }
    )
    get_settings.cache_clear()
