"""Persona definitions.

System prompts live in `prompts/*.md` and are read once at import. They are
files rather than f-strings on purpose: interpolating anything — a timestamp,
a task id, an agent position — into a system prompt destroys prompt-cache hits
on every call thereafter (CLAUDE.md §4). Per-task context belongs in the
message list.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from app.protocol.events import Persona

PROMPT_DIR = Path(__file__).parent / "prompts"

ModelTier = Literal["planning", "utility"]


@cache
def load_prompt(name: str) -> str:
    """Read a persona prompt. Cached so the exact same string object is reused
    every turn — the cheapest possible guarantee of byte-stability."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"missing persona prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class PersonaSpec:
    agent_id: str
    persona: Persona
    display_name: str
    prompt_name: str
    model_tier: ModelTier
    # Which tools this persona may call. Narrower is better: a tool the
    # persona cannot use is a tool it cannot misuse, and every extra schema
    # costs context on every call.
    tool_names: tuple[str, ...]

    @property
    def system_prompt(self) -> str:
        return load_prompt(self.prompt_name)


PM = PersonaSpec(
    agent_id="pm-1",
    persona=Persona.PM,
    display_name="Iris",
    prompt_name="pm",
    model_tier="planning",
    # Decomposition only — no filesystem access at all.
    tool_names=("create_tasks",),
)

CODER = PersonaSpec(
    agent_id="coder-1",
    persona=Persona.ARCHITECT,
    display_name="Ada",
    prompt_name="coder",
    model_tier="planning",
    tool_names=("read_file", "write_file", "edit_file", "list_dir", "run_command"),
)

REVIEWER = PersonaSpec(
    agent_id="reviewer-1",
    persona=Persona.REVIEWER,
    display_name="Bo",
    prompt_name="reviewer",
    model_tier="planning",
    # Read-only plus the verdict tool: a reviewer that can edit the code it is
    # reviewing stops being a review.
    tool_names=("read_file", "list_dir", "run_command", "submit_review"),
)

WRITER = PersonaSpec(
    agent_id="writer-1",
    persona=Persona.WRITER,
    display_name="Cy",
    prompt_name="writer",
    model_tier="utility",
    tool_names=("read_file", "write_file", "edit_file", "list_dir"),
)

ROSTER: tuple[PersonaSpec, ...] = (PM, CODER, REVIEWER, WRITER)
BY_ID: dict[str, PersonaSpec] = {spec.agent_id: spec for spec in ROSTER}
