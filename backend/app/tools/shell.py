"""Shell tool.

Commands are untrusted model output. Three defences, all required
(CLAUDE.md §8):

1. An **allowlist** of executables. A blocklist eventually misses a case; an
   allowlist fails closed on anything unanticipated.
2. Rejection of *unquoted* shell operators. The command runs with shell=False,
   so `&&`, `|`, `;` and `$()` are already inert — they would be passed as
   literal arguments. The check exists because their presence means the agent
   expected a shell, and running a mangled argument list silently is worse
   than saying so. Crucially the check is applied to shlex tokens, not to the
   raw string: `grep "foo|bar"` and `python3 -c 'a; b'` are legitimate and
   must not be rejected for characters inside quotes.
3. A timeout and an output cap, so one command cannot hang the graph or fill
   the context window.

This is not a sandbox. It reduces blast radius inside an already-confined
workspace; running the whole backend in a container is still the right
deployment posture.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.tools.filesystem import ToolResult
from app.tools.workspace import Workspace

# Read-only inspection plus the test/lint runners an agent needs to verify its
# own work. Deliberately excludes anything that installs, fetches, or mutates
# state outside the workspace.
DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ls", "cat", "head", "tail", "wc", "find", "grep", "rg", "diff",
        "python", "python3", "pytest", "ruff", "mypy",
        "node", "npm", "npx", "tsc",
        "git",
    }
)

MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT_S = 60


class RunCommandInput(BaseModel):
    command: str = Field(
        description=(
            "A single command to run in the workspace root. No shell syntax: "
            "no pipes, redirects, chaining or substitution."
        )
    )
    timeout_s: int = Field(default=DEFAULT_TIMEOUT_S, gt=0, le=600)


@dataclass
class CommandOutcome:
    stdout: str
    stderr: str
    return_code: int


class ShellTool:
    def __init__(
        self,
        workspace: Workspace,
        allowlist: frozenset[str] = DEFAULT_ALLOWLIST,
    ) -> None:
        self.ws = workspace
        self.allowlist = allowlist

    def run_command(self, args: RunCommandInput) -> ToolResult:
        command = args.command.strip()
        if not command:
            return ToolResult("command is empty", is_error=True, misuse=True)

        try:
            argv, operator = _tokenize(command)
        except ValueError as exc:
            return ToolResult(
                f"Could not parse command: {exc}", is_error=True, misuse=True
            )

        if operator is not None:
            return ToolResult(
                f"Shell syntax is not available: {operator!r} was used outside "
                f"quotes. Run one program at a time; if you need to combine "
                f"steps, run them as separate calls.",
                is_error=True,
                misuse=True,
            )

        if not argv:
            return ToolResult("command is empty", is_error=True, misuse=True)

        executable = argv[0]
        if executable not in self.allowlist:
            return ToolResult(
                f"{executable!r} is not an allowed command. Allowed: "
                f"{', '.join(sorted(self.allowlist))}",
                is_error=True,
                misuse=True,
            )

        try:
            completed = subprocess.run(  # noqa: S603 - argv is allowlisted, shell=False
                argv,
                cwd=self.ws.root,
                capture_output=True,
                text=True,
                timeout=args.timeout_s,
                shell=False,
                check=False,
            )
        except FileNotFoundError:
            return ToolResult(f"{executable!r} is not installed", is_error=True)
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"Command timed out after {args.timeout_s}s: {command}", is_error=True
            )

        return ToolResult(
            _render(
                CommandOutcome(
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    return_code=completed.returncode,
                )
            ),
            # A non-zero exit is a real result the agent must see and reason
            # about (a failing test suite), not a tool malfunction.
            is_error=False,
        )


def _tokenize(command: str) -> tuple[list[str], str | None]:
    """Split a command, reporting any shell operator used outside quotes.

    `punctuation_chars=True` makes shlex emit `;`, `|`, `&&`, `(`, `>` and
    friends as standalone tokens when they are unquoted, and leave them inside
    words when they are quoted — which is exactly the distinction we need.

    Newlines are handled separately: shlex treats an unquoted newline as plain
    whitespace, so `ls\\nrm -rf /` would silently become `ls rm -rf /`. We
    compare the newline count in the raw string against the count surviving
    inside tokens; a surplus means at least one acted as a separator.

    Returns (argv, offending_operator_or_None).
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)

    punctuation = set(lexer.punctuation_chars)
    for token in tokens:
        if token and all(char in punctuation for char in token):
            return tokens, token

    if command.count("\n") > sum(token.count("\n") for token in tokens):
        return tokens, "\n"

    return tokens, None


def _render(outcome: CommandOutcome) -> str:
    parts = [f"exit code: {outcome.return_code}"]
    if outcome.stdout:
        parts.append(f"stdout:\n{_cap(outcome.stdout)}")
    if outcome.stderr:
        parts.append(f"stderr:\n{_cap(outcome.stderr)}")
    if not outcome.stdout and not outcome.stderr:
        parts.append("(no output)")
    return "\n\n".join(parts)


def _cap(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[truncated at {MAX_OUTPUT_CHARS} chars]"
