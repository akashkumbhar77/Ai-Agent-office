"""Shell tool.

Commands are untrusted model output. Four defences, all required
(CLAUDE.md §8):

1. A **sandbox**. This is the actual security boundary — see `sandbox.py`.
   Everything below reduces blast radius or improves the error the agent
   sees; only this contains a command that is genuinely trying to escape.
2. **Argument confinement**. Every token that looks like a path goes through
   the same `Workspace.resolve()` chokepoint the file tools use. Without it,
   an allowlisted `cat` reads any file on the host — which is exactly what it
   did until Phase 5.1, because only `argv[0]` was ever checked.
3. An **allowlist** of executables. A blocklist eventually misses a case; an
   allowlist fails closed on anything unanticipated. Note what this is *not*:
   with interpreters on the list it was never a boundary, since `python -c`
   runs anything. It keeps agents on the tools they need. The sandbox is what
   makes it safe to have interpreters on it at all, which is why the list
   shrinks to `INERT_ALLOWLIST` when no sandbox is available.
4. Rejection of *unquoted* shell operators. The command runs with shell=False,
   so `&&`, `|`, `;` and `$()` are already inert — they would be passed as
   literal arguments. The check exists because their presence means the agent
   expected a shell, and running a mangled argument list silently is worse
   than saying so. Crucially the check is applied to shlex tokens, not to the
   raw string: `grep "foo|bar"` and `python3 -c 'a; b'` are legitimate and
   must not be rejected for characters inside quotes.

Plus a timeout and an output cap, so one command cannot hang the graph or
fill the context window.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.tools.filesystem import ToolResult
from app.tools.sandbox import Sandbox
from app.tools.workspace import Workspace, WorkspaceViolation

# Commands that read and report, and nothing else. Safe to run unsandboxed
# once their arguments are confined, because the program itself cannot be
# talked into doing something other than what its name says.
INERT_ALLOWLIST: frozenset[str] = frozenset(
    {"ls", "cat", "head", "tail", "wc", "find", "grep", "rg", "diff"}
)

# The tools an agent needs to verify its own work. Every one of these is
# general-purpose execution: `python -c` runs arbitrary code, `npx` fetches
# and runs a package, `git push` reaches the network. They are permitted only
# when a sandbox is containing them.
EXECUTING_ALLOWLIST: frozenset[str] = frozenset(
    {
        "python", "python3", "pytest", "ruff", "mypy",
        "node", "npm", "npx", "tsc",
        "git",
    }
)

DEFAULT_ALLOWLIST: frozenset[str] = INERT_ALLOWLIST | EXECUTING_ALLOWLIST

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
        allowlist: frozenset[str] | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        self.ws = workspace
        self.sandbox = sandbox
        # Default the allowlist to what the isolation actually supports.
        # Passing one explicitly is for tests; production goes through this
        # branch so the two can never drift apart in a deployment.
        if allowlist is None:
            allowlist = DEFAULT_ALLOWLIST if sandbox else INERT_ALLOWLIST
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
            hint = ""
            if executable in EXECUTING_ALLOWLIST:
                # Distinguish "never allowed" from "not allowed right now".
                # Without this the agent reads a flat refusal for a tool it
                # has seen work, and burns turns retrying it.
                hint = (
                    f" {executable!r} needs the sandbox, which is not running "
                    f"on this host — see the office banner."
                )
            return ToolResult(
                f"{executable!r} is not an allowed command. Allowed: "
                f"{', '.join(sorted(self.allowlist))}.{hint}",
                is_error=True,
                misuse=True,
            )

        violation = self._confine(argv[1:])
        if violation is not None:
            return violation

        try:
            completed = subprocess.run(  # noqa: S603 - allowlisted, confined, shell=False
                self.sandbox.wrap(argv) if self.sandbox else argv,
                cwd=self.ws.root,
                capture_output=True,
                text=True,
                timeout=args.timeout_s,
                shell=False,
                check=False,
            )
        except FileNotFoundError:
            return ToolResult(_not_installed(executable, bool(self.sandbox)), is_error=True)
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"Command timed out after {args.timeout_s}s: {command}", is_error=True
            )

        # bwrap reports a missing executable itself, on stderr, with its own
        # name in front of it. Left alone the agent reads
        # `bwrap: execvp pytest: No such file or directory` — which leaks an
        # implementation detail and suggests nothing it could do instead.
        missing = _missing_executable(completed.stderr)
        if missing is not None:
            return ToolResult(
                _not_installed(missing, bool(self.sandbox)), is_error=True
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


    def _confine(self, arguments: list[str]) -> ToolResult | None:
        """Reject any argument that names something outside the workspace.

        Returns None when every argument is acceptable.

        The hard part is deciding which tokens are paths at all, and the
        answer here is to not decide: anything that *could* be read as a path
        escaping the workspace is rejected, whether or not it was meant as
        one. A guess that lets an escape through is a hole; a guess that
        rejects `--format=a/b` costs the agent one corrected retry, which it
        is built to handle. Fail closed on the ambiguity.

        This is deliberately not a `..` blocklist. Tokens go through the same
        `Workspace.resolve()` the file tools use, so symlinks, traversal and
        their combinations are all covered by one containment check — the
        reasoning in workspace.py applies unchanged.
        """
        for argument in arguments:
            for candidate in _path_candidates(argument):
                if candidate.startswith("/"):
                    return ToolResult(
                        f"{candidate!r} is an absolute path. Commands run "
                        f"inside the workspace and may only name paths "
                        f"relative to it.",
                        is_error=True,
                        misuse=True,
                    )
                try:
                    self.ws.resolve(candidate)
                except WorkspaceViolation as exc:
                    return ToolResult(
                        f"{candidate!r} is outside the workspace: {exc}",
                        is_error=True,
                        misuse=True,
                    )
        return None


def _missing_executable(stderr: str) -> str | None:
    """Pull the program name out of bwrap's own not-found message."""
    marker = "bwrap: execvp "
    for line in stderr.splitlines():
        if line.startswith(marker) and ":" in line[len(marker) :]:
            return line[len(marker) :].split(":", 1)[0].strip()
    return None


def _not_installed(executable: str, sandboxed: bool) -> str:
    """An actionable refusal.

    The agent's next move should be obvious from the message. "Not installed"
    on its own sends it round the loop trying the same thing.
    """
    message = f"{executable!r} is not available in this workspace."
    if sandboxed:
        message += (
            " Commands run isolated, so only the system tools and whatever the"
            " project itself provides in .venv/bin or node_modules/.bin are on"
            " PATH. Try `python3 -m <module>` instead, or check what the"
            " project actually ships before reaching for a tool."
        )
    return message


def _path_candidates(argument: str) -> list[str]:
    """The parts of one argument that have to survive containment.

    `--include=src/*.py` carries its path after the `=`, and a bare `src/x`
    is one. A token with no separator (`-l`, `HEAD`, `--strict`) names no
    path and is left alone — checking it would reject ordinary flags.
    """
    if not argument:
        return []

    body = argument
    if argument.startswith("-") and "=" in argument:
        body = argument.split("=", 1)[1]
    elif argument.startswith("-"):
        # A bare flag. `-rf` is not a path, and treating it as one would
        # reject most real commands.
        return []

    if not body:
        return []
    # `/` catches relative and absolute paths alike; a leading `..` catches
    # traversal that never uses a separator on its own (`..`).
    if "/" in body or body == ".." or body.startswith("../"):
        return [body]
    return []


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
