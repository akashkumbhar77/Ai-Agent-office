"""Process isolation for agent-run commands.

The allowlist in `shell.py` was never a security boundary and could not be
one: it contained interpreters, and `python -c` runs anything. This module is
the boundary. With it, the allowlist goes back to being what it should always
have been — a guardrail that keeps agents on the tools they need.

The runtime is bubblewrap (`bwrap`), not Docker. Three reasons: it is
unprivileged, so the backend does not need daemon access or a group that is
equivalent to root; it starts in milliseconds, which matters when an agent
runs a test suite thirty times in a run; and it is already present on most
Linux systems that ship Flatpak. Docker would also have meant building and
distributing an image, which is a deployment story this project does not have.

What a sandboxed command can see:

- the workspace, read-write, bound at the same absolute path it has outside,
  so paths in compiler and test output are the ones the operator sees
- `/usr` and the standard symlinks into it, read-only, so interpreters and
  their standard libraries work
- a private `/tmp`, `/proc` and `/dev` that are discarded on exit

What it cannot: the rest of the filesystem, the network, other processes, or
the host's environment variables.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

BWRAP = "bwrap"

# Read-only mounts the sandbox needs for anything to execute at all. Missing
# entries are skipped rather than fatal: distributions disagree about which of
# these exist, and bwrap fails the whole run on a bind to a missing source.
_RO_BINDS: tuple[str, ...] = ("/usr", "/etc/alternatives", "/opt")

# Merged-/usr symlinks. Created inside the sandbox rather than bound, because
# on a merged-/usr host they *are* symlinks and binding them would need the
# target anyway.
_SYMLINKS: tuple[tuple[str, str], ...] = (
    ("usr/lib", "/lib"),
    ("usr/lib64", "/lib64"),
    ("usr/bin", "/bin"),
    ("usr/sbin", "/sbin"),
)


class SandboxUnavailable(Exception):
    """No isolation runtime on this host."""


def _toolchain_prefixes(workspace_root: Path) -> list[Path]:
    """Directories a workspace virtualenv needs in order to run at all.

    Found live. A venv inside the workspace is bound like the rest of it, but
    its `bin/python3` is a *symlink to the interpreter it was built from* —
    anaconda, pyenv, uv, homebrew — which usually lives outside `/usr`. With
    that target missing, every script in the venv fails `execvp` with ENOENT
    even though the file is plainly there, because the kernel cannot find the
    interpreter named in its shebang.

    So bind the base prefix, read-only. This widens what an agent can *read*
    to the toolchain the project itself declares it was built with — a real
    but bounded expansion, and the alternative is a venv that cannot run.
    Nothing here becomes writable, and the network stays closed.
    """
    venv = workspace_root / ".venv"
    candidates: list[Path] = []

    # pyvenv.cfg records the base installation as `home = <prefix>/bin`.
    config = venv / "pyvenv.cfg"
    if config.is_file():
        try:
            for line in config.read_text().splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "home" and value.strip():
                    home = Path(value.strip())
                    candidates.append(home.parent if home.name == "bin" else home)
        except OSError:
            log.warning("pyvenv_cfg_unreadable", path=str(config))

    # Belt and braces: follow the symlink itself, in case the config is
    # missing or disagrees with reality.
    interpreter = venv / "bin" / "python3"
    if interpreter.is_symlink() or interpreter.is_file():
        try:
            real = interpreter.resolve()
            if real.parent.name == "bin":
                candidates.append(real.parent.parent)
        except OSError:
            pass

    prefixes: list[Path] = []
    for candidate in candidates:
        # Already covered, or so broad that binding it would defeat the point:
        # a home directory contains the very things the sandbox exists to keep
        # out of reach.
        if candidate in prefixes or not candidate.is_dir():
            continue
        if candidate == Path("/") or candidate in candidate.parents:
            continue
        if any(str(candidate).startswith(ro) for ro in _RO_BINDS):
            continue
        if candidate in (Path.home(), Path("/home"), Path("/usr"), Path("/etc")):
            log.warning("toolchain_prefix_too_broad", prefix=str(candidate))
            continue
        prefixes.append(candidate)
    return prefixes


@dataclass(frozen=True)
class Sandbox:
    """Wraps an argv so it runs isolated.

    Holds no state beyond the workspace root: one instance is reused for every
    command, and each invocation gets a fresh namespace that dies with it.
    """

    workspace_root: Path

    def wrap(self, argv: list[str]) -> list[str]:
        """Return the argv to actually execute."""
        root = str(self.workspace_root)
        command = [
            BWRAP,
            # The sandbox must not outlive the backend, even if the backend is
            # killed rather than shut down.
            "--die-with-parent",
            # Every namespace: network included. This is what makes `npx` and
            # `git push` safe to allow at all.
            "--unshare-all",
            # Detach the controlling terminal so a command cannot inject into
            # the parent's tty with TIOCSTI.
            "--new-session",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
        ]

        read_only = [Path(p) for p in _RO_BINDS] + _toolchain_prefixes(
            self.workspace_root
        )
        for source in read_only:
            if source.exists():
                command += ["--ro-bind", str(source), str(source)]

        for target, link in _SYMLINKS:
            if not Path(link).is_symlink() and Path(link).exists():
                # A split-/usr host has real directories here; bind them.
                command += ["--ro-bind", link, link]
            else:
                command += ["--symlink", target, link]

        command += [
            # Bound at its real path, so a traceback or a failing test names a
            # path the operator can actually open.
            "--bind", root, root,
            "--chdir", root,
            # A blank environment except what a toolchain genuinely needs.
            # Inheriting the parent's environment would hand the agent every
            # secret the backend was started with, including the API key.
            "--clearenv",
            "--setenv", "HOME", root,
            # A project's own tooling comes first, exactly as it would for a
            # developer with the venv activated. Found live: before this, an
            # agent asked to "run the tests" got `execvp pytest: No such file
            # or directory`, because pytest lives in the *backend's* venv —
            # which is outside the workspace and rightly not mounted. These
            # two paths are inside the workspace, so they are already bound;
            # only PATH was missing.
            "--setenv", "PATH", (
                f"{root}/.venv/bin:{root}/node_modules/.bin"
                ":/usr/bin:/bin:/usr/sbin:/sbin"
            ),
            "--setenv", "LANG", os.environ.get("LANG", "C.UTF-8"),
            # Keeps interpreters from writing caches into a read-only /usr.
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--",
        ]
        return command + argv


def available() -> bool:
    return shutil.which(BWRAP) is not None


def build(mode: str, workspace_root: Path) -> Sandbox | None:
    """Resolve the configured sandbox mode to an instance, or None.

    `None` means commands run unisolated, and every caller of this function is
    responsible for making that visible rather than letting it pass quietly —
    a security posture that degrades silently is the one nobody notices has
    degraded (PLAN.md §6 Phase 5.1).
    """
    if mode == "off":
        log.warning("sandbox_disabled", reason="SANDBOX=off")
        return None

    if available():
        return Sandbox(workspace_root=workspace_root)

    if mode == "bwrap":
        # Explicitly asked for and not present: that is a configuration error,
        # not a reason to silently run commands on the bare host.
        raise SandboxUnavailable(
            "SANDBOX=bwrap but bubblewrap is not installed. Install it "
            "(Debian/Ubuntu: `apt install bubblewrap`), or set SANDBOX=off to "
            "accept running agent commands unisolated."
        )

    log.warning("sandbox_unavailable", mode=mode, reason="bwrap not installed")
    return None
