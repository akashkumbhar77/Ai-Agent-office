"""Workspace confinement.

Every path an agent supplies is untrusted model output. This module is the
single chokepoint that turns such a path into a real filesystem path, and
every file tool must go through it (CLAUDE.md §8).

The rule is simple and absolute: the resolved path must live inside the
workspace root. `resolve()` follows symlinks, so a symlink pointing outside
the root is caught by the same check that catches `..` — there is no separate
traversal blocklist to keep up to date, because a blocklist is exactly the
thing that eventually misses a case.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class WorkspaceViolation(Exception):
    """A tool tried to touch something outside the workspace."""


class Workspace:
    def __init__(self, root: Path) -> None:
        # strict=True: a root that does not exist is a configuration error, and
        # failing here beats silently creating one somewhere unexpected.
        self.root = root.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceViolation(f"workspace root is not a directory: {self.root}")

    def resolve(self, relative: str) -> Path:
        """Map a model-supplied relative path to an absolute path inside root.

        Raises WorkspaceViolation for anything that escapes, is absolute, or
        is empty. Does not require the file to exist — writes need to resolve
        paths that do not exist yet.
        """
        if not relative or not relative.strip():
            raise WorkspaceViolation("path is empty")

        # Reject absolute input before joining: Path("/a") / "/etc/passwd"
        # yields "/etc/passwd", so joining an absolute path silently discards
        # the root. Check both flavours — a Windows-style path arriving here
        # is malformed input, not a valid relative path.
        pure = PurePosixPath(relative)
        if pure.is_absolute() or relative.startswith("\\") or ":" in relative[:2]:
            raise WorkspaceViolation(f"path must be relative to the workspace: {relative!r}")

        # resolve() collapses `..` and follows symlinks, so one containment
        # check covers traversal, symlink escapes, and their combinations.
        resolved = (self.root / relative).resolve()

        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise WorkspaceViolation(
                f"path escapes the workspace: {relative!r} -> {resolved}"
            )
        return resolved

    def relative(self, path: Path) -> str:
        """Workspace-relative form, for the wire.

        `file.change` events carry relative paths; an absolute path reaching
        the wire means confinement was bypassed (PROTOCOL.md §4.8).
        """
        return str(path.resolve().relative_to(self.root))

    def ensure_parent(self, path: Path) -> None:
        """Create the parent directory of a path already validated by resolve()."""
        path.parent.mkdir(parents=True, exist_ok=True)
