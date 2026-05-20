"""Path helpers for commands that are run from the repository root."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root containing the legacy paper scripts."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path
