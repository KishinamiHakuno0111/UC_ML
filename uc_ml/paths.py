"""Path helpers for commands that are run from the repository root."""

from __future__ import annotations

from pathlib import Path


WORKFLOW_DIRECTORIES = (
    "case_data",
    "case_data/matpower",
    "training_data",
    "module_save",
    "fig_data",
    "figures",
)


def project_root() -> Path:
    """Return the repository root containing the legacy paper scripts."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_project_directories(root: str | Path | None = None) -> list[tuple[str, Path, bool]]:
    """Create standard workflow directories and report whether each existed."""
    base = Path(root) if root is not None else project_root()
    created: list[tuple[str, Path, bool]] = []
    for dirname in WORKFLOW_DIRECTORIES:
        path = base / dirname
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        created.append((dirname, path, existed))
    return created
