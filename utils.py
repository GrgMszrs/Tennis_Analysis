from pathlib import Path
import os

def _ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def _display_path(path: Path) -> str:
    """Return *path* relative to the CWD if possible; else its absolute form.

    Avoids `ValueError` when the two directories live on different mounts or
    virtual sandboxes (e.g., "/home/pyodide").
    """
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path) 