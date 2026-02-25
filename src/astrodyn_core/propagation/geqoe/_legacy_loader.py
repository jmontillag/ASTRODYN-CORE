"""Helpers to import legacy GEqOE modules from the repository root."""

import sys
from importlib import import_module
from pathlib import Path


def import_legacy_module(module_name: str):
    """Import a legacy module by temporarily prepending the repo root to ``sys.path``.

    Args:
        module_name: Fully qualified legacy module name.

    Returns:
        Imported Python module object.
    """
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return import_module(module_name)
