"""Bundled data assets for astrodyn-core.

Provides helpers to locate bundled preset configuration files.
"""

from __future__ import annotations

import importlib.resources as _res
from pathlib import Path


def get_propagation_model(name: str) -> Path:
    """Return the path to a bundled propagation-model YAML preset.

    Parameters
    ----------
    name:
        Preset name, with or without the ``.yaml`` extension.
        Examples: ``"high_fidelity"``, ``"j2_model.yaml"``.

    Returns
    -------
    Path
        Absolute path to the YAML file.

    Raises
    ------
    FileNotFoundError
        If no preset with that name exists.
    """
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    ref = _res.files("astrodyn_core.data.propagation_models").joinpath(name)

    # importlib.resources may return a Traversable; resolve to a real path
    with _res.as_file(ref) as path:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Propagation model preset '{name}' not found. "
                f"Available presets: {list_propagation_models()}"
            )
        return Path(resolved)


def list_propagation_models() -> list[str]:
    """Return names of all bundled propagation-model presets."""
    pkg = _res.files("astrodyn_core.data.propagation_models")
    return sorted(p.name for p in pkg.iterdir() if hasattr(p, "name") and p.name.endswith(".yaml"))
