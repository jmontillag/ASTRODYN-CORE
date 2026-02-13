#!/usr/bin/env python3
"""
Create/update the conda environment and install ASTRODYN-CORE in editable mode.

Usage:
    python setup_env.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / "environment.yml"


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _package_manager() -> str:
    if shutil.which("mamba"):
        return "mamba"
    if shutil.which("conda"):
        return "conda"
    raise RuntimeError("Neither 'mamba' nor 'conda' is available in PATH.")


def _get_env_name() -> str:
    if not ENV_FILE.exists():
        raise FileNotFoundError(f"Missing {ENV_FILE}")

    text = ENV_FILE.read_text(encoding="utf-8")
    match = re.search(r"^name:\s*(\S+)", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"No environment name found in {ENV_FILE}")

    return match.group(1)


def _env_exists(env_name: str) -> bool:
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    env_paths = data.get("envs", [])
    return any(Path(path).name == env_name for path in env_paths)


def main() -> int:
    env_name = _get_env_name()
    manager = _package_manager()

    _run(["conda", "--version"])

    if _env_exists(env_name):
        _run([manager, "env", "update", "-f", str(ENV_FILE), "-n", env_name])
    else:
        _run([manager, "env", "create", "-f", str(ENV_FILE)])

    _run(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "-m",
            "pip",
            "install",
            "-e",
            ".[dev]",
        ]
    )

    print(f"\nDone. Activate with: conda activate {env_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
