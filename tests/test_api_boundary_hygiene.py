from __future__ import annotations

import re
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src" / "astrodyn_core"

# Compatibility façades intentionally keep legacy aliases.
ALLOWED_FACADE_FILES = {
    "mission/maneuvers.py",
    "uncertainty/propagator.py",
    "states/orekit.py",
    "propagation/config.py",
}

_PRIVATE_IMPORT_RE = re.compile(r"from\s+astrodyn_core\.[\w\.]+\s+import\s+(_\w+)")


def test_no_private_cross_module_imports_outside_facades() -> None:
    violations: list[str] = []

    for path in SRC_ROOT.rglob("*.py"):
        rel = path.relative_to(SRC_ROOT).as_posix()
        if rel in ALLOWED_FACADE_FILES:
            continue

        text = path.read_text(encoding="utf-8")
        for match in _PRIVATE_IMPORT_RE.finditer(text):
            symbol = match.group(1)
            violations.append(f"{rel}: imports private symbol {symbol}")

    assert not violations, "\n".join(violations)
