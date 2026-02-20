"""API boundary and import hygiene tests.

These tests enforce the Phase C API governance policy:
- No private underscore cross-module imports outside designated compatibility facades.
- Root __all__ is consistent with actual module contents.
- Examples do not import private symbols or use internal module paths.
"""

from __future__ import annotations

import re
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src" / "astrodyn_core"
EXAMPLES_ROOT = WORKSPACE_ROOT / "examples"

# Compatibility facades intentionally keep legacy aliases via __getattr__.
ALLOWED_FACADE_FILES = {
    "mission/maneuvers.py",
    "uncertainty/propagator.py",
    "states/orekit.py",
    "propagation/config.py",
}

_PRIVATE_IMPORT_RE = re.compile(r"from\s+astrodyn_core\.[\w\.]+\s+import\s+(_\w+)")


def test_no_private_cross_module_imports_outside_facades() -> None:
    """Non-facade source modules must not import private underscore symbols from siblings."""
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


def test_root_all_consistency() -> None:
    """Every name in root __all__ must be importable from the package."""
    import astrodyn_core

    missing: list[str] = []
    for name in astrodyn_core.__all__:
        if not hasattr(astrodyn_core, name):
            missing.append(name)

    assert not missing, f"Root __all__ lists names not in module: {missing}"


def test_examples_do_not_import_private_symbols() -> None:
    """Example scripts must not import underscore-prefixed symbols from astrodyn_core."""
    if not EXAMPLES_ROOT.exists():
        return

    violations: list[str] = []
    for path in EXAMPLES_ROOT.rglob("*.py"):
        rel = path.relative_to(WORKSPACE_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for match in _PRIVATE_IMPORT_RE.finditer(text):
            symbol = match.group(1)
            violations.append(f"{rel}: imports private symbol {symbol}")

    assert not violations, "\n".join(violations)


def test_examples_prefer_public_subpackage_paths() -> None:
    """Examples should import from public subpackage paths, not from internal modules.

    Internal modules are: states.validation, states.orekit_*, propagation.parsers.*,
    propagation.universe, uncertainty.matrix_io, uncertainty.records, etc.
    """
    _INTERNAL_MODULE_PATTERN = re.compile(
        r"from\s+astrodyn_core\."
        r"(?:"
        r"states\.orekit_\w+|"
        r"states\.validation|"
        r"propagation\.parsers\.\w+|"
        r"propagation\.universe|"
        r"uncertainty\.matrix_io|"
        r"uncertainty\.records|"
        r"uncertainty\.stm|"
        r"mission\.models|"
        r"mission\.timeline|"
        r"mission\.intents|"
        r"mission\.kinematics|"
        r"mission\.simulation"
        r")\s+import"
    )

    if not EXAMPLES_ROOT.exists():
        return

    violations: list[str] = []
    for path in EXAMPLES_ROOT.rglob("*.py"):
        rel = path.relative_to(WORKSPACE_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for match in _INTERNAL_MODULE_PATTERN.finditer(text):
            violations.append(f"{rel}: {match.group(0).strip()}")

    assert not violations, (
        "Examples import from internal modules. Use facade clients or "
        "public subpackage APIs instead:\n" + "\n".join(violations)
    )


def test_facade_modules_use_getattr_for_deprecated_aliases() -> None:
    """Compatibility facade modules must use __getattr__ for private aliases, not bare assignment."""
    bare_alias_re = re.compile(r"^(_\w+)\s*=\s*\w+", re.MULTILINE)

    violations: list[str] = []
    for rel in ALLOWED_FACADE_FILES:
        path = SRC_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in bare_alias_re.finditer(text):
            alias = match.group(1)
            violations.append(f"{rel}: bare private alias {alias} (should use __getattr__)")

    assert not violations, "\n".join(violations)
