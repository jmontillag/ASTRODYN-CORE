"""API boundary and import hygiene tests.

These tests enforce API governance policy:
- No private underscore cross-module imports in source modules.
- Root __all__ is consistent with actual module contents.
- Examples do not import private symbols or use internal module paths.
"""

from __future__ import annotations

import re
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src" / "astrodyn_core"
EXAMPLES_ROOT = WORKSPACE_ROOT / "examples"
TESTS_ROOT = WORKSPACE_ROOT / "tests"

_PRIVATE_IMPORT_RE = re.compile(r"from\s+astrodyn_core\.[\w\.]+\s+import\s+(_\w+)")
_SHIM_IMPORT_RE = re.compile(
    r"(?:from\s+astrodyn_core\.propagation\.(?:universe|assembly|dsst_assembly)\s+import|"
    r"import\s+astrodyn_core\.propagation\.(?:universe|assembly|dsst_assembly)\b)"
)


def test_no_private_cross_module_imports() -> None:
    """Source modules must not import private underscore symbols from siblings."""
    violations: list[str] = []

    for path in SRC_ROOT.rglob("*.py"):
        rel = path.relative_to(SRC_ROOT).as_posix()
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


def test_internal_source_modules_do_not_import_propagation_shims() -> None:
    """Internal source should use canonical module paths, not removed shim paths."""
    violations: list[str] = []

    for path in SRC_ROOT.rglob("*.py"):
        rel = path.relative_to(SRC_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for match in _SHIM_IMPORT_RE.finditer(text):
            violations.append(f"{rel}: {match.group(0).strip()}")

    assert not violations, (
        "Internal source imports removed shim modules. Use canonical paths "
        "instead:\n" + "\n".join(violations)
    )


def test_propagation_shim_modules_removed() -> None:
    """Final architecture should not ship compatibility shim module files."""
    removed = [
        SRC_ROOT / "propagation" / "universe.py",
        SRC_ROOT / "propagation" / "assembly.py",
        SRC_ROOT / "propagation" / "dsst_assembly.py",
    ]
    still_present = [p.relative_to(WORKSPACE_ROOT).as_posix() for p in removed if p.exists()]
    assert not still_present, "Removed shim modules still exist:\n" + "\n".join(still_present)


def test_tests_do_not_import_removed_propagation_shims() -> None:
    """Tests should target canonical modules or public APIs after shim removal."""
    violations: list[str] = []
    for path in TESTS_ROOT.rglob("*.py"):
        rel = path.relative_to(WORKSPACE_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for match in _SHIM_IMPORT_RE.finditer(text):
            violations.append(f"{rel}: {match.group(0).strip()}")

    assert not violations, (
        "Tests import removed shim modules. Update to canonical paths:\n"
        + "\n".join(violations)
    )


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
    uncertainty.matrix_io, uncertainty.records, etc.
    """
    _INTERNAL_MODULE_PATTERN = re.compile(
        r"from\s+astrodyn_core\."
        r"(?:"
        r"states\.orekit_\w+|"
        r"states\.validation|"
        r"propagation\.parsers\.\w+|"
        r"uncertainty\.matrix_io|"
        r"uncertainty\.records|"
        r"uncertainty\.stm|"
        r"uncertainty\.factory|"
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
