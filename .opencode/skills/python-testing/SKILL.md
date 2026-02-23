---
name: python-testing
description: How to run Python code, tests, and scripts in this project. Covers the required conda environment, Orekit JVM initialisation, and common pytest patterns.
metadata:
  audience: developers
  workflow: testing
---

## Conda Environment

This project uses a conda environment defined in `environment.yml` at the project root.

- **Environment name**: `astrodyn-core-env`
- **Python version**: 3.11
- **Key dependencies**: `orekit=13.1`, `numpy`, `scipy`, `pybind11`, `scikit-build-core`

### Running Python commands

**NEVER** use the base/default Python interpreter. Always prefix commands with `conda run -n astrodyn-core-env`:

```bash
# Run tests
conda run -n astrodyn-core-env python -m pytest tests/ -v

# Run a single test file
conda run -n astrodyn-core-env python -m pytest tests/test_geqoe_refactor.py -v

# Run a script
conda run -n astrodyn-core-env python examples/some_script.py

# Run Python interactively
conda run -n astrodyn-core-env python -c "import astrodyn_core; print('ok')"

# Install the package in editable mode (already done, but if needed)
conda run -n astrodyn-core-env pip install -e ".[dev]"
```

### Why this matters

The base Python environment does **not** have `orekit` installed (it is a conda-forge package requiring a JVM). Running tests with the wrong Python will cause:
- `ModuleNotFoundError: No module named 'orekit'`
- `ModuleNotFoundError: No module named 'org'` (Orekit Java classes via JPype)
- Orekit-dependent tests being incorrectly skipped instead of actually running

## Orekit JVM Initialisation

Orekit is a Java library accessed via JPype. Before any `org.orekit.*` or `org.hipparchus.*` imports, the JVM must be started:

```python
import orekit
orekit.initVM()
```

Additionally, Orekit needs reference data (leap seconds, Earth orientation, etc.). This project has `orekit-data.zip` symlinked in the project root. Load it with:

```python
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()
```

### Pattern for Orekit-dependent tests

Follow this established pattern used throughout the test suite:

```python
import pytest

# Skip the entire module if orekit is not importable
orekit = pytest.importorskip("orekit")
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402
setup_orekit_curdir()

# Now Orekit Java classes are available
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
# ... etc
```

Key points:
- `pytest.importorskip("orekit")` skips the module gracefully if orekit is not installed
- `orekit.initVM()` must be called at module level, before any `org.*` imports
- `setup_orekit_curdir()` loads `orekit-data.zip` from the current working directory
- These three lines go at module level (not inside fixtures or test functions)
- Tests that do NOT need Orekit should NOT import or initialise it

### Pattern for checking Orekit availability at runtime (non-test code)

In production code, use try/except for lazy Orekit detection:

```python
def _orekit_available() -> bool:
    try:
        import orekit
        orekit.initVM()
        from org.orekit.propagation import AbstractPropagator
        return True
    except Exception:
        return False
```

## Project Structure

- **Source code**: `src/astrodyn_core/`
- **Tests**: `tests/`
- **Examples**: `examples/`
- **Orekit data**: `orekit-data.zip` (symlink to shared data)
- **Build system**: `scikit-build-core` + `pybind11` + CMake (for C++ modules)

## pytest Configuration

Defined in `pyproject.toml`:
- `pythonpath = ["src"]` — source is importable without install
- `testpaths = ["tests"]` — default test discovery path

## Common Test Commands

```bash
# Run all tests
conda run -n astrodyn-core-env python -m pytest tests/ -v

# Run tests matching a pattern
conda run -n astrodyn-core-env python -m pytest tests/ -v -k "geqoe"

# Run with stop-on-first-failure
conda run -n astrodyn-core-env python -m pytest tests/ -v -x

# Run a specific test class
conda run -n astrodyn-core-env python -m pytest tests/test_geqoe_provider.py::TestGEqOEProviderRegistration -v
```
