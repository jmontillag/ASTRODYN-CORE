# AGENTS.md

## Project Execution Environment (Required)

Use the Conda environment `astrodyn-core-env` for all project commands that run
Python code, build extensions, or execute tests.

Do not use the system Python for this repository unless the user explicitly asks.

## Default command policy

Prefer these forms:

- `conda run -n astrodyn-core-env python ...`
- `conda run -n astrodyn-core-env pytest ...`
- `conda run -n astrodyn-core-env pip ...` (or `python -m pip ...`)

Examples:

- Install editable dev environment:
  - `conda run -n astrodyn-core-env python -m pip install -e .[dev]`
- Run full tests:
  - `conda run -n astrodyn-core-env pytest -q`
- Run targeted tests:
  - `conda run -n astrodyn-core-env pytest -q tests/test_api_boundary_hygiene.py`
- Run examples/scripts:
  - `conda run -n astrodyn-core-env python examples/<script>.py`

## Orekit-related tests

Orekit-backed tests should also be run in `astrodyn-core-env` so the correct
Python packages and JVM bridge dependencies are available.

If a test still skips due to missing Orekit data or package installation, report
the skip/error clearly instead of silently switching to another environment.

## If the env is missing or broken

If `conda run -n astrodyn-core-env ...` fails because the environment does not
exist or is broken, stop and report the failure. Ask before creating or
modifying environments.
