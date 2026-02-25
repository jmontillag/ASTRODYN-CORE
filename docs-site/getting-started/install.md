# Install

## Prerequisites

- Python 3.11+
- Conda (recommended for this project)
- A working C/C++ toolchain if you plan to build native extensions locally

## Recommended setup

From the repository root:

```bash
python setup_env.py
conda run -n astrodyn-core-env python -m pip install -e .[dev,docs]
```

Why this is recommended:

- Uses the project-standard environment (`astrodyn-core-env`)
- Installs dev tools (tests/lint) and docs tooling (MkDocs)
- Keeps runtime and docs builds consistent

## Verify the environment

```bash
conda run -n astrodyn-core-env pytest -q -rs
```

Expected result:

- Tests pass
- Some tests may skip if optional dependencies/data are missing (for example Orekit data)

## Optional developer shortcuts

The repo includes `make` wrappers:

```bash
make help
make test
make docs-build
```
