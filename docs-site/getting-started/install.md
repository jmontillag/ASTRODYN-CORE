# Install

## Prerequisites

- Python 3.11+
- `pip` (for package users)
- Conda (recommended for contributors / full repo workflows)
- A working C/C++ toolchain if you plan to build native extensions locally

## Choose your install path

This docs site supports two user modes:

- **Package user (pip)**: you want to use the library from your own project
- **Repo user / contributor**: you want examples, tests, docs build, and local development tools

## Option A: Package user (pip)

Use this if you want to install and use the library without cloning the repo.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install astrodyn-core
```

Optional extras:

```bash
# TLE download/Space-Track workflows
python -m pip install "astrodyn-core[tle]"
```

Notes:

- You will still need Orekit data for Orekit-backed workflows (see [Orekit data setup](orekit-data.md))
- The tutorials/how-to pages increasingly include self-contained code snippets so package users can follow along without the repo `examples/` folder

## Option B: Repo user / contributor (Conda + editable install)

From the repository root:

```bash
python setup_env.py
conda run -n astrodyn-core-env python -m pip install -e .[dev,docs]
```

Why this is recommended for repo work:

- Uses the project-standard environment (`astrodyn-core-env`)
- Installs dev tools (tests/lint) and docs tooling (MkDocs)
- Keeps runtime and docs builds consistent

## Verify the installation

### Package user (quick import check)

```bash
python - <<'PY'
import astrodyn_core
print("astrodyn_core import OK")
PY
```

### Repo user / contributor (test check)

```bash
conda run -n astrodyn-core-env pytest -q -rs
```

Expected result:

- Tests pass
- Some tests may skip if optional dependencies/data are missing (for example Orekit data)

## Optional developer shortcuts (repo users)

The repo includes `make` wrappers:

```bash
make help
make test
make docs-build
```
