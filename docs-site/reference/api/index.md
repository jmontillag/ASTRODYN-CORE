# API Reference Overview

This API reference is generated from Python docstrings via `mkdocstrings`.

## Coverage plan

Priority order for documentation coverage:

1. Root exports (`astrodyn_core`)
2. `propagation` public APIs (facade/specs/interfaces)
3. `orekit_env` shared environment helpers
4. Other user-facing subsystems (`states`, `mission`, `uncertainty`, `ephemeris`, `tle`)

## Build prerequisites

Make sure the package and docs dependencies are installed in the project env:

```bash
conda run -n astrodyn-core-env python -m pip install -e .[docs]
```
