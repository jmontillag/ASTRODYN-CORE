# Orekit Data Setup

Orekit-based workflows require Orekit data files (Earth orientation parameters,
gravity models, time scales, etc.).

## Project-local data

This repository includes an `orekit-data.zip` at the root. Depending on your
workflow, examples/tests may expect Orekit data to be available in the working
directory or configured via Orekit helper utilities.

## Typical workflow

1. Keep `orekit-data.zip` in the repo root (already present in this repo)
2. Run examples/tests from the repo root
3. If an example or test still fails on Orekit data lookup, check the script's
   setup pattern and `orekit.pyhelpers.setup_orekit_curdir()` usage

## Verify Orekit availability

```bash
conda run -n astrodyn-core-env python - <<'PY'
import orekit
orekit.initVM()
print("Orekit JVM initialized")
PY
```

## Common issues

- `ModuleNotFoundError: orekit`: package not installed in the active interpreter
- JVM init issues: ensure you are using `astrodyn-core-env`
- Missing data/model errors: run from repo root and confirm Orekit data path assumptions

See [Common Setup Issues](../troubleshooting/common-setup-issues.md) for a longer checklist.
