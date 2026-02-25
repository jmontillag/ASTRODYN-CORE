# Common Setup Issues

## Use the correct environment

This project expects commands to run in the Conda env:

- `astrodyn-core-env`

Examples:

```bash
conda run -n astrodyn-core-env pytest -q
conda run -n astrodyn-core-env python examples/quickstart.py --mode all
```

If tests or imports fail unexpectedly, first confirm the active interpreter.

## `orekit` import or JVM issues

Common causes:

- package installed in a different interpreter
- not running in `astrodyn-core-env`
- Orekit data path assumptions not met

Quick check:

```bash
conda run -n astrodyn-core-env python - <<'PY'
import orekit
orekit.initVM()
print("ok")
PY
```

## Release packaging: `python -m build` fails from repo root

This repo may contain a local `build/` directory, which can shadow the PyPI
`build` tool package.

Symptoms:

- `No module named build.__main__; 'build' is a package and cannot be directly executed`

Workaround:

- Run `python -m build` from outside the repo root
- Use the AGENTS-documented release commands (fresh `/tmp` build dir + `--no-isolation`)

## Wheel build fails but sdist succeeds

If you see a `scikit-build-core` failure while parsing CMake File API replies
(`IndexError` in `reply.py`), the issue may be environment- or sandbox-specific.

Recommended steps:

1. Delete temporary scikit-build directories under `/tmp` and retry with a fresh build dir
2. Use `--no-isolation` in restricted/offline environments
3. Confirm the issue outside sandboxed/agent runtimes if possible
4. Publish `sdist`-only as a fallback if wheel build remains blocked and you need a release

The project can still be valid and test-clean even if a specific runtime cannot
produce wheels.
