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

## Orekit MCP usage (Preferred for API tasks)

When a task involves Orekit API usage, class/method selection, signatures,
overloads, or uncertain Orekit behavior, use the `orekit_docs` MCP tools before
writing code.

Preferred workflow:

1. `orekit_docs_info` to confirm the active docs/index.
2. `orekit_search_symbols` to find the relevant classes/methods.
3. `orekit_get_class_doc` and/or `orekit_get_member_doc` to retrieve the docs.
4. Then write code, noting Java-to-Python wrapper differences when relevant.

Do not rely only on memory for exact Orekit signatures if the MCP tools are
available in the current session.

If the current Codex runtime does not expose MCP tool calls (or the server
handshake fails), state that limitation clearly and fall back to local knowledge.

## Release packaging (sdist & wheel builds)

### Build backend

This project uses **scikit-build-core** with CMake and pybind11 to compile C++
extensions. The build system is defined in `pyproject.toml` under
`[build-system]`.

### Directory shadowing hazard

The repository contains a `build/` directory at the root. Python's `-m build`
resolves `build` as this local package instead of the `build` PyPI tool, causing:

    No module named build.__main__; 'build' is a package and cannot be directly executed

**Workaround:** Always invoke `python -m build` from outside the repo root, or
use an explicit output directory and build-dir override:

```bash
conda run -n astrodyn-core-env python -m build --no-isolation \
  -Cbuild-dir=/tmp/skbuild-build \
  /home/astror/Projects/ASTRODYN-CORE \
  -o /home/astror/Projects/ASTRODYN-CORE/dist
```

### Build isolation and network access

`python -m build` defaults to creating an isolated virtualenv and downloading
build dependencies from PyPI. In offline or network-restricted environments this
fails silently with repeated pip retries.

**Workaround:** Use `--no-isolation` so the build uses packages already installed
in `astrodyn-core-env`. Make sure the env has the required build tools:

```bash
conda run -n astrodyn-core-env python -m pip install build twine scikit-build-core pybind11
```

### CMake File API and stale build directories

scikit-build-core relies on the CMake File API (`.cmake/api/v1/reply/`) to
discover build targets. If this directory is empty after configure, the wheel
build crashes with:

    IndexError: list index out of range  (in scikit_build_core reply.py)

Known causes of empty File API replies:

- **Stale or corrupted build directories** from prior failed attempts (e.g.,
  network-aborted isolation builds). Always delete the build directory between
  retries: `rm -rf /tmp/skbuild-build`.
- **CMake version churn** — swapping CMake versions (pip cmake vs conda cmake
  vs system cmake) mid-session can leave inconsistent generator state.
- **Residual cmake caches** — if the generator or toolchain changed, the old
  `CMakeCache.txt` may prevent File API files from being written.

**Best practice:** Use a fresh `/tmp` build directory for each wheel attempt
(`-Cbuild-dir=/tmp/skbuild-<purpose>`) and remove it before retrying.

### Verified build commands

Build both sdist and wheel (offline-safe):

```bash
# Clean previous artifacts
rm -rf /tmp/skbuild-release

# sdist
conda run -n astrodyn-core-env python -m build --no-isolation -s \
  -Cbuild-dir=/tmp/skbuild-release \
  /home/astror/Projects/ASTRODYN-CORE \
  -o /home/astror/Projects/ASTRODYN-CORE/dist

# wheel (use a fresh build dir)
rm -rf /tmp/skbuild-release
conda run -n astrodyn-core-env python -m build --no-isolation -w \
  -Cbuild-dir=/tmp/skbuild-release \
  /home/astror/Projects/ASTRODYN-CORE \
  -o /home/astror/Projects/ASTRODYN-CORE/dist
```

Validate artifacts:

```bash
conda run -n astrodyn-core-env twine check dist/*
```

### Required release tooling in env

Ensure these are installed before building:

- `build`
- `twine`
- `scikit-build-core>=0.8.0`
- `pybind11>=2.11.0`
- `cmake` (3.15+, any provider — pip, conda, or system)
- `ninja` (used by scikit-build-core as the default generator)

### Do not modify toolchain versions mid-build

Upgrading or downgrading `cmake`, `scikit-build-core`, or `ninja` between build
attempts without clearing the build directory will likely cause File API or
generator mismatches. Either:

1. Clear the build dir (`rm -rf /tmp/skbuild-*`) after any toolchain change, or
2. Pin versions and avoid changes during a release session.

### Troubleshooting: wheel build fails but sdist succeeds

If the wheel build fails with `IndexError: list index out of range` in
`scikit_build_core` (File API reply parsing) while the sdist builds fine:

1. **Never attempt `python -m build` with isolation first.** If you already did
   and it failed (e.g., due to network restrictions), the partial isolated venv
   left behind in `/tmp` can corrupt subsequent `--no-isolation` attempts.
   Clean up before retrying:

   ```bash
   rm -rf /tmp/skbuild-* /tmp/build-env-* /tmp/tmp*
   ```

2. **Always use `--no-isolation` with an explicit `-Cbuild-dir`** pointing to a
   fresh `/tmp` directory (see "Verified build commands" above).

3. **Do not upgrade or swap cmake/scikit-build-core/ninja versions as a
   debugging step.** This has been tried exhaustively and does not fix the
   issue. It only adds stale generator state on top of the original problem.

4. **If the failure persists after cleaning `/tmp`**, the issue is likely in the
   runtime sandbox (container overlay filesystem, restricted `/tmp` mount
   options, etc.) rather than the project itself. The project's C++ code,
   CMakeLists.txt, and pyproject.toml are known-good — the wheel builds
   successfully in a standard environment. Report the sandbox limitation
   rather than modifying project build configuration.

5. **Fallback: publish sdist only.** The sdist is a valid release artifact.
   Users installing from sdist will compile the C++ extensions locally via
   scikit-build-core, which works correctly.

## If the env is missing or broken

If `conda run -n astrodyn-core-env ...` fails because the environment does not
exist or is broken, stop and report the failure. Ask before creating or
modifying environments.
