# GEqOE Propagator Porting and Optimization Plan

**Context:** This document outlines the step-by-step phased approach for porting the Generalized Equinoctial Orbital Elements (GEqOE) Taylor series propagator from its original monolithic Python state (`temp_mosaic_modules/geqoe_utils`) into the highly optimized, C++-backed architecture of `ASTRODYN-CORE`.

**Audience:** This plan is explicitly written to guide an LLM agent through the refactoring and porting process.

**Last updated:** 2026-02-23

---

## Status Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Python Refactoring & Compartmentalization | COMPLETE |
| 2 | Python Verification (Parity Check) | COMPLETE |
| 5 | Orekit Provider Integration | COMPLETE |
| 2.5 | Precomputation Optimization (engine + provider) | COMPLETE |
| 3 | C++ Porting of Internal Tools | COMPLETE |
| 4 | C++ Porting of the Taylor Propagator | IN PROGRESS (Order 1 complete) |

> **Note:** Phase 5 was completed before full Phase 4 because the Python GEqOE propagator intentionally does NOT depend on the C++ `math_cpp` module. `_derivatives.py` remains the canonical Python implementation; C++ work is a parity/performance path.

---

## Session Handoff (2026-02-23)

This section captures the exact current state so the next session can resume directly.

### Completed in this session

1. **Phase 3 finalized for existing GEqOE utilities**
  - C++ conversion/Jacobian/Kepler module remains passing and integrated.
  - Existing tests still pass (`tests/test_geqoe_cpp.py`, `tests/test_geqoe_refactor.py`).

2. **Phase 4 started: Order-1 staged C++ port completed**
  - Added staged C++ API and Order-1 core:
    - `src/astrodyn_core/geqoe_cpp/include/taylor_order_1.hpp`
    - `src/astrodyn_core/geqoe_cpp/src/taylor_order_1.cpp`
    - `src/astrodyn_core/geqoe_cpp/include/taylor_pipeline.hpp`
    - `src/astrodyn_core/geqoe_cpp/src/taylor_pipeline.cpp`
  - Added pybind exposure in `src/astrodyn_core/geqoe_cpp/src/bindings.cpp` and package exports in `src/astrodyn_core/geqoe_cpp/__init__.py`:
    - `prepare_taylor_coefficients_cpp`
    - `evaluate_taylor_cpp`
    - `prepare_cart_coefficients_cpp`
    - `evaluate_cart_taylor_cpp`
  - Order-1 math preserved with Python variable-name parity.
  - Derivative helpers in `taylor_order_1.cpp` now call shared `math_cpp` kernels (`math_utils.hpp`) instead of local duplicate implementations.

3. **Build + setup workflow confirmed**
  - Repo setup path (`setup_env.py` -> `conda run -n astrodyn-core-env python -m pip install -e .[dev]`) works.
  - No manual `.so` copy is required when using the project conda env.
  - Added compiled artifact ignore patterns in `.gitignore` (`*.so`, `*.pyd`, `*.dylib`).

4. **Parity validation assets added**
  - Focused test: `tests/test_geqoe_cpp_staged_order1.py` (Python staged vs C++ staged, Order 1).
  - Example script: `examples/geqoe_cpp_order1_parity.py`.

### Current constraints

- C++ staged pipeline currently supports **order=1 only** by design.
- Calling C++ staged functions with order 2-4 raises a clear runtime error.
- Existing conversion/jacobian behavior and tolerance baselines are preserved.

### Resume checklist (next session)

1. **Start Phase 4 Order-2 port (minimal incremental path)**
  - Add `taylor_order_2.hpp/.cpp` with compute/evaluate split.
  - Reuse Order-1 coefficients/scratch as input; preserve order chaining semantics.
  - Keep `fic` overwrite behavior exactly aligned with Python Order-2.

2. **Extend staged dispatch (pipeline + bindings)**
  - Update `taylor_pipeline.cpp` dispatch to allow `order=2`.
  - Keep API signatures unchanged.

3. **Add parity tests for Order-2 staged path**
  - Mirror existing Order-1 test pattern in `tests/test_geqoe_cpp_staged_order1.py` (or split per-order test files).
  - Validate GEqOE state, STM, and map component parity vs Python staged engine.

4. **Run validation sequence**
  - `conda run -n astrodyn-core-env pytest -q tests/test_geqoe_cpp_staged_order1.py`
  - `conda run -n astrodyn-core-env pytest -q tests/test_geqoe_cpp.py tests/test_geqoe_refactor.py`
  - Optional smoke: `conda run -n astrodyn-core-env python examples/geqoe_cpp_order1_parity.py`

---

## Architectural Standard: The "Core + Wrapper" Pattern

All C++ code introduced in this project MUST strictly adhere to the "Core + Wrapper" pattern.
*   **Pure C++ Core (`include/*.hpp` and `src/*.cpp`):** These files contain the raw mathematical logic. They accept standard C++ pointers (e.g., `double*`) and fundamental types. They must have **zero knowledge** of Python or PyBind11.
*   **PyBind11 Wrappers (`src/bindings.cpp`):** This file acts as the bridge. It unpacks Python/NumPy arrays (e.g., `py::array_t<double>`), extracts the raw memory pointers, calls the Pure C++ Core functions, and packages the results back into NumPy arrays.

---

## Phase 1: Python Refactoring & Compartmentalization -- COMPLETE

**Goal:** Break down the massive >2,000 line `propagator.py` into manageable, logical Python modules inside the main source tree. Do **not** change the mathematical logic; simply reorganize it.

**What was done:**

1.  **Target Directory:** Created `src/astrodyn_core/propagation/geqoe/`.
2.  **State Encapsulation:** Created `state.py` defining:
    *   `GEqOEPropagationContext` -- lightweight context object holding `y_prop`, `map_components`, `scratch` dict, and propagation constants. The `scratch` dict is the state-passing mechanism between order modules.
    *   `GEqOEConstants` -- body constants dataclass (`mu`, `j2`, `re`).
    *   `GEqOETaylorCoefficients` -- frozen dataclass for precomputation (added in Phase 2.5).
3.  **Compartmentalization:** Split the `j2_taylor_propagator` function based on the Taylor expansion order:
    *   `core.py` -- Entry points (`taylor_cart_propagator`, `j2_taylor_propagator`), setup, normalization, STM assembly.
    *   `taylor_order_1.py` -- 1st-order Taylor expansion (~848 lines).
    *   `taylor_order_2.py` -- 2nd-order Taylor expansion (~960 lines).
    *   `taylor_order_3.py` -- 3rd-order Taylor expansion (~1155 lines).
    *   `taylor_order_4.py` -- 4th-order Taylor expansion.
4.  **Supporting Utilities:**
    *   `conversion.py` -- Cartesian <-> GEqOE conversions (`rv2geqoe`, `geqoe2rv`).
    *   `jacobians.py` -- Jacobian utilities (`get_pEqpY`, `get_pYpEq`).
    *   `_derivatives.py` -- Pure-Python derivative chain-rule utilities (renamed from `_math_compat.py`). This is the canonical implementation; does NOT depend on the C++ `math_cpp` module.
    *   `utils.py` -- Shared helper functions.

**Key design decisions:**
- Each `compute_order_N` function calls `compute_order_{N-1}` first (chaining).
- The `context.scratch` dict is the state-passing mechanism between order modules.
- Variable naming: `_0` suffix = 0th-order time derivatives, `p` suffix = higher derivatives, `_nu`/`_q1`/`_q2`/`_p1`/`_p2`/`_Lr` suffixes for partials w.r.t. initial conditions.
- Order 2 updates `fic` at line ~377-378 via `fic_vector = derivatives_of_inverse(c_vector)` and stores back to scratch.
- Order 3 creates fresh `beta_vector = np.array([beta, bp, bpp])` at line ~509.
- STM assembly happens after all orders (in `core.py`), column 0 (`_nu` partials) scaled by `T` (time_scale). Output normalization: `y_prop[:,0] /= T`.

---

## Phase 2: Python Verification (Strict Parity Check) -- COMPLETE

**Goal:** Ensure the compartmentalized Python code produces the *exact* same numerical output as the original monolithic code.

**What was done:**

1.  **Test Suite:** Created `tests/test_geqoe_refactor.py` (12 tests).
2.  **Implementation:** Tests generate randomized initial Cartesian states, pass them through both the original `temp_mosaic_modules/geqoe_utils/propagator.py` and the new refactored code.
3.  **Assertion:** `numpy.testing.assert_allclose(..., rtol=1e-13, atol=1e-13)` -- all 12 tests pass with bit-level parity.

**Tests:**
- `test_geqoe_refactor_taylor_cart_parity` -- Full pipeline parity.
- `test_geqoe_refactor_j2_core_parity` -- J2 core function parity.
- `test_geqoe_context_contract` -- Context dataclass contract.
- `test_geqoe_order_validation` -- Order parameter validation.
- `test_geqoe_order{1,2,3,4}_parity_{j2,cartesian}` -- Per-order parity (8 tests).

**Architecture cleanup completed alongside:**
- Removed `backend` parameter from public API.
- Removed vestigial `GEqOEHistoryBuffers` dataclass from `state.py`.
- Renamed `_math_compat.py` -> `_derivatives.py`.

---

## Phase 5: Orekit Provider Integration -- COMPLETE

> This phase was executed before Phases 3-4 because the Python engine is fully functional and does not need C++ backing for correctness.

**Goal:** Integrate the GEqOE propagator into the official `ASTRODYN-CORE` provider architecture so it is auto-discoverable via `PropagationClient`.

**What was done:**

1.  **Provider package:** Created `src/astrodyn_core/propagation/providers/geqoe/` with:
    *   `__init__.py` -- Package exports (`GEqOEProvider`, `GEqOEPropagator`, `make_orekit_geqoe_propagator`).
    *   `provider.py` -- `GEqOEProvider` dataclass with `kind="geqoe"`, `CapabilityDescriptor(is_analytical=True, supports_custom_output=True)`, `build_propagator()` method.
    *   `propagator.py` -- `GEqOEPropagator` class and `make_orekit_geqoe_propagator()` factory.
2.  **`GEqOEPropagator` features:**
    *   Plain Python class. Orekit `AbstractPropagator` subclassing is done lazily via `make_orekit_geqoe_propagator()` (see JPype factory pattern below).
    *   Methods: `propagate()`, `resetInitialState()`, `getInitialState()`, `get_native_state()`, `propagate_array()`.
    *   `propagate()` and `propagate_array()` use the cached coefficients from Phase 2.5 (`evaluate_cart_taylor`).
    *   `resetInitialState()` recomputes the cached coefficients for the new state via `prepare_cart_coefficients()`.
    *   Body constants resolved from Orekit WGS84 when `body_constants=None` (via `_resolve_body_constants_from_orekit()`).
    *   Mu consistency check: `_check_mu_consistency()` warns if `orbit.getMu()` disagrees with `body_constants["mu"]`.
3.  **JPype factory pattern:** `make_orekit_geqoe_propagator()` dynamically creates `_OrekitGEqOEPropagator` (a JPype subclass of `AbstractPropagator`) the first time it is called and caches the class. The Orekit subclass holds a `GEqOEPropagator` instance in `self._impl` and delegates all calls to it. This pattern is necessary because JPype does not support lazy Orekit imports at class-definition time. When Orekit is not available, the factory returns a plain `GEqOEPropagator`.
4.  **Registration:** `register_analytical_providers()` and `register_all_providers()` added to `providers/__init__.py`. `client.py`'s `build_factory()` now calls `register_all_providers()`.
5.  **Test Suite:** Created `tests/test_geqoe_provider.py` (21 tests: 14 non-Orekit + 7 Orekit integration).

**Provider architecture notes:**
- Two `Protocol` classes in `interfaces.py`: `BuilderProvider` (returns Orekit PropagatorBuilder) and `PropagatorProvider` (returns Orekit Propagator directly).
- `ProviderRegistry` stores providers in two dicts keyed by `provider.kind` (any string).
- `PropagatorFactory` resolves providers via `spec.kind` and delegates to `provider.build_propagator(spec, context)`.
- `provider.build_propagator()` calls `make_orekit_geqoe_propagator()` (not `GEqOEPropagator` directly) so the returned object always has Orekit `AbstractPropagator` as its base when Orekit is available.
- `PropagatorSpec.orekit_options` is a `Mapping[str, Any]` for custom options (e.g., `{"taylor_order": 4}`).
- `BuildContext.require_body_constants()` resolves from Orekit `Constants.WGS84_*` when `body_constants=None`.

---

## Phase 2.5: Precomputation Optimization -- COMPLETE

**Goal:** Split the Taylor coefficient computation (dt-independent) from the polynomial evaluation (dt-dependent), so that coefficients are computed once and reused across multiple time grids. Wire the optimization into the high-level `GEqOEPropagator` so it is used automatically through the provider pipeline.

**Motivation:** When propagating the same initial state to many different time points (common in trajectory analysis, filter iterations, sensor tasking), the expensive coefficient computation (~95% of work) should only happen once. The cheap polynomial evaluation (~5%) is then repeated for each time grid. Measured speedup: **3.6–5.4x** for repeated calls at the GEqOE level; **2–3x** for repeated `propagate()` calls through the Orekit adapter.

### Part A: Engine-level split (all 4 orders)

Split each `compute_order_N` into `compute_coefficients_N` (scalar math only) and `evaluate_order_N` (polynomial eval). The existing `compute_order_N` becomes a thin wrapper that calls both. This is minimal diff and preserves parity tests exactly.

#### New type in `state.py`

`GEqOETaylorCoefficients` -- Frozen dataclass (8 fields):

| Field | Description |
|-------|-------------|
| `initial_geqoe` | (6,) GEqOE state at epoch |
| `peq_py_0` | (6,6) Jacobian d(GEqOE)/d(Cartesian) at epoch (populated by `prepare_cart_coefficients`; zero placeholder in `prepare_taylor_coefficients`) |
| `constants` | `GEqOEPropagationConstants` -- normalised propagation constants |
| `order` | Taylor expansion order (1-4) |
| `scratch` | `dict` of dt-independent scalar coefficients and partials |
| `map_components` | (6, order) Taylor coefficient matrix |
| `initial_state` | `GEqOEState` named-field decomposition |
| `body_params` | `BodyConstants` or `(j2, re, mu)` tuple -- for `geqoe2rv`/`get_pYpEq` |

#### Functions in `core.py`

**GEqOE-space staged API (operates on GEqOE initial state `eq0`):**

- `prepare_taylor_coefficients(y0, p, order) -> GEqOETaylorCoefficients`  
  Runs `compute_coefficients_N` once. Strips dt-dependent STM accumulator keys from scratch before freezing. Used directly in tests and in the Cartesian helpers below.
- `evaluate_taylor(coeffs, dt) -> (y_prop, y_y0, map_components)`  
  Creates a fresh context with the new `dt` grid, injects the scalar scratch, runs `evaluate_order_N`, assembles STM. O(M) in the number of time points.

**Cartesian-space staged API (operates on Cartesian initial state `y0_cart`):**

- `prepare_cart_coefficients(y0_cart, p, order) -> (GEqOETaylorCoefficients, peq_py_0)`  
  Wraps `rv2geqoe` + `get_pEqpY` + `prepare_taylor_coefficients` in one call. Returns the coefficient set and the dt-independent epoch Jacobian separately (the Jacobian is not stored inside the frozen dataclass to keep the GEqOE-space API clean).
- `evaluate_cart_taylor(coeffs, peq_py_0, tspan) -> (y_out, dy_dy0)`  
  Calls `evaluate_taylor`, then `geqoe2rv` + `get_pYpEq` + Cartesian STM composition. This is the fast path used by `GEqOEPropagator`.

**Monolithic API (unchanged, calls both stages internally):**

- `j2_taylor_propagator(dt, y0, p, order)` -- unchanged public signature.
- `taylor_cart_propagator(tspan, y0, p, order)` -- unchanged public signature.

#### Split status

| Order | `compute_coefficients_N` | `evaluate_order_N` | `compute_order_N` wrapper | Tests passing |
|-------|--------------------------|-------------------|--------------------------|--------------|
| 1 | DONE | DONE | DONE | YES (12/12) |
| 2 | DONE | DONE | DONE | YES (12/12) |
| 3 | DONE | DONE | DONE | YES (12/12) |
| 4 | DONE | DONE | DONE | YES (12/12) |

#### Critical implementation details

1. **STM accumulators are dt-dependent**: In order 2+, the STM accumulator reads (e.g., `Lr_nu`, `q1_nu`, etc.) from scratch are values written by `evaluate_order_{N-1}` and are dt-dependent arrays. In the split version, these reads MUST be in `evaluate_order_N`, NOT `compute_coefficients_N`.

2. **`prepare_taylor_coefficients` uses a dummy `dt=1.0`** to size scratch arrays during `compute_coefficients_1` (which initialises `y_prop`, `y_y0`, `map_components` using `M = len(ctx.dt_norm)`). After the coefficient run, only scalar (non-STM-accumulator) scratch entries are kept. The STM accumulator keys that are filtered out are: `nu_nu`, `{q1,q2,p1,p2,Lr}_nu`, `{Lr,q1,q2,p1,p2}_{Lr,q1,q2,p1,p2}`.

3. **Order 2 updates `fic`**: At line ~377-378, `fic_vector = derivatives_of_inverse(c_vector)` overwrites `fic` and introduces `ficp`. The updated `fic` is stored back to scratch.

4. **LSP false positives**: The split introduces no new bugs, but type-checkers may report false positives for:
   - `fic_vector[0]` -- LSP thinks `derivatives_of_inverse` returns `float` but it returns `np.ndarray`.
   - `ctx.y_prop`/`ctx.map_components` subscript -- initialized as `None`, populated by `compute_coefficients_1`.

### Part B: Provider-level wiring

`GEqOEPropagator` now uses the staged API throughout:

- **`__init__`**: calls `prepare_cart_coefficients(self._y0, self._bc, self._order)` once and stores `self._coeffs: GEqOETaylorCoefficients` and `self._peq_py_0: np.ndarray`. When Orekit is not available, both are `None`.
- **`propagate(target)`**: calls `evaluate_cart_taylor(self._coeffs, self._peq_py_0, [dt_seconds])` instead of `taylor_cart_propagator(...)`.
- **`propagate_array(dt_seconds)`**: calls `evaluate_cart_taylor(self._coeffs, self._peq_py_0, dt_seconds)` instead of `taylor_cart_propagator(...)`. Guard changed from `self._y0 is None` to `self._coeffs is None`.
- **`resetInitialState(state)`**: after updating `self._y0`/`self._epoch`, calls `prepare_cart_coefficients` to refresh the cache for the next propagation batch (used in estimator iterations and manoeuvre-like workflows).

---

## Phase 3: C++ Porting of Internal Tools -- COMPLETE

**Goal:** Port the internal GEqOE pipeline tools (Cartesian to GEqOE conversions, Jacobians) to C++ using the Core + Wrapper pattern.

Completed implementation resides in `src/astrodyn_core/geqoe_cpp/`:

- Pure core:
  - `include/conversions.hpp` / `src/conversions.cpp`
  - `include/jacobians.hpp` / `src/jacobians.cpp`
  - `include/kepler_solver.hpp` / `src/kepler_solver.cpp`
- Wrapper:
  - `src/bindings.cpp` exposes the C++ functions to Python.
- Validation:
  - `tests/test_geqoe_cpp.py` parity tests pass against Python reference.

**Existing C++ reference:** The only C++ module currently in the repo is `math_cpp` at `src/astrodyn_core/math_cpp/`:
- `include/math_utils.hpp` -- Pure C++ header (namespace `astrodyn_core::math`, raw `double*` pointers).
- `src/math_utils.cpp` -- Pure C++ implementation.
- `src/bindings.cpp` -- PyBind11 wrappers.
- CMake target: `pybind11_add_module(math_utils_cpp ...)`.

---

## Phase 4: C++ Porting of the Taylor Propagator -- IN PROGRESS (Order 1 complete)

**Goal:** Port the massive unrolled algebraic equations into high-performance C++.

1.  **Pure Core Implementation:**
    *   Create `include/taylor_expansion.hpp`.
    *   Create `src/taylor_order_1.cpp` through `src/taylor_order_4.cpp`.
    *   Translate the algebraic equations directly into C++ variables (e.g., `Xp3_nu = rp3_nu*cosL + ...`).
    *   Wrap the core execution in an OpenMP parallel `for` loop (`#pragma omp parallel for`) to vectorize the calculation across $N$ states simultaneously.
2.  **Wrapper Implementation:** Expose a single `py_j2_taylor_propagator` function in `src/bindings.cpp` that accepts the initial states, time steps, and constants, and returns the propagated states and State Transition Matrices (STM).
3.  **Integration:** Modify the Python `core.py` (from Phase 1) to import and use the C++ backend by default, falling back to Python only if necessary.
4.  **Testing:** Ensure the C++ propagator exactly matches the Python propagator outputs.

**Current status:**

- **Order 1 implemented and validated** (GEqOE-space + Cartesian-space staged parity).
- Shared staged pipeline scaffolding is in place (`taylor_pipeline.hpp/.cpp`).
- `propagator_core` reused for constants, time normalization, STM assembly.

**Remaining scope:**

- Port Order 2, 3, 4 math kernels and enable dispatch incrementally per order.

**Note on Phase 2.5 synergy:** The precomputation split (`compute_coefficients_N` / `evaluate_order_N`) maps naturally to C++ and is now active for Order 1.

---

## File Map

### GEqOE Engine (`propagation/geqoe/`)

```
src/astrodyn_core/propagation/geqoe/
    __init__.py          -- Package exports: GEqOEState, GEqOEPropagationConstants,
                            GEqOEPropagationContext, j2_taylor_propagator,
                            taylor_cart_propagator
    state.py             -- GEqOEPropagationContext, GEqOEPropagationConstants,
                            GEqOEState, GEqOETaylorCoefficients
    core.py              -- All entry points:
                              prepare_taylor_coefficients, evaluate_taylor    (GEqOE-space staged)
                              prepare_cart_coefficients, evaluate_cart_taylor  (Cartesian-space staged)
                              j2_taylor_propagator, taylor_cart_propagator      (monolithic, public API)
                              build_context, _run_staged_j2                     (internal helpers)
    utils.py             -- Shared helpers
    conversion.py        -- Cartesian <-> GEqOE (rv2geqoe, geqoe2rv, BodyConstants)
    jacobians.py         -- Jacobian utilities (get_pEqpY, get_pYpEq)
    _derivatives.py      -- Pure-Python derivative chain-rule utilities
    _legacy_loader.py    -- Legacy module importer (kept for test comparisons only)
    taylor_order_1.py    -- 1st order (compute_coefficients_1 / evaluate_order_1 / compute_order_1)
    taylor_order_2.py    -- 2nd order (compute_coefficients_2 / evaluate_order_2 / compute_order_2)
    taylor_order_3.py    -- 3rd order (compute_coefficients_3 / evaluate_order_3 / compute_order_3)
    taylor_order_4.py    -- 4th order (compute_coefficients_4 / evaluate_order_4 / compute_order_4)
```

### Provider Integration (`propagation/providers/geqoe/`)

```
src/astrodyn_core/propagation/providers/geqoe/
    __init__.py          -- Package exports: GEqOEProvider, GEqOEPropagator,
                            make_orekit_geqoe_propagator
    provider.py          -- GEqOEProvider dataclass (kind="geqoe");
                            build_propagator() calls make_orekit_geqoe_propagator()
    propagator.py        -- GEqOEPropagator (plain Python class, uses staged API);
                            make_orekit_geqoe_propagator() (JPype AbstractPropagator factory)
```

### Tests

```
tests/
    test_geqoe_refactor.py   -- 12 parity tests (Phase 1/2)
    test_geqoe_provider.py   -- 21 provider tests (Phase 5): 14 non-Orekit + 7 Orekit integration
```

### Examples

```
examples/
    geqoe_propagator.py        -- 4-mode demo: provider pipeline, direct adapter,
                                   pure-numpy engine, multi-grid benchmark
    geqoe_legacy_vs_staged.py  -- Parity check + multi-grid performance benchmark
                                   (legacy vs monolithic vs staged)
```

### Legacy Reference (read-only)

```
temp_mosaic_modules/geqoe_utils/propagator.py  -- Original 2,257-line monolith
```

---

## Environment Notes

- **Conda environment `astrodyn-core-env`** is required for ALL Python execution. The base Python does NOT have Orekit.
- **Run commands with:** `conda run -n astrodyn-core-env python -m pytest ...`
- **Orekit JVM init pattern** (required in tests before any `org.orekit.*` imports):
  ```python
  orekit = pytest.importorskip("orekit")
  orekit.initVM()
  from orekit.pyhelpers import setup_orekit_curdir
  setup_orekit_curdir()
  ```
- **Body constants** must ALWAYS be resolved from Orekit -- never hardcoded. `GEqOEPropagator.__init__` accepts `body_constants: Mapping[str, float] | None = None` and resolves from `org.orekit.utils.Constants` (WGS84) when None.
- **OpenCode skill** documented at `.opencode/skills/python-testing/SKILL.md`.
