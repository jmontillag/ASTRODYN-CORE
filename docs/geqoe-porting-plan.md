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
| 2.5 | Precomputation Optimization | DONE |
| 3 | C++ Porting of Internal Tools | NOT STARTED |
| 4 | C++ Porting of the Taylor Propagator | NOT STARTED |

> **Note:** Phase 5 was completed before Phases 3-4 because the Python GEqOE propagator intentionally does NOT depend on the C++ `math_cpp` module. `_derivatives.py` is the canonical pure-Python implementation. C++ porting (Phases 3-4) is a future performance optimization.

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
    *   `jacobians.py` -- Jacobian utilities (`get_pEqpY`).
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
    *   Wraps `taylor_cart_propagator` with Orekit-compatible interface.
    *   Methods: `propagate()`, `resetInitialState()`, `getInitialState()`, `get_native_state()`, `propagate_array()`.
    *   Body constants resolved from Orekit WGS84 when `body_constants=None` (via `_resolve_body_constants_from_orekit()`).
    *   Mu consistency check: `_check_mu_consistency()` warns if `orbit.getMu()` disagrees with `body_constants["mu"]`.
    *   `make_orekit_geqoe_propagator()` creates a JPype `AbstractPropagator` subclass instance.
3.  **Registration:** `register_analytical_providers()` and `register_all_providers()` added to `providers/__init__.py`. `client.py`'s `build_factory()` now calls `register_all_providers()`.
4.  **Test Suite:** Created `tests/test_geqoe_provider.py` (21 tests: 14 non-Orekit + 7 Orekit integration).

**Provider architecture notes:**
- Two `Protocol` classes in `interfaces.py`: `BuilderProvider` (returns Orekit PropagatorBuilder) and `PropagatorProvider` (returns Orekit Propagator directly).
- `ProviderRegistry` stores providers in two dicts keyed by `provider.kind` (any string).
- `PropagatorFactory` resolves providers via `spec.kind` and delegates to `provider.build_propagator(spec, context)`.
- `PropagatorSpec.orekit_options` is a `Mapping[str, Any]` for custom options (e.g., `{"taylor_order": 4}`).
- `BuildContext.require_body_constants()` resolves from Orekit `Constants.WGS84_*` when `body_constants=None`.

---

## Phase 2.5: Precomputation Optimization -- DONE

**Goal:** Split the Taylor coefficient computation (dt-independent) from the polynomial evaluation (dt-dependent), so that coefficients are computed once and reused across multiple time grids.

**Motivation:** When propagating the same initial state to many different time points (common in trajectory analysis), the expensive coefficient computation (~95% of work) should only happen once. The cheap polynomial evaluation (~5%) is then repeated for each time grid.

### Approach: Split each `compute_order_N`

Split each `compute_order_N` into `compute_coefficients_N` (scalar math only) and `evaluate_order_N` (polynomial eval). The existing `compute_order_N` becomes a thin wrapper that calls both. This is minimal diff and preserves parity tests exactly.

### New types (in `state.py`)

- `GEqOETaylorCoefficients` -- Frozen dataclass holding: `initial_geqoe`, `peq_py_0`, `constants`, `order`, `scratch` (dict of all computed coefficients), `map_components`, `initial_state`, `body_params`.

### New functions to add (in `core.py`)

- `prepare_taylor_coefficients(y0, p, order) -> GEqOETaylorCoefficients` -- Runs steps 1-3 (rv2geqoe, get_pEqpY, coefficient computation). Returns frozen coefficient set.
- `evaluate_taylor(coeffs, tspan) -> (y_out, stm)` -- Runs steps 4-5 (polynomial eval + GEqOE->Cartesian + STM composition). Cheap, O(N).

### Split pattern for each order module

**`compute_coefficients_N(ctx)`** contains:
1. Calls `compute_coefficients_{N-1}(ctx)` (chaining).
2. Reads ALL intermediates from scratch (NOT the STM accumulators -- those are dt-dependent).
3. Computes all scalar coefficients, partials, vectors.
4. Stores EOM values, EOM partials, and all intermediate values to scratch.
5. Fills `map_components[:, N-1]`.

**`evaluate_order_N(ctx)`** contains:
1. Calls `evaluate_order_{N-1}(ctx)` first (which populates STM accumulators from lower orders).
2. Reads the STM accumulators from scratch (written by `evaluate_order_{N-1}`).
3. Reads the Nth-order EOM partial coefficients from scratch (written by `compute_coefficients_N`).
4. Computes `dt_N = dt_norm**N`.
5. Adds `coeff * dt_N / N!` to `y_prop`.
6. Adds `partial_coeff * dt_N / N!` to STM accumulators.
7. Stores updated STM accumulators back to scratch.

**`compute_order_N(ctx)`** is a thin wrapper calling both.

### Current split status

| Order | compute_coefficients_N | evaluate_order_N | compute_order_N wrapper | Tests passing |
|-------|----------------------|-----------------|----------------------|--------------|
| 1 | DONE | DONE | DONE | YES (12/12) |
| 2 | DONE | DONE | DONE | YES (12/12) |
| 3 | DONE | DONE | DONE | YES (12/12) |
| 4 | DONE | DONE | DONE | YES (12/12) |

### Exact split points for remaining files

**Order 3** (`taylor_order_3.py`, ~1155 lines):
- Split at line 816 (`dt3 = dt_norm**3`).
- `compute_coefficients_3` calls `compute_coefficients_2`.
- STM accumulator reads (lines 389-399) must move to `evaluate_order_3`.
- dt-dependent section: lines 816-931.

**Order 4** (`taylor_order_4.py`):
- Same pattern. Split at the `dt4 = dt_norm**4` line.
- `compute_coefficients_4` calls `compute_coefficients_3`.
- STM accumulator reads must move to `evaluate_order_4`.

### Changes to `GEqOEPropagator` (after all splits)

- Constructor calls `prepare_taylor_coefficients` once at init and caches the result.
- `propagate()`, `propagate_array()`, `get_native_state()` call `evaluate_taylor(self._coeffs, dt)`.
- `resetInitialState()` recomputes `self._coeffs` with the new state.

### What stays the same

- `taylor_cart_propagator()` keeps its current signature (calls prepare + evaluate internally).
- `j2_taylor_propagator()` same -- public API unchanged.
- All parity tests stay unchanged.

### Critical implementation details

1. **STM accumulators are dt-dependent**: In order 2+, the STM accumulator reads (e.g., `Lr_nu`, `q1_nu`, etc.) from scratch are values written by `evaluate_order_{N-1}` and are dt-dependent arrays. In the split version, these reads MUST be in `evaluate_order_N`, NOT `compute_coefficients_N`.

2. **Order 2 updates `fic`**: At line ~377-378, `fic_vector = derivatives_of_inverse(c_vector)` overwrites `fic` and introduces `ficp`. The updated `fic` is stored back to scratch.

3. **LSP false positives**: The split introduces no new bugs, but type-checkers may report false positives for:
   - `fic_vector[0]` -- LSP thinks `derivatives_of_inverse` returns `float` but it returns `np.ndarray`.
   - `ctx.y_prop`/`ctx.map_components` subscript -- initialized as `None`, populated by `compute_coefficients_1`.

---

## Phase 3: C++ Porting of Internal Tools -- NOT STARTED

**Goal:** Port the internal GEqOE pipeline tools (Cartesian to GEqOE conversions, Jacobians) to C++ using the Core + Wrapper pattern.

1.  **Target Directory:** Create a new dedicated `src/astrodyn_core/geqoe_cpp/` module if the scope demands it, or place alongside the existing `math_cpp` module.
2.  **Pure Core Implementation:**
    *   Create `include/conversions.hpp` and `src/conversions.cpp`.
    *   Create `include/jacobians.hpp` and `src/jacobians.cpp`.
    *   Translate the vectorized NumPy logic from Phase 1 into standard `for` loops operating on `double*` arrays. Ensure these functions can handle $N$ satellites simultaneously.
3.  **Wrapper Implementation:** Create/update `src/bindings.cpp` to expose these functions to Python.
4.  **Testing:** Add test cases to verify the C++ tools exactly match the refactored Python tools.

**Existing C++ reference:** The only C++ module currently in the repo is `math_cpp` at `src/astrodyn_core/math_cpp/`:
- `include/math_utils.hpp` -- Pure C++ header (namespace `astrodyn_core::math`, raw `double*` pointers).
- `src/math_utils.cpp` -- Pure C++ implementation.
- `src/bindings.cpp` -- PyBind11 wrappers.
- CMake target: `pybind11_add_module(math_utils_cpp ...)`.

---

## Phase 4: C++ Porting of the Taylor Propagator -- NOT STARTED

**Goal:** Port the massive unrolled algebraic equations into high-performance C++.

1.  **Pure Core Implementation:**
    *   Create `include/taylor_expansion.hpp`.
    *   Create `src/taylor_order_1.cpp` through `src/taylor_order_4.cpp`.
    *   Translate the algebraic equations directly into C++ variables (e.g., `Xp3_nu = rp3_nu*cosL + ...`).
    *   Wrap the core execution in an OpenMP parallel `for` loop (`#pragma omp parallel for`) to vectorize the calculation across $N$ states simultaneously.
2.  **Wrapper Implementation:** Expose a single `py_j2_taylor_propagator` function in `src/bindings.cpp` that accepts the initial states, time steps, and constants, and returns the propagated states and State Transition Matrices (STM).
3.  **Integration:** Modify the Python `core.py` (from Phase 1) to import and use the C++ backend by default, falling back to Python only if necessary.
4.  **Testing:** Ensure the C++ propagator exactly matches the Python propagator outputs.

**Note on Phase 2.5 synergy:** The precomputation split (`compute_coefficients_N` / `evaluate_order_N`) maps naturally to C++: the coefficient functions become the heavy C++ core, and the evaluation functions can remain lightweight (potentially even stay in Python for flexibility).

---

## File Map

### GEqOE Engine
```
src/astrodyn_core/propagation/geqoe/
    __init__.py          -- Package exports
    state.py             -- GEqOEPropagationContext, GEqOEConstants, GEqOETaylorCoefficients
    core.py              -- Entry points (taylor_cart_propagator, j2_taylor_propagator)
    utils.py             -- Shared helpers
    conversion.py        -- Cartesian <-> GEqOE (rv2geqoe, geqoe2rv)
    jacobians.py         -- Jacobian utilities (get_pEqpY)
    _derivatives.py      -- Pure-Python derivative chain-rule utilities
    _legacy_loader.py    -- Legacy module importer (kept for test comparisons only)
    taylor_order_1.py    -- 1st order (SPLIT: compute_coefficients_1 / evaluate_order_1 / compute_order_1)
    taylor_order_2.py    -- 2nd order (SPLIT: compute_coefficients_2 / evaluate_order_2 / compute_order_2)
    taylor_order_3.py    -- 3rd order (SPLIT: compute_coefficients_3 / evaluate_order_3 / compute_order_3)
    taylor_order_4.py    -- 4th order (SPLIT: compute_coefficients_4 / evaluate_order_4 / compute_order_4)
```

### Provider Integration
```
src/astrodyn_core/propagation/providers/geqoe/
    __init__.py          -- Package exports (GEqOEProvider, GEqOEPropagator, make_orekit_geqoe_propagator)
    provider.py          -- GEqOEProvider dataclass (kind="geqoe")
    propagator.py        -- GEqOEPropagator class, make_orekit_geqoe_propagator factory
```

### Tests
```
tests/
    test_geqoe_refactor.py   -- 12 parity tests (Phase 1/2)
    test_geqoe_provider.py   -- 21 provider tests (Phase 5)
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
