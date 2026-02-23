# GEqOE C++ Propagator Architecture (Phase 4)

This document defines the implementation architecture for porting the staged Python GEqOE Taylor propagator to C++ while preserving numerical parity.

## Current Implementation Status (2026-02-23)

- Implemented:
  - Shared runtime core: `propagator_core.hpp/.cpp`
  - Staged pipeline scaffolding: `taylor_pipeline.hpp/.cpp`
  - Full Order-1 port: `taylor_order_1.hpp/.cpp`
  - Python bindings for staged C++ API in `bindings.cpp`
  - Focused parity tests for staged Order-1 vs Python
- Not yet implemented:
  - Order-2/3/4 C++ kernels and dispatch
- Important constraint:
  - C++ staged path currently supports `order=1` only; higher orders raise a clear runtime error.

## Goals

- Match the staged Python pipeline semantics exactly:
  - coefficient precomputation (`prepare_*_coefficients`)
  - cheap polynomial evaluation (`evaluate_*_taylor`)
  - legacy monolithic wrappers (`j2_taylor_propagator`, `taylor_cart_propagator`)
- Keep strict separation between pure C++ core and PyBind11 wrappers.
- Make order-1..4 conversion incremental and testable.
- Preserve variable naming parity with Python to reduce transcription risk.

## Non-Goals

- Changing GEqOE math or introducing new dynamics terms.
- Refactoring equations for aesthetics before parity is reached.
- Premature micro-optimization before parity and profiling.

---

## 1) Target C++ Module Layout

The current `geqoe_cpp` module already contains Kepler + conversion + Jacobian utilities. The propagator port extends this module.

```text
src/astrodyn_core/geqoe_cpp/
  include/
    kepler_solver.hpp
    conversions.hpp
    jacobians.hpp
    propagator_core.hpp           # shared core primitives (constants, STM assembly)
    taylor_order_1.hpp            # order-specific kernels (implemented)
    taylor_order_2.hpp
    taylor_order_3.hpp
    taylor_order_4.hpp
    taylor_pipeline.hpp           # staged API orchestration (implemented)
  src/
    kepler_solver.cpp
    conversions.cpp
    jacobians.cpp
    propagator_core.cpp           # implemented (shared runtime pieces)
    taylor_order_1.cpp            # implemented (translated equations)
    taylor_order_2.cpp
    taylor_order_3.cpp
    taylor_order_4.cpp
    taylor_pipeline.cpp           # implemented (dispatch + staged wrappers)
    bindings.cpp                  # pybind wrappers only
```

---

## 2) Runtime Contracts (C++ core)

### 2.1 Constants and scales

- Input constants: `(J2, Re, mu)`
- Derived constants:
  - `length_scale = Re`
  - `time_scale = sqrt(Re^3 / mu)`
  - `mu_norm = 1.0`
  - `a_half_j2 = J2 / 2`

### 2.2 Shapes and memory layout

- GEqOE state vectors: length 6 in order `[nu, q1, q2, p1, p2, Lr]`
- `y_prop`: contiguous row-major `(M, 6)`
- `map_components`: contiguous row-major `(6, order)`
- `y_y0`: contiguous row-major `(6, 6, M)` with index mapping:
  - `idx = row * 6 * M + col * M + k`

This matches the Python STM assembly semantics used in `core.py`.

### 2.3 Staged API equivalents

Implemented C++ core APIs:

1. `prepare_taylor_coefficients_cpp(eq0, p, order) -> coeff_blob`
2. `evaluate_taylor_cpp(coeff_blob, dt) -> (y_prop, y_y0, map_components)`
3. `prepare_cart_coefficients_cpp(y0_cart, p, order) -> (coeff_blob, peq_py_0)`
4. `evaluate_cart_taylor_cpp(coeff_blob, peq_py_0, tspan) -> (y_out, dy_dy0)`

Monolithic wrappers call the staged functions to preserve behavior.

---

## 3) Scratch Strategy for Order Conversion

The difficult part is the huge intermediate chain in `taylor_order_1..4.py`.

### Decision: typed scratch structs per order (not string maps in final path)

- `Order1Coefficients` (implemented), then `Order2/3/4` typed structs
- Each struct contains all dt-independent scalar intermediates and partials used by that order and downstream orders.
- `StmAccumulatorView` carries dt-dependent arrays (`nu_nu`, `q1_nu`, ... `Lr_Lr`) for evaluation/assembly.

Why:

- safer than `unordered_map<string,double>` for long-term maintenance
- better compiler diagnostics when a field is missing
- faster and deterministic memory layout

Migration note:

- Order-1 currently uses typed structs and explicit STM accumulator vectors.
- Continue same typed pattern for Order-2/3/4.

## 3.1 Shared math utilities policy

- Do not duplicate inverse/product derivative algebra inside order kernels.
- Reuse `math_cpp` core functions from `src/astrodyn_core/math_cpp/` (`math_utils.hpp/.cpp`) in GEqOE C++ kernels.
- Keep one canonical implementation for those utilities to reduce divergence risk.

---

## 4) Order File Porting Pattern (repeat for 1..4)

For each `taylor_order_N.py`, keep exact split:

- `compute_coefficients_N` (dt-independent)
- `evaluate_order_N` (dt-dependent polynomial + STM accumulators)
- `compute_order_N` wrapper (calls both)

### Required invariants

- Order chaining remains intact:
  - Order 2 calls order 1 compute path first, etc.
- Order-2 `fic` overwrite behavior is preserved.
- Order-3 `beta_vector = [beta, bp, bpp]` reset behavior is preserved.
- STM assembly is centralized and identical to Python.
- Final normalization `y_prop[:,0] /= T` remains at end of staged run.

---

## 5) Translation Workflow for Each Large Order File

1. **Freeze reference surface**
   - Add/keep parity tests for that order in GEqOE-space and Cartesian-space.
2. **Create variable ledger**
   - Enumerate all symbols read/write in Python order file.
3. **Transcribe compute block first**
   - Keep Python variable names in C++.
4. **Transcribe evaluate block second**
   - Only dt-dependent expressions and STM accumulators.
5. **Hook into staged dispatch**
   - Add order to C++ pipeline dispatch tables.
6. **Parity gate before moving on**
   - C++ vs Python for that order only.

This minimizes blast radius and isolates defects to one order at a time.

For Order-2 onward, preserve these already-validated invariants from Python:

- Order-2 overwrites `fic` via updated inverse-derivative vector logic.
- Order-3 rebuilds `beta_vector = [beta, bp, bpp]`.
- STM assembly remains centralized in `propagator_core` path.

---

## 6) Testing and Acceptance Gates

### 6.1 Per-order gates

- GEqOE-space parity (`compute + evaluate`) for order N.
- Cartesian projection parity for order N.
- STM parity for order N.

Current gate status:

- Order-1: passing (GEqOE-space + Cartesian-space staged parity).
- Orders 2-4: pending.

### 6.2 Cross-order gates

- Full staged API parity for orders 1..4.
- Monolithic wrapper parity for orders 1..4.

### 6.3 Jacobian/Inverse consistency

- Keep strict inverse sanity checks (`pEqpY @ pYpEq ≈ I`) at tightened tolerances.
- Keep separate tolerance rationale for near-zero partial components.

---

## 7) Performance Plan (after parity)

1. Baseline single-thread parity build.
2. Hotspot profile by order and by function family.
3. Add OpenMP parallelization over state batch dimension `N`.
4. Keep deterministic mode available for parity debugging.

---

## 8) Incremental Delivery Plan

1. Shared runtime pieces (`propagator_core`) and C++ staged pipeline scaffolding.
2. Order-1 full port and parity integration.
3. Order-2 full port and parity integration.
4. Order-3 full port and parity integration.
5. Order-4 full port and parity integration.
6. Bind staged C++ API into Python provider path behind feature flag.
7. Default C++ backend once parity + stability criteria are met.

Progress update:

- Steps 1-2 are complete.
- Step 3 (Order-2) is the next active task for the next session.

## 9) Build and Setup Notes

- The supported workflow is the repo setup/install flow (`setup_env.py` + editable install), not manual `.so` copying.
- Validation command:

```bash
conda run -n astrodyn-core-env python -m pip install -e .[dev]
```

- Any local manually-copied extension artifacts in `src/` should be avoided.
