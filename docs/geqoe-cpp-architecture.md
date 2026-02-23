# GEqOE C++ Propagator Architecture (Phase 4)

This document defines the implementation architecture for porting the staged Python GEqOE Taylor propagator to C++ while preserving numerical parity.

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
    taylor_order_1.hpp            # order-specific kernels
    taylor_order_2.hpp
    taylor_order_3.hpp
    taylor_order_4.hpp
    taylor_pipeline.hpp           # staged API orchestration
  src/
    kepler_solver.cpp
    conversions.cpp
    jacobians.cpp
    propagator_core.cpp           # implemented now (shared runtime pieces)
    taylor_order_1.cpp            # translated equations
    taylor_order_2.cpp
    taylor_order_3.cpp
    taylor_order_4.cpp
    taylor_pipeline.cpp           # dispatch + staged wrappers
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

Planned C++ core APIs:

1. `prepare_taylor_coefficients_cpp(eq0, p, order) -> coeff_blob`
2. `evaluate_taylor_cpp(coeff_blob, dt) -> (y_prop, y_y0, map_components)`
3. `prepare_cart_coefficients_cpp(y0_cart, p, order) -> (coeff_blob, peq_py_0)`
4. `evaluate_cart_taylor_cpp(coeff_blob, peq_py_0, tspan) -> (y_out, dy_dy0)`

Monolithic wrappers call the staged functions to preserve behavior.

---

## 3) Scratch Strategy for Order Conversion

The difficult part is the huge intermediate chain in `taylor_order_1..4.py`.

### Decision: typed scratch structs per order (not string maps in final path)

- `Order1Scalars`, `Order2Scalars`, `Order3Scalars`, `Order4Scalars`
- Each struct contains all dt-independent scalar intermediates and partials used by that order and downstream orders.
- `StmAccumulatorView` carries dt-dependent arrays (`nu_nu`, `q1_nu`, ... `Lr_Lr`) for evaluation/assembly.

Why:

- safer than `unordered_map<string,double>` for long-term maintenance
- better compiler diagnostics when a field is missing
- faster and deterministic memory layout

Migration convenience:

- During initial translation of an order file, it is acceptable to stage with a temporary name-preserving map in an isolated branch.
- Before merging each order, promote to typed structs.

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

---

## 6) Testing and Acceptance Gates

### 6.1 Per-order gates

- GEqOE-space parity (`compute + evaluate`) for order N.
- Cartesian projection parity for order N.
- STM parity for order N.

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
