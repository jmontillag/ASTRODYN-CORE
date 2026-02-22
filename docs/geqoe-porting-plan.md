# GEqOE Propagator Porting and Optimization Plan

**Context:** This document outlines the step-by-step phased approach for porting the Generalized Equinoctial Orbital Elements (GEqOE) Taylor series propagator from its original monolithic Python state (`temp_mosaic_modules/geqoe_utils`) into the highly optimized, C++-backed architecture of `ASTRODYN-CORE`.

**Audience:** This plan is explicitly written to guide an LLM agent through the refactoring and porting process. 

## Architectural Standard: The "Core + Wrapper" Pattern
All C++ code introduced in this project MUST strictly adhere to the "Core + Wrapper" pattern. 
*   **Pure C++ Core (`include/*.hpp` and `src/*.cpp`):** These files contain the raw mathematical logic. They accept standard C++ pointers (e.g., `double*`) and fundamental types. They must have **zero knowledge** of Python or PyBind11.
*   **PyBind11 Wrappers (`src/bindings.cpp`):** This file acts as the bridge. It unpacks Python/NumPy arrays (e.g., `py::array_t<double>`), extracts the raw memory pointers, calls the Pure C++ Core functions, and packages the results back into NumPy arrays.

---

## Phase 1: Python Refactoring & Compartmentalization
**Goal:** Break down the massive >2,000 line `propagator.py` into manageable, logical Python modules inside the main source tree. Do **not** change the mathematical logic; simply reorganize it.

1.  **Target Directory:** Create `src/astrodyn_core/propagation/geqoe/`.
2.  **State Encapsulation:** Create a `state.py` module defining a lightweight dataclass (e.g., `GEqOEState`) to hold the `[nu, q1, q2, p1, p2, Lr]` elements, replacing raw array passing where appropriate for readability.
3.  **Compartmentalization:** Split the `j2_taylor_propagator` function based on the Taylor expansion order:
    *   `src/astrodyn_core/propagation/geqoe/core.py` (Setup, normalization, main loop, and constant resolution).
    *   `src/astrodyn_core/propagation/geqoe/taylor_order_1.py`
    *   `src/astrodyn_core/propagation/geqoe/taylor_order_2.py`
    *   `src/astrodyn_core/propagation/geqoe/taylor_order_3.py`
    *   `src/astrodyn_core/propagation/geqoe/taylor_order_4.py`
4.  **Supporting Utilities:** Move and clean up `conversion.py` and `jacobians.py` from the temporary folder into `src/astrodyn_core/propagation/geqoe/`.

## Phase 2: Python Verification (Strict Parity Check)
**Goal:** Ensure the compartmentalized Python code produces the *exact* same numerical output as the original monolithic code.

1.  **Test Suite:** Create `tests/test_geqoe_refactor.py`.
2.  **Implementation:** Generate randomized initial Cartesian states. Pass them through both the original `temp_mosaic_modules/geqoe_utils/propagator.py` and the new `src/astrodyn_core/propagation/geqoe/core.py`.
3.  **Assertion:** Use `numpy.testing.assert_allclose(..., rtol=1e-13, atol=1e-13)` to guarantee mathematical equivalence. Do not proceed to Phase 3 until this passes.

## Phase 3: C++ Porting of Internal Tools (Conversions & Jacobians)
**Goal:** Port the internal GEqOE pipeline tools (Cartesian to GEqOE conversions, Jacobians) to C++ using the Core + Wrapper pattern.

1.  **Target Directory:** Create a new dedicated `src/astrodyn_core/geqoe_cpp/` module if the scope demands it.
2.  **Pure Core Implementation:**
    *   Create `include/conversions.hpp` and `src/conversions.cpp`.
    *   Create `include/jacobians.hpp` and `src/jacobians.cpp`.
    *   Translate the vectorized NumPy logic from Phase 1 into standard `for` loops operating on `double*` arrays. Ensure these functions can handle $N$ satellites simultaneously.
3.  **Wrapper Implementation:** Create/update `src/bindings.cpp` to expose these functions to Python.
4.  **Testing:** Add test cases to verify the C++ tools exactly match the refactored Python tools.

## Phase 4: C++ Porting of the Taylor Propagator
**Goal:** Port the massive unrolled algebraic equations into high-performance C++.

1.  **Pure Core Implementation:**
    *   Create `include/taylor_expansion.hpp`.
    *   Create `src/taylor_order_1.cpp` through `src/taylor_order_4.cpp`.
    *   Translate the algebraic equations directly into C++ variables (e.g., `Xp3_nu = rp3_nu*cosL + ...`).
    *   Wrap the core execution in an OpenMP parallel `for` loop (`#pragma omp parallel for`) to vectorize the calculation across $N$ states simultaneously.
2.  **Wrapper Implementation:** Expose a single `py_j2_taylor_propagator` function in `src/bindings.cpp` that accepts the initial states, time steps, and constants, and returns the propagated states and State Transition Matrices (STM).
3.  **Integration:** Modify the Python `core.py` (from Phase 1) to import and use the C++ backend by default, falling back to Python only if necessary.
4.  **Testing:** Ensure the C++ propagator exactly matches the Python propagator outputs.

## Phase 5: The ASTRODYN-CORE Orekit Adapter
**Goal:** Integrate the new C++ backed GEqOE propagator into the official `ASTRODYN-CORE` architecture.

1.  **Target File:** Create `src/astrodyn_core/propagation/providers/geqoe.py`.
2.  **Implementation:**
    *   Subclass `org.orekit.propagation.AbstractPropagator` (via JPype).
    *   Implement the capability descriptor setting `is_analytical=True` and `supports_custom_output=True`.
    *   Extract physical constants ($J_2$, $R_e$, $\mu$) dynamically from the Orekit `BuildContext` (`Constants.WGS84_EARTH_MU`, etc.). **Never hardcode these.**
    *   In the propagation loop, convert the incoming Orekit `SpacecraftState` to GEqOE, call the C++ backend, and convert the resulting GEqOE state back to an Orekit `SpacecraftState`.
3.  **Registration:** Ensure the `GEqOEPropagator` is registered with the `ProviderRegistry` under a specific string identifier (e.g., `"geqoe"`).
4.  **System Test:** Write an end-to-end test propagating an Orekit `Orbit` object using the `"geqoe"` provider via the `PropagationClient`.

The folder structure suggested in this documen is an initial guess of how the final code should be organized. From the general arquitecture of the codebase and the complexity of the porting you may consider it might be better to use another organization for the geqoe files (for example the internall tools might be better placed alongside the core propagator logic on other folders...)