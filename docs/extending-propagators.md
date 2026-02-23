# Extending ASTRODYN-CORE with Custom Propagators

Last updated: 2026-02-23

This document describes how to add a new custom/analytical propagator to
ASTRODYN-CORE.  The architecture supports any propagator that can produce
Orekit-compatible `SpacecraftState` output, regardless of internal
implementation (analytical, semi-analytical, or numerical with a non-Orekit
integrator).

The GEqOE J2 Taylor-series propagator is the reference implementation of this
pattern; its source lives in `src/astrodyn_core/propagation/geqoe/` (engine)
and `src/astrodyn_core/propagation/providers/geqoe/` (Orekit adapter).

## 1) Design Principles

- Custom propagators are **exposed as Orekit `AbstractPropagator` subclasses**
  (via JPype).  This ensures they integrate with all downstream workflows:
  trajectory export, STM/covariance propagation, mission execution, ephemeris
  conversion.
- Internally, the propagator is free to use any algorithm (Taylor series,
  closed-form solutions, custom ODE solvers, etc.).  The Orekit wrapper is a
  thin output adapter.
- Physical constants (mu, J2, Re) must **never be hardcoded**.  They are
  resolved from Orekit `Constants` (e.g. `Constants.WGS84_EARTH_MU`) either
  via `BuildContext.require_body_constants()` or directly at build time.
- The propagator may expose backend-specific output (raw numpy arrays,
  internal element states, analytical STM) alongside the standard Orekit
  `SpacecraftState` results.  Set `supports_custom_output=True` in the
  capability descriptor to signal this.
- If the propagator has dt-independent setup costs (e.g. Taylor coefficient
  computation), those should be computed once at construction / on
  `resetInitialState()` and cached, so that repeated `propagate()` calls are
  cheap.

## 2) Step-by-Step Extension Pattern

### Step 1: Define the provider kind

Choose a string identifier for the new propagator kind (e.g. `"geqoe"`).
This does NOT require modifying the `PropagatorKind` enum -- the registry
accepts any string.

### Step 2: Implement the propagator class and JPype factory

Because JPype cannot resolve Orekit Java classes at Python module-import time
(Orekit must be initialized first), **do not** inherit from `AbstractPropagator`
directly in the class definition.  Instead use the lazy factory pattern:

1. Write a **plain Python class** containing all propagation logic (no Orekit
   base class).
2. Write a **factory function** that, when called with Orekit available,
   dynamically creates a JPype subclass of `AbstractPropagator` that delegates
   to an instance of the plain class.  Cache the dynamically-created class on
   the factory function to avoid recreating it on every call.

```python
# src/astrodyn_core/propagation/providers/myprop/propagator.py

from __future__ import annotations
import numpy as np
from astrodyn_core.propagation.geqoe.conversion import BodyConstants


class MyPropagator:
    """Plain Python class — no Orekit base at definition time.

    Orekit AbstractPropagator subclassing is done lazily by
    make_orekit_myprop_propagator() below.
    """

    def __init__(self, initial_orbit, body_constants=None, order=4, mass_kg=1000.0):
        # Resolve body constants from Orekit if not provided explicitly.
        if body_constants is None:
            from org.orekit.utils import Constants
            body_constants = {
                "mu": float(Constants.WGS84_EARTH_MU),
                "j2": float(-Constants.WGS84_EARTH_C20),
                "re": float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS),
            }

        self._order = int(order)
        self._mass_kg = float(mass_kg)
        self._bc = BodyConstants(**body_constants)
        self._initial_orbit = initial_orbit

        # Extract initial Cartesian state once.
        from org.orekit.propagation import SpacecraftState
        initial_state = SpacecraftState(initial_orbit, mass_kg)
        self._y0, self._frame, self._epoch, self._mu, _ = (
            _orekit_state_to_cartesian(initial_state)
        )

        # --- Precompute any dt-independent quantities here. ---
        # Example: Taylor coefficient precomputation (see GEqOE implementation).
        # self._coeffs, self._peq_py_0 = prepare_cart_coefficients(
        #     self._y0, self._bc, self._order
        # )

        self._last_result = None

    def propagate(self, start, target=None):
        """Propagate and return an Orekit SpacecraftState."""
        if target is None:
            target = start
            start = self._epoch

        dt = float(target.durationFrom(self._epoch))

        # --- Run internal propagation (replace with real math) ---
        y_out, stm = _run_propagation(self._y0, dt, self._bc, self._order)

        self._last_result = (y_out, stm)
        return _cartesian_to_orekit_state(y_out, self._frame, target,
                                          self._mu, self._mass_kg)

    def resetInitialState(self, state):
        """Reset to a new initial state (e.g. after a manoeuvre or estimator step).

        Re-run any precomputation that depends on the initial state.
        """
        self._y0, self._frame, self._epoch, self._mu, self._mass_kg = (
            _orekit_state_to_cartesian(state)
        )
        # Recompute cached quantities for the new state.
        # self._coeffs, self._peq_py_0 = prepare_cart_coefficients(...)
        self._last_result = None

    def getInitialState(self):
        from org.orekit.propagation import SpacecraftState
        return SpacecraftState(self._initial_orbit, self._mass_kg)

    def get_native_state(self, target_date):
        """Return raw Cartesian state and STM as numpy arrays.

        Backend-specific output; not part of the standard Orekit interface.

        Returns
        -------
        y : ndarray, shape (6,)
            Cartesian state [rx, ry, rz, vx, vy, vz] in SI units.
        stm : ndarray, shape (6, 6)
            Cartesian state transition matrix from epoch to target_date.
        """
        self.propagate(target_date)
        y, stm = self._last_result
        return y.copy(), stm.copy()

    @property
    def order(self):
        return self._order

    @property
    def body_constants(self):
        return {
            "mu": self._bc.mu,
            "j2": self._bc.j2,
            "re": self._bc.re,
        }


def make_orekit_myprop_propagator(initial_orbit, body_constants=None,
                                   order=4, mass_kg=1000.0):
    """Return a MyPropagator that is also an Orekit AbstractPropagator.

    When Orekit is available the returned object is an instance of a
    dynamically-created class that inherits from both MyPropagator logic
    and AbstractPropagator.  When Orekit is unavailable, returns a plain
    MyPropagator (useful for non-propagation tests).
    """
    try:
        from org.orekit.propagation import AbstractPropagator as _AP

        if not hasattr(make_orekit_myprop_propagator, "_cls"):

            class _OrekitMyPropagator(_AP):
                """MyPropagator with Orekit AbstractPropagator base."""

                def __init__(self, initial_orbit, body_constants, order, mass_kg):
                    _AP.__init__(self)
                    self._impl = MyPropagator(
                        initial_orbit=initial_orbit,
                        body_constants=body_constants,
                        order=order,
                        mass_kg=mass_kg,
                    )
                    self.resetInitialState(self._impl.getInitialState())

                # Orekit AbstractPropagator abstract method.
                def propagateOrbit(self, date):
                    return self._impl.propagate(date).getOrbit()

                def getMass(self, date):
                    return self._impl._mass_kg

                def resetIntermediateState(self, state, forward):
                    self._impl.resetInitialState(state)

                # Delegate all public methods to the plain implementation.
                def propagate(self, *args):
                    return self._impl.propagate(*args)

                def resetInitialState(self, state):
                    return self._impl.resetInitialState(state)

                def get_native_state(self, target_date):
                    return self._impl.get_native_state(target_date)

                @property
                def order(self):
                    return self._impl.order

                @property
                def body_constants(self):
                    return self._impl.body_constants

            make_orekit_myprop_propagator._cls = _OrekitMyPropagator

        cls = make_orekit_myprop_propagator._cls
        return cls(initial_orbit, body_constants, order, mass_kg)

    except Exception:
        # Orekit not available — return plain Python propagator.
        return MyPropagator(
            initial_orbit=initial_orbit,
            body_constants=body_constants,
            order=order,
            mass_kg=mass_kg,
        )
```

Key implementation notes:

- `AbstractPropagator` requires at minimum `propagateOrbit(date)` and
  `getMass(date)` to be implemented in the JPype subclass.
- Use `resetIntermediateState(state, forward)` (not `resetInitialState`) as
  the Orekit-side hook for state resets; delegate it to `resetInitialState`.
- The `_impl` composition pattern keeps the propagation logic in plain Python
  (testable without Orekit) while the outer class satisfies the Java interface.
- Convert Orekit `SpacecraftState` → Cartesian numpy arrays at entry, run
  the propagation logic, then convert back using `CartesianOrbit` /
  `SpacecraftState` constructors.

### Step 3: Implement the provider

Create a provider dataclass that satisfies the `PropagatorProvider` protocol:

```python
# src/astrodyn_core/propagation/providers/myprop/provider.py

from dataclasses import dataclass
from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.specs import PropagatorSpec
from .propagator import make_orekit_myprop_propagator


@dataclass(frozen=True, slots=True)
class MyPropProvider:
    kind: str = "myprop"
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=False,      # no builder pattern for analytical propagators
        supports_propagator=True,
        supports_stm=True,           # provides analytical STM
        is_analytical=True,
        supports_custom_output=True, # exposes get_native_state()
    )

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext):
        body_constants = context.require_body_constants()
        initial_orbit = context.require_initial_orbit()
        order = spec.orekit_options.get("taylor_order", 4)
        mass_kg = spec.mass_kg or 1000.0
        # Always call the factory, not the plain class, so the returned
        # object has AbstractPropagator as its base when Orekit is available.
        return make_orekit_myprop_propagator(
            initial_orbit=initial_orbit,
            body_constants=body_constants,
            order=order,
            mass_kg=mass_kg,
        )
```

### Step 4: Register the provider

For propagators that ship with the package, add registration to a dedicated
function in `providers/__init__.py`:

```python
# src/astrodyn_core/propagation/providers/__init__.py

def register_analytical_providers(registry):
    from astrodyn_core.propagation.providers.geqoe.provider import GEqOEProvider
    from astrodyn_core.propagation.providers.myprop.provider import MyPropProvider
    registry.register_propagator_provider(GEqOEProvider())
    registry.register_propagator_provider(MyPropProvider())

def register_all_providers(registry):
    register_default_orekit_providers(registry)
    register_analytical_providers(registry)
```

For ad-hoc or user-defined propagators, register directly:

```python
from astrodyn_core.propagation.registry import ProviderRegistry
from mypackage.myprop.provider import MyPropProvider

registry = ProviderRegistry()
registry.register_propagator_provider(MyPropProvider())
```

### Step 5: Use via PropagatorSpec

```python
from astrodyn_core import AstrodynClient, PropagatorSpec, BuildContext

spec = PropagatorSpec(
    kind="myprop",
    mass_kg=450.0,
    orekit_options={"taylor_order": 4},
)

# body_constants=None -> require_body_constants() resolves from Orekit Constants
ctx = BuildContext(initial_orbit=orbit)

propagator = AstrodynClient().propagation.build_propagator(spec, ctx)
# propagator is an AbstractPropagator -- works with all downstream tools

# Or use from_state_record if you have an OrbitStateRecord:
# ctx = BuildContext.from_state_record(state_record)
```

## 3) What the Custom Propagator Gets for Free

By implementing the factory pattern and registering via the provider pattern,
the custom propagator automatically participates in:

- **Trajectory export** (`export_trajectory_from_propagator()`)
- **STM/covariance propagation** (`STMCovariancePropagator`) -- if the
  propagator supports `setupMatricesComputation` or provides its own STM
- **Mission execution** (`ScenarioExecutor`) -- detector-driven closed-loop
- **Ephemeris generation** (`state_series_to_ephemeris()`)
- **State I/O** -- output states are standard `OrbitStateRecord` objects
- **Frame conversion** -- Orekit handles frame transforms on output
- **Orekit estimation tooling** -- the propagator can be used inside Orekit's
  batch least squares or Kalman filter estimators

## 4) Exposing Backend-Specific Output

Custom propagators that set `supports_custom_output=True` should provide
additional methods for accessing internal state representations.  These methods
are not part of the standard `AbstractPropagator` interface and must be called
by users who know they are working with a specific propagator type.

`get_native_state(target_date)` should return **Cartesian** SI quantities, not
internal element representations, so that the output is frame-agnostic and
comparable across propagator types:

```python
propagator = factory.build_propagator(spec, context)

# Standard Orekit interface -- works for any propagator
state = propagator.propagate(target)
pv = state.getPVCoordinates()

# Backend-specific -- only available on propagators with supports_custom_output
if hasattr(propagator, "get_native_state"):
    y, stm = propagator.get_native_state(target)
    # y   : numpy array [rx, ry, rz, vx, vy, vz] in SI (metres / m/s)
    # stm : numpy array (6, 6), Cartesian-to-Cartesian state transition matrix
```

For the GEqOE propagator specifically, the raw GEqOE elements are available
by calling the engine layer directly:

```python
from astrodyn_core.propagation.geqoe.core import (
    prepare_cart_coefficients,
    evaluate_cart_taylor,
)
coeffs, peq_py_0 = prepare_cart_coefficients(y0_cart, body_constants, order=4)
# evaluate_cart_taylor returns (Cartesian states, Cartesian STM)
y_out, dy_dy0 = evaluate_cart_taylor(coeffs, peq_py_0, tspan)
```

## 5) File Layout Convention

Custom propagators follow a two-layer structure:

```
src/astrodyn_core/propagation/
    geqoe/                          # Engine layer: pure math, no Orekit dependency
        __init__.py
        core.py                     # All entry points (staged + monolithic APIs)
        state.py                    # Context and coefficient dataclasses
        conversion.py               # Coordinate transforms (rv2geqoe, geqoe2rv)
        jacobians.py                # Analytical Jacobians (get_pEqpY, get_pYpEq)
        _derivatives.py             # Chain-rule derivative utilities
        taylor_order_{1,2,3,4}.py   # Per-order Taylor expansion modules
        utils.py                    # Shared helpers

    providers/
        __init__.py                 # register_analytical_providers(),
                                    # register_all_providers()
        orekit_native.py            # Built-in Orekit providers
        integrators.py              # Orekit integrator builder helpers
        geqoe/                      # Orekit adapter layer for GEqOE
            __init__.py             # Exports: GEqOEProvider, GEqOEPropagator,
                                    #          make_orekit_geqoe_propagator
            propagator.py           # GEqOEPropagator (plain class) +
                                    # make_orekit_geqoe_propagator() (JPype factory)
            provider.py             # GEqOEProvider dataclass
```

The **engine layer** (`geqoe/`) must be importable without Orekit — all Orekit
imports are deferred to method bodies or use `try/except`.  Tests for the math
logic do not require Orekit.

The **adapter layer** (`providers/geqoe/`) wraps the engine for the Orekit
ecosystem.  The plain class (`GEqOEPropagator`) can be instantiated for
non-Orekit tests; the factory (`make_orekit_geqoe_propagator`) is used
everywhere Orekit is available.

## 6) Testing Custom Propagators

Custom propagator tests should cover two independent groups:

**Group A — no Orekit required (test the math layer):**

1. **Parity against reference**: propagated state matches a known-good
   implementation (bit-level or to tolerance).
2. **Staged API consistency**: `prepare_* + evaluate_*` produces identical
   output to the monolithic entry point.
3. **Order validation**: out-of-range orders raise `ValueError`.

**Group B — Orekit integration (marked with `@requires_orekit` or
`pytest.importorskip`):**

4. **Orekit interface compliance**: `propagate()` returns a valid
   `SpacecraftState` with correct epoch, frame, and PV coordinates.
5. **PV parity**: `propagate()` output matches `get_native_state()` (after
   Cartesian conversion) to within numerical tolerance.
6. **`propagate_array` shape**: batch output shapes are `(N, 6)` and
   `(6, 6, N)`.
7. **`resetInitialState` correctness**: propagator state after reset matches
   direct propagation from the new initial state.
8. **Registry integration**: provider registers and resolves correctly via
   `PropagatorFactory`.
9. **Body constants**: propagator uses constants from context, not hardcoded.
10. **STM correctness** (if applicable): STM determinant near 1 for
    Hamiltonian systems; `Phi @ P0 @ Phi.T` produces valid covariance.

Follow the test structure in `tests/test_geqoe_provider.py` and
`tests/test_geqoe_refactor.py` as the canonical reference.

## 7) Constraints and Non-Goals

- Custom propagators must produce **Orekit-compatible output** (`SpacecraftState`).
  Pure-numpy propagators that bypass Orekit are not supported by the downstream
  workflow tooling.
- The `PropagatorKind` enum is not intended to be extended by users. Use plain
  strings for custom kinds.  The enum documents the built-in kinds only.
- Force model assembly (`assemble_force_models`) is Orekit-specific and is not
  used by analytical providers.  Analytical providers configure their force
  models internally (e.g. J2 coefficient from `body_constants`).
- `get_native_state` must return **Cartesian** output (`[rx, ry, rz, vx, vy, vz]`
  in SI), not internal element representations, regardless of the propagator's
  internal coordinate system.
