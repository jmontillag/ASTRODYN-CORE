# Extending ASTRODYN-CORE with Custom Propagators

Last updated: 2026-02-22

This document describes how to add a new custom/analytical propagator to
ASTRODYN-CORE.  The architecture supports any propagator that can produce
Orekit-compatible `SpacecraftState` output, regardless of internal
implementation (analytical, semi-analytical, or numerical with a non-Orekit
integrator).

## 1) Design Principles

- Custom propagators are **subclasses of Orekit's `AbstractPropagator`** (via
  JPype).  This ensures they integrate with all downstream workflows: trajectory
  export, STM/covariance propagation, mission execution, ephemeris conversion.
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

## 2) Step-by-Step Extension Pattern

### Step 1: Define the provider kind

Choose a string identifier for the new propagator kind (e.g. `"geqoe"`).
This does NOT require modifying the `PropagatorKind` enum -- the registry
accepts any string.

### Step 2: Implement the AbstractPropagator subclass

Create a JPype subclass of `org.orekit.propagation.AbstractPropagator` that
wraps the custom propagation logic:

```python
# src/astrodyn_core/propagation/providers/geqoe/propagator.py

from org.orekit.propagation import AbstractPropagator

class GEqOEPropagator(AbstractPropagator):
    """J2 Taylor-series propagator in GEqOE coordinates.

    Internally propagates using analytical Taylor expansion, then converts
    results back to Orekit SpacecraftState for API compatibility.
    """

    def __init__(self, initial_state, body_constants, order=4):
        super().__init__()
        self._body_constants = body_constants
        self._order = order
        # Store initial state, set up internal GEqOE representation
        self.resetInitialState(initial_state)
        ...

    def propagate(self, start, target):
        # 1. Convert Orekit state to numpy Cartesian
        # 2. Run Taylor propagator internally
        # 3. Convert numpy result back to SpacecraftState
        # 4. Return SpacecraftState
        ...

    def get_native_state(self, target_date):
        """Return raw GEqOE state and STM as numpy arrays.

        This is backend-specific output not available through the standard
        Orekit Propagator interface.
        """
        ...
```

Key implementation notes:

- `AbstractPropagator` requires implementing the `propagate(start, target)`
  method signature.
- Convert Orekit `SpacecraftState` -> Cartesian numpy arrays at entry, run
  the analytical propagator, then convert back using Orekit constructors
  (`CartesianOrbit`, `SpacecraftState`).
- The overhead of converting back to Orekit objects is acceptable and provides
  significant benefits: frame conversion, orbit type conversion, and
  compatibility with all downstream tools.

### Step 3: Implement the provider

Create a provider dataclass that satisfies the `BuilderProvider` and/or
`PropagatorProvider` protocols:

```python
# src/astrodyn_core/propagation/providers/geqoe/provider.py

from dataclasses import dataclass
from astrodyn_core.propagation.capabilities import CapabilityDescriptor
from astrodyn_core.propagation.interfaces import BuildContext
from astrodyn_core.propagation.specs import PropagatorSpec


@dataclass(frozen=True, slots=True)
class GEqOEProvider:
    kind: str = "geqoe"
    capabilities: CapabilityDescriptor = CapabilityDescriptor(
        supports_builder=False,      # no builder pattern for analytical
        supports_propagator=True,
        supports_stm=True,           # GEqOE provides analytical STM
        is_analytical=True,
        supports_custom_output=True, # exposes raw GEqOE state/STM
    )

    def build_propagator(self, spec: PropagatorSpec, context: BuildContext):
        body_constants = context.require_body_constants()
        initial_orbit = context.require_initial_orbit()
        initial_state = ...  # create SpacecraftState from orbit
        order = spec.orekit_options.get("taylor_order", 4)
        return GEqOEPropagator(initial_state, body_constants, order=order)
```

### Step 4: Register the provider

Add the provider to the registry, either in a registration function or
directly in user code:

```python
from astrodyn_core.propagation.registry import ProviderRegistry
from astrodyn_core.propagation.providers.orekit_native import register_default_orekit_providers

registry = ProviderRegistry()
register_default_orekit_providers(registry)

# Register custom provider
from astrodyn_core.propagation.providers.geqoe.provider import GEqOEProvider
registry.register_propagator_provider(GEqOEProvider())
```

For built-in providers that ship with the package, add registration to a
function like `register_default_orekit_providers()` or create a separate
`register_analytical_providers()` function.

### Step 5: Use via PropagatorSpec

```python
from astrodyn_core import PropagatorSpec, BuildContext

spec = PropagatorSpec(
    kind="geqoe",
    mass_kg=450.0,
    orekit_options={"taylor_order": 4},
)

context = BuildContext.from_state_record(
    state_record,
    body_constants={
        "mu": float(Constants.WGS84_EARTH_MU),
        "j2": float(-Constants.WGS84_EARTH_C20),
        "re": float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS),
    },
)

propagator = factory.build_propagator(spec, context)
# propagator is an AbstractPropagator -- works with all downstream tools
```

Or let `require_body_constants()` resolve them automatically from Orekit
`Constants`:

```python
context = BuildContext.from_state_record(state_record)
# body_constants=None -> require_body_constants() resolves from Orekit Constants
```

## 3) What the Custom Propagator Gets for Free

By subclassing `AbstractPropagator` and registering via the provider pattern,
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
by users who know they are working with a specific propagator type:

```python
propagator = factory.build_propagator(spec, context)

# Standard Orekit interface -- works for any propagator
state = propagator.propagate(start, target)
pv = state.getPVCoordinates()

# Backend-specific -- only available on GEqOEPropagator
if hasattr(propagator, "get_native_state"):
    geqoe_state, geqoe_stm = propagator.get_native_state(target)
    # geqoe_state: numpy array [nu, q1, q2, p1, p2, Lr]
    # geqoe_stm: numpy array [6, 6]
```

## 5) File Layout Convention

Custom propagators should be organized under `propagation/providers/`:

```
src/astrodyn_core/propagation/providers/
    __init__.py
    integrators.py          # Orekit integrator builder helpers
    orekit_native.py        # Built-in Orekit providers
    geqoe/                  # Custom analytical propagator
        __init__.py
        propagator.py       # AbstractPropagator subclass
        provider.py         # Provider dataclass
        conversion.py       # Coordinate transforms
        jacobians.py        # Analytical Jacobians
        taylor.py           # Taylor series propagation core
        math_utils.py       # Derivative combinatorics helpers
```

## 6) Testing Custom Propagators

Custom propagator tests should verify:

1. **Round-trip consistency**: Cartesian -> internal elements -> propagate ->
   Cartesian matches expected behavior.
2. **Orekit interface compliance**: `propagator.propagate()` returns valid
   `SpacecraftState` with correct epoch, frame, and PV coordinates.
3. **Registry integration**: provider registers and resolves correctly.
4. **STM correctness** (if applicable): STM determinant near 1 for
   Hamiltonian systems; `Phi @ P0 @ Phi.T` produces valid covariance.
5. **Body constants**: test that the propagator uses constants from the
   context, not hardcoded values.

## 7) Constraints and Non-Goals

- Custom propagators must produce **Orekit-compatible output** (`SpacecraftState`).
  Pure-numpy propagators that bypass Orekit are not supported by the downstream
  workflow tooling.
- The `PropagatorKind` enum is not intended to be extended by users. Use plain
  strings for custom kinds.  The enum documents the built-in kinds only.
- Force model assembly (`assemble_force_models`) is Orekit-specific and is not
  used by analytical providers.  Analytical providers configure their force
  models internally (e.g. J2 coefficient from `body_constants`).
