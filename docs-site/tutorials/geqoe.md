# Tutorial: GEqOE Propagator

This tutorial walks through `examples/geqoe_propagator.py`, which demonstrates
the GEqOE J2 Taylor-series propagator at four usage levels.

It is an advanced tutorial intended for users who want to evaluate or adopt the
analytical GEqOE workflow, either through the standard ASTRODYN-CORE provider
pipeline or through lower-level interfaces.

## Learning goals

By the end of this tutorial, you should be able to:

- build a GEqOE propagator through the standard `AstrodynClient` pipeline
- use the direct `GEqOEPropagator` adapter with Orekit objects
- run the pure-numpy GEqOE engine without Orekit
- understand the cached-coefficient benchmark mode at a high level
- choose which GEqOE usage level fits your use case

## Source script (executable truth)

- `examples/geqoe_propagator.py`

Run all modes:

```bash
conda run -n astrodyn-core-env python examples/geqoe_propagator.py --mode all
```

Supported modes:

- `provider`
- `adapter`
- `numpy`
- `benchmark`
- `all`

## Before you run it

Prerequisites depend on the mode:

- `provider`: requires Orekit (uses `AstrodynClient` + provider registry)
- `adapter`: requires Orekit (direct Orekit adapter wrapper)
- `numpy`: no Orekit required (pure numpy GEqOE engine)
- `benchmark`: requires Orekit (constructs `GEqOEPropagator` and compares strategies)

If you are just learning the concepts, start with:

- `provider`
- then `adapter`
- then `numpy`
- and leave `benchmark` for last

## Mental model: four API tiers for the same GEqOE capability

The script intentionally shows the same propagator family at different levels of
abstraction:

1. **Provider pipeline** (most consistent with the rest of ASTRODYN-CORE)
2. **Direct Orekit adapter** (advanced control, still Orekit-centric)
3. **Pure-numpy engine** (algorithm-level access)
4. **Benchmark mode** (performance behavior, coefficient caching)

This is useful because different teams need different integration styles.

## Mode 1: `provider` (recommended starting point)

Run:

```bash
conda run -n astrodyn-core-env python examples/geqoe_propagator.py --mode provider
```

What this demonstrates:

- create `PropagatorSpec(kind=\"geqoe\")`
- build via `app.propagation.build_propagator(spec, ctx)`
- use the resulting propagator through a standard ASTRODYN-CORE interface
- inspect resolved body constants (`mu`, `j2`, `re`) from Orekit

Why this mode matters:

- it proves GEqOE can be used through the same facade/provider workflow as
  `numerical`, `keplerian`, `dsst`, and `tle`
- this is the best entry point for user code that may switch propagator kinds

## Mode 2: `adapter` (direct `GEqOEPropagator`)

Run:

```bash
conda run -n astrodyn-core-env python examples/geqoe_propagator.py --mode adapter
```

What this demonstrates:

- instantiate `GEqOEPropagator` directly from an Orekit orbit
- call `propagate(target_date)` for `SpacecraftState` output
- call `get_native_state(target_date)` for raw numpy state + STM
- call `propagate_array(dt_grid)` for batch propagation
- call `resetInitialState(...)` for maneuver-like workflows

Use this mode when:

- you want GEqOE-specific methods not exposed by the higher-level facade
- you are integrating with custom analysis logic that consumes raw numpy arrays
- you want an Orekit-facing adapter but not the full factory flow

## Mode 3: `numpy` (pure-numpy GEqOE engine)

Run:

```bash
conda run -n astrodyn-core-env python examples/geqoe_propagator.py --mode numpy
```

What this demonstrates:

- direct use of `taylor_cart_propagator(...)`
- manual state vector `y0` in SI units
- explicit Earth/body constants
- Taylor-order comparison (1..4)
- trajectory and STM output arrays over a time grid

Why this mode is useful:

- no Orekit dependency (algorithm experimentation / testing)
- easiest way to study GEqOE numerical behavior in isolation
- convenient for validating staged implementations and parity

## Mode 4: `benchmark` (cached coefficients and performance behavior)

Run:

```bash
conda run -n astrodyn-core-env python examples/geqoe_propagator.py --mode benchmark
```

What the benchmark compares:

- baseline: repeated calls to the lower-level `taylor_cart_propagator(...)`
  (recomputes coefficients each time)
- `GEqOEPropagator.propagate()` loop (uses cached coefficients)
- `GEqOEPropagator.propagate_array()` batch call (cached + vectorized)

What to take away:

- cached coefficients matter when propagating many epochs from the same initial
  condition
- `propagate_array()` is typically fastest for batch workloads
- the script also performs a parity check against the baseline implementation

This mode is primarily for performance intuition and validation, not first-time
adoption.

## Which GEqOE mode should you use?

- **Most application scripts**: `provider`
- **Advanced Orekit-centric workflows**: `adapter`
- **Algorithm research / parity work / unit tests**: `numpy`
- **Performance characterization**: `benchmark`

## GEqOE vs other propagator kinds (practical guidance)

GEqOE is a specialized analytical/semi-analytical option in this project.

Use GEqOE when you want:

- fast repeated propagation from a fixed initial condition
- access to native arrays / STM-friendly workflows
- a J2-focused analytical model and benchmarkable Taylor behavior

Use `numerical` / `dsst` when you need:

- broader force-model realism via Orekit force assemblies
- workflows already aligned to standard Orekit numerical/DSST semantics

## Common pitfalls

- **Units in pure-numpy mode are SI**
  `y0` and body constants are in meters / m/s / SI units.

- **Do not start with benchmark mode**
  It is easy to confuse performance behavior with API usage behavior.

- **Provider mode still returns a real propagator object**
  You can use it like other propagators in the ASTRODYN-CORE ecosystem.

## Related docs

- [Propagation Quickstart](propagation-quickstart.md)
- [How-To / Cookbook](../how-to/index.md)
- [API: `astrodyn_core.propagation`](../reference/api/propagation.md)
- [Advanced Docs](../advanced/index.md)
