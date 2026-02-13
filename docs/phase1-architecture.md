# Phase 1 Architecture

This repository starts with a builder-first design that keeps Orekit concepts explicit.

## Core flow

1. User defines a `PropagatorSpec`.
2. `PropagatorFactory` routes by `PropagatorKind` through `ProviderRegistry`.
3. The selected provider returns an Orekit-native object:
   - builder lane: `PropagatorBuilder`
   - propagator lane: `Propagator`

## Package layout

- `astrodyn_core.propagation.specs`: declarative config types.
- `astrodyn_core.propagation.interfaces`: provider contracts and `BuildContext`.
- `astrodyn_core.propagation.registry`: provider registration/discovery.
- `astrodyn_core.propagation.factory`: high-level construction entry points.
- `astrodyn_core.propagation.providers`: concrete implementations.

## Default providers in Phase 1

- `numerical` -> `NumericalPropagatorBuilder`
- `keplerian` -> `KeplerianPropagatorBuilder`
- `dsst` -> `DSSTPropagatorBuilder`
- `tle` -> `TLEPropagatorBuilder` or direct `TLEPropagator`

## Why this shape

- Avoids hiding Orekit APIs behind heavy wrappers.
- Keeps extension points open for future custom analytical plugins.
- Supports migration from existing config-driven pipelines.
