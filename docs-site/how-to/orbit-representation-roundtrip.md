# How-To: Validate an Orbit Representation Round-Trip

This recipe shows how to verify that a Keplerian state definition converts to
an Orekit orbit and back without losing precision.

It is based on:

- `examples/cookbook/orbit_comparison.py`

## When to use this recipe

Use this when you want to:

- sanity-check a new `OrbitStateRecord`
- confirm units/angles are being interpreted correctly
- verify a state serialization pipeline before running propagation

## What it demonstrates

The script:

1. Defines an `OrbitStateRecord` in **Keplerian representation**
2. Converts it to an Orekit orbit via `app.state.to_orekit_orbit(...)`
3. Extracts Keplerian elements back from Orekit
4. Prints the Cartesian state implied by the orbit
5. Computes round-trip errors (`a`, `e`, `i`)

Because this is a representation conversion check (not a propagation run), it
is short and deterministic.

## Run it

```bash
conda run -n astrodyn-core-env python examples/cookbook/orbit_comparison.py
```

Expected outcome:

- prints input Keplerian elements
- prints Orekit-converted Keplerian values
- prints Cartesian position/velocity
- prints near-zero round-trip errors (machine precision)

## Core data shape

The key input is an `OrbitStateRecord` with:

- `representation="keplerian"`
- `frame="GCRF"`
- `elements={...}` containing:
  - `a_m`
  - `e`
  - `i_deg`
  - `argp_deg`
  - `raan_deg`
  - `anomaly_deg`
  - `anomaly_type`
- `mu_m3_s2="WGS84"`

This is a good template for building your own initial conditions in a
state-file-friendly format.

## What "good" looks like

The script checks:

- `|da|` in meters
- `|de|`
- `|di|` in degrees

In the bundled example, all are effectively zero (machine precision). If your
values are larger than expected, the most common causes are:

- wrong angle units (degrees vs radians)
- incorrect anomaly type
- frame mismatch
- incorrect `mu` constant

## Adapt it to your own state

Replace the `kep_state` values in `examples/cookbook/orbit_comparison.py` with
your own state and rerun the script.

Suggested practice:

1. Validate your initial state with this recipe
2. Then use the same `OrbitStateRecord` in a propagation example
3. Compare propagated outputs with confidence that the initial conversion is correct

## Related docs

- [Propagation Quickstart](../tutorials/propagation-quickstart.md)
- [Scenario + Mission Workflows](../tutorials/scenario-missions.md)
- [API: `astrodyn_core`](../reference/api/astrodyn_core.md)
