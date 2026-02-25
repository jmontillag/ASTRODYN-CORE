# Tutorial: Propagation Quickstart

This tutorial walks through the main workflows demonstrated in
`examples/quickstart.py`.

It is the best first tutorial because it exercises the facade-first API and
shows multiple propagator types in one place.

## Learning goals

By the end of this tutorial, you should be able to:

- choose an API tier (facade-first vs advanced)
- build a propagator from typed specs and a `BuildContext`
- run `keplerian`, `numerical`, `dsst`, and `tle` propagation examples
- understand when Orekit initialization is required
- export a trajectory and generate a basic orbital-elements plot

## Source script (executable truth)

- `examples/quickstart.py`

Use the script while reading this page:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode all
```

The script supports these modes:

- `basics`
- `keplerian`
- `numerical`
- `dsst`
- `tle`
- `tle_resolve`
- `plot`
- `all` (runs everything in sequence)

## Before you run it

Prerequisites:

- project installed in `astrodyn-core-env`
- Orekit available in that environment (for all modes except `basics`)
- run from the repo root so example paths and Orekit data helpers resolve as expected

See:

- [Install](../getting-started/install.md)
- [Orekit Data](../getting-started/orekit-data.md)
- [Common Setup Issues](../troubleshooting/common-setup-issues.md)

## How the script is structured (important mental model)

`examples/quickstart.py` is organized as one function per workflow:

- `run_basics()` (no Orekit required)
- `run_keplerian()`
- `run_numerical()`
- `run_dsst()`
- `run_tle()`
- `run_tle_resolve()` (downloads/caches TLEs, requires credentials)
- `run_plot()` (exports a trajectory + plot)

It also uses shared helpers in:

- `examples/_common.py`

Those helpers do three key things:

- initialize Orekit once per process
- create `examples/generated/` when needed
- build a representative LEO orbit for demos

## Step 1: Basics mode (no Orekit, model/spec discovery)

Run:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode basics
```

What it teaches:

- discover bundled propagation and spacecraft presets
- validate a `PropagatorSpec` and `SpacecraftSpec`
- load YAML-based dynamics and spacecraft config presets

This is useful even before dealing with JVM/Orekit initialization.

Key API concepts involved:

- `PropagatorSpec`
- `IntegratorSpec`
- `SpacecraftSpec`
- `load_dynamics_config(...)`
- `load_spacecraft_config(...)`

See API reference scaffolds:

- [API: `astrodyn_core`](../reference/api/astrodyn_core.md)
- [API: `astrodyn_core.propagation`](../reference/api/propagation.md)

## Step 2: Keplerian propagation (simplest Orekit-backed run)

Run:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode keplerian
```

What happens in the script:

1. Orekit is initialized (`init_orekit()`)
2. A sample LEO orbit is created (`make_leo_orbit()`)
3. `AstrodynClient` is instantiated
4. A `PropagatorSpec(kind=KEPLERIAN, ...)` is created
5. A builder is created with `BuildContext(initial_orbit=orbit)`
6. The Orekit propagator is built and propagated to `epoch + 1h`
7. Position is printed

Core pattern to remember (facade-first):

```python
app = AstrodynClient()
builder = app.propagation.build_builder(spec, BuildContext(initial_orbit=orbit))
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())
state = propagator.propagate(target_date)
```

This pattern is the basis for the numerical and DSST modes too.

## Step 3: Numerical propagation (forces + spacecraft)

Run:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode numerical
```

What this demonstrates:

- loading a bundled "medium_fidelity" dynamics preset
- attaching a spacecraft configuration
- building an Orekit numerical propagator
- inspecting which force models were assembled
- propagating and printing position/velocity

Why it matters:

- this is the likely default path for many users doing realistic LEO propagation
- it shows how typed config and Orekit-native force models work together

Watch for this pattern in the script:

- `spec = load_dynamics_config(...)`
- `spec = spec.with_spacecraft(...)`

That is the transition from preset config -> customized runtime spec.

## Step 4: DSST propagation (semi-analytical)

Run:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode dsst
```

What this demonstrates:

- using `PropagatorKind.DSST`
- configuring DSST-specific propagation/state types
- building a DSST propagator through the same facade workflow

This is a good example of why the typed-spec layer is valuable:

- the high-level build flow remains stable
- only the spec fields change depending on propagator kind

## Step 5: TLE propagation (analytical TLE mode)

Run:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode tle
```

What this demonstrates:

- constructing a `TLESpec` directly from TLE lines
- building a TLE propagator with `BuildContext()`
- propagating from the TLE epoch and inspecting orbit elements

This is the fastest path to try ASTRODYN-CORE with real-world style orbital
inputs without building a custom initial Orekit orbit object.

## Step 6: TLE resolve mode (download + cache, optional)

Run:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode tle_resolve
```

This mode is optional and more operationally complex.

Requirements:

- `spacetrack` extra installed
- valid credentials in repo-root `secrets.ini`
- network access

What it demonstrates:

- TLE query construction
- resolution against a target epoch
- local cache usage (`examples/generated/tle_cache/`)
- subsequent propagation using the resolved `TLESpec`

If you are just learning the API, skip this mode at first and come back later.

## Step 7: Export + plot mode (state workflow bridge)

Run:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode plot
```

What this demonstrates:

- building a simple Keplerian propagator
- generating an `OutputEpochSpec`
- exporting a trajectory via `app.state`
- reloading the series and plotting orbital elements via `app.mission`

Outputs written under:

- `examples/generated/quickstart_orbit_series.yaml`
- `examples/generated/quickstart_orbit_elements.png`

This is the key bridge between propagation and downstream analysis/mission tooling.

## Suggested learning path (best order)

Use this sequence instead of `--mode all` when learning:

1. `basics`
2. `keplerian`
3. `numerical`
4. `plot`
5. `dsst`
6. `tle`
7. `tle_resolve` (last, optional)

Why:

- you learn the facade + spec pattern first
- then see higher-fidelity propagation
- then see how results flow into state/mission outputs
- finally add special modes (DSST/TLE)

## How this tutorial page connects to the docs system

This page is included in the site navigation through `mkdocs.yml` under:

- `Tutorials -> Propagation Quickstart`

The API pages linked above are generated using `mkdocstrings`, which reads Python
docstrings directly from `src/astrodyn_core/...`.

That means:

- this tutorial explains the workflow
- API reference pages explain classes/functions in detail
- example scripts remain runnable truth for end-to-end behavior

## Next tutorial

- [Scenario + Mission Workflows](scenario-missions.md)
