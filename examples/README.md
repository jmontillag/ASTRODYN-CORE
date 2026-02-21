# Examples

This folder uses three entry points.

## Recommended API style

Examples are façade-first by default:

- Start with `AstrodynClient`
- Use domain façades (`app.propagation`, `app.state`, `app.mission`, `app.uncertainty`, `app.tle`)
- Keep low-level factory/registry wiring for advanced Orekit-native control only

## 1) Quickstart

Run broad propagation capabilities in one script:

```bash
python examples/quickstart.py --mode all
```

Modes:
- `basics`
- `keplerian`
- `numerical`
- `dsst`
- `tle`
- `tle_resolve` (downloads TLE via Space-Track using root `secrets.ini`, caches under `examples/generated/tle_cache/`)
- `plot`

## 2) Scenario + Missions

Run scenario/state-file workflows and mission execution modes:

```bash
python examples/scenario_missions.py --mode all
```

Modes:
- `io`
- `inspect`
- `intent`
- `detector`

## 3) Uncertainty

Run covariance/STM uncertainty workflows:

```bash
python examples/uncertainty.py
```

## 4) Cookbook

Self-contained topical examples in `examples/cookbook/`:

```bash
python examples/cookbook/multi_fidelity_comparison.py
python examples/cookbook/orbit_comparison.py
python examples/cookbook/force_model_sweep.py
python examples/cookbook/ephemeris_from_oem.py
python examples/cookbook/sma_maintenance_analysis.py
```

Examples:
- `multi_fidelity_comparison.py` — Compare Keplerian, DSST, and numerical propagation fidelity
- `orbit_comparison.py` — Cartesian to Keplerian round-trip verification
- `force_model_sweep.py` — Gravity field degree/order convergence analysis
- `ephemeris_from_oem.py` — OEM file parse and ephemeris round-trip
- `sma_maintenance_analysis.py` — Full SMA maintenance mission workflow (scenario -> detector execution -> analysis)

## Generated outputs

Scripts write generated artifacts under:

- `examples/generated/`
