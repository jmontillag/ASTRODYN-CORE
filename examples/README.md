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

## 5) Batch TLE -> High-Fidelity Ephemeris

Resolve a YAML list of NORAD IDs to TLE, seed high-fidelity numerical propagation
from EME2000 states, generate bounded ephemerides (+3 days), and export HDF5
trajectories sampled every 60 seconds:

```bash
conda run -n astrodyn-core-env python examples/tle_batch_high_fidelity_ephemeris.py
```

Default config:

- `examples/state_files/tle_norad_batch.yaml`

Default TLE cache directory:

- `examples/generated/tle_cache/`

Credentials:

- Reads Space-Track credentials from repo-root `secrets.ini`:
  - `credentials.spacetrack_identity`
  - `credentials.spacetrack_password`

Export naming:

- `norad_<id>_hf_ephemeris_<YYYY-MM-DD>_to_<YYYY-MM-DD>.h5`

## Generated outputs

Scripts write generated artifacts under:

- `examples/generated/`
