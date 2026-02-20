# Examples

This folder uses three entry points.

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

## Generated outputs

Scripts write generated artifacts under:

- `examples/generated/`
