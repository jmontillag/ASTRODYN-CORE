# Run Examples

The project ships executable examples and cookbook scripts. Treat them as the
source of truth for runnable workflows while the tutorials/docs expand.

## Two kinds of examples in the docs ecosystem

- **Docs-embedded examples (self-contained)**: copy-paste snippets in tutorial/how-to pages for `pip` users
- **Repo examples (`examples/`)**: fuller scripts for repo users and contributors

Recommended usage:

- If you installed with `pip`, start with docs-embedded examples first
- If you cloned the repo, use the `examples/` scripts as executable references

The goal is for docs pages to remain useful even when the `examples/` folder is
not available in your installation.

## Core entry-point examples

Run from the repo root:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode all
conda run -n astrodyn-core-env python examples/scenario_missions.py --mode all
conda run -n astrodyn-core-env python examples/uncertainty.py
conda run -n astrodyn-core-env python examples/geqoe_propagator.py --mode all
conda run -n astrodyn-core-env python examples/tle_batch_high_fidelity_ephemeris.py
```

Notes for the batch TLE example:

- Reads NORAD IDs + target epoch from `examples/state_files/tle_norad_batch.yaml`
- Uses Space-Track credentials from repo-root `secrets.ini`
- Writes date-windowed HDF5 outputs under `examples/generated/tle_hf_ephemerides/`

## Cookbook examples

```bash
conda run -n astrodyn-core-env python examples/cookbook/multi_fidelity_comparison.py
conda run -n astrodyn-core-env python examples/cookbook/force_model_sweep.py
conda run -n astrodyn-core-env python examples/cookbook/ephemeris_from_oem.py
```

## Generated outputs

Many examples write outputs to:

- `examples/generated/`

Typical outputs include:

- plots (`.png`)
- state series (`.yaml`, `.h5`)
- enriched scenario/state files

## Choosing an example

- New user: `examples/quickstart.py`
- State-file / mission workflows: `examples/scenario_missions.py`
- Covariance/STM: `examples/uncertainty.py`
- GEqOE: `examples/geqoe_propagator.py`

See `examples/README.md` for mode breakdowns and script-specific notes.

## Documentation growth path (what to expect)

New docs pages are being written with a consistent pattern:

1. **Self-contained example** (works for package users)
2. **Extended repo example** (links to `examples/...`)
3. **API/reference links** (for deeper exploration)

This keeps the docs useful for both package users and repo contributors.
