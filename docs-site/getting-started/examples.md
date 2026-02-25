# Run Examples

The project ships executable examples and cookbook scripts. Treat them as the
source of truth for runnable workflows while the tutorials/docs expand.

## Core entry-point examples

Run from the repo root:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode all
conda run -n astrodyn-core-env python examples/scenario_missions.py --mode all
conda run -n astrodyn-core-env python examples/uncertainty.py
conda run -n astrodyn-core-env python examples/geqoe_propagator.py --mode all
```

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
