# `data` module

Bundled data and preset resolution helpers.

## Purpose

The `data` module provides stable access to package-shipped assets so users do not hardcode absolute paths.

Primary responsibilities:
- expose propagation model presets (`propagation_models/*.yaml`),
- expose spacecraft presets (`spacecraft_models/*.yaml`),
- provide discovery functions for available bundled models.

## Key API

From `astrodyn_core.data`:
- `list_propagation_models()`
- `get_propagation_model(name)`
- `list_spacecraft_models()`
- `get_spacecraft_model(name)`

Behavior highlights:
- accepts names with or without `.yaml`,
- resolves via `importlib.resources` (package-safe),
- raises clear `FileNotFoundError` with available options.

## Intended use cases

1. **Quick preset discovery for new users**
   - list available models and inspect options.
2. **Bootstrapping propagation specs from YAML**
   - combine with `propagation.load_dynamics_config()` and `propagation.load_spacecraft_config()`.
3. **Reproducible examples/tests**
   - rely on package-relative assets, not local machine paths.

See `examples/quickstart.py` in `basics` and `numerical` modes.

## Boundaries

- This module does **not** validate propagation physics itself.
- It only locates bundled files; parsing/validation occurs in `propagation.config`.
