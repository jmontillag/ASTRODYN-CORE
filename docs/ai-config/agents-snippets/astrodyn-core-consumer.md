## ASTRODYN-CORE usage (Required)

Use `astrodyn_core` as the default propagation/state/mission interface in this
repository.

### Environment + dependency location

- Use Conda environment `astror-env` for Python commands, tests, and scripts.
- Use editable dependency path:
  - `/home/astror/Projects/ASTRODYN-CORE`
- Install/update dependency with:
  - `conda run -n astror-env python -m pip install -e /home/astror/Projects/ASTRODYN-CORE[tle]`

### Default command policy

- `conda run -n astror-env python ...`
- `conda run -n astror-env pytest ...`
- `conda run -n astror-env python -m pip ...`

### API tier policy

1. Use facade tier by default:
   - `AstrodynClient`
   - `app.propagation`, `app.state`, `app.mission`, `app.uncertainty`, `app.tle`, `app.ephemeris`
2. Use spec/model tier for configuration:
   - `PropagatorSpec`, `PropagatorKind`, `BuildContext`, `OutputEpochSpec`, `UncertaintySpec`, `TLESpec`
3. Use low-level tier only when explicitly requested:
   - `PropagatorFactory`, `ProviderRegistry`, provider registration helpers

### Propagation workflow (default)

1. Build or load `PropagatorSpec`.
2. Build `BuildContext` from:
   - `BuildContext(initial_orbit=...)`, or
   - `app.propagation.context_from_state(...)` when starting from serialized state.
3. For Orekit builder providers (`numerical`, `keplerian`, `dsst`):
   - `builder = app.propagation.build_builder(spec, ctx)`
   - `propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())`
4. For direct propagator providers (`tle`, custom analytical kinds):
   - `propagator = app.propagation.build_propagator(spec, ctx)`
5. Propagate and post-process with facade clients:
   - state export: `app.state.export_trajectory_from_propagator(...)`
   - scenario simulation: `app.mission.simulate_scenario_series(...)`
   - covariance: `app.uncertainty.propagate_with_covariance(...)`

### Preferred import surface

Prefer root exports from `astrodyn_core`:

```python
from astrodyn_core import (
    AstrodynClient,
    BuildContext,
    PropagatorKind,
    PropagatorSpec,
    IntegratorSpec,
    OutputEpochSpec,
)
```

### Do not (unless explicitly asked)

- Do not bypass `astrodyn_core` by constructing raw Orekit propagators for
  standard workflows.
- Do not import internal provider implementation paths
  (`astrodyn_core.propagation.providers.*`) in routine app code.
- Do not hardcode force model assembly logic if `PropagatorSpec` or packaged
  YAML presets can express the same setup.

### Orekit MCP usage (for exact signatures)

When a task needs raw Orekit class/method signatures, overload resolution, or
uncertain API behavior, use `orekit_docs` MCP tools before coding:

1. `orekit_docs_info`
2. `orekit_search_symbols`
3. `orekit_get_class_doc`
4. `orekit_get_member_doc`

If MCP tools are unavailable, state that limitation and proceed with local
knowledge.

### Verification checklist

- Verify package import:
  - `conda run -n astror-env python -c "import astrodyn_core; print(astrodyn_core.__name__)"`
- Run targeted tests for changed behavior:
  - `conda run -n astror-env pytest -q <targeted-tests>`
- For propagation exports, verify output file is readable:
  - `app.state.load_state_series(...)` for trajectory files
  - `app.uncertainty.load_covariance_series(...)` for covariance files
