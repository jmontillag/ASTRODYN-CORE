#!/usr/bin/env python
"""Build a numerical propagator from a YAML-defined initial state.

Run from repo root:
    python examples/demo_state_file_numerical.py
"""

from __future__ import annotations

from pathlib import Path

import orekit

orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir  # noqa: E402

setup_orekit_curdir()

from org.orekit.frames import FramesFactory  # noqa: E402

from astrodyn_core import (  # noqa: E402
    BuildContext,
    PropagatorFactory,
    ProviderRegistry,
    get_propagation_model,
    get_spacecraft_model,
    load_dynamics_config,
    load_initial_state,
    load_spacecraft_config,
    register_default_orekit_providers,
)

state_path = Path(__file__).resolve().parent / "state_files" / "leo_initial_state.yaml"
state_record = load_initial_state(state_path)

spec        = load_dynamics_config(get_propagation_model("medium_fidelity"))
spacecraft  = load_spacecraft_config(get_spacecraft_model("leo_smallsat"))
spec        = spec.with_spacecraft(spacecraft)

registry = ProviderRegistry()
register_default_orekit_providers(registry)
factory = PropagatorFactory(registry=registry)

ctx = BuildContext.from_state_record(state_record)
builder = factory.build_builder(spec, ctx)
propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

gcrf = FramesFactory.getGCRF()
start_date = ctx.require_initial_orbit().getDate()
state_30min = propagator.propagate(start_date.shiftedBy(1800.0))
pos = state_30min.getPVCoordinates(gcrf).getPosition()

print(f"Loaded state file: {state_path.name}")
print(f"Builder: {builder.getClass().getSimpleName()}")
print(f"Propagator: {propagator.getClass().getSimpleName()}")
print(
    "Position after 30 min (km): "
    f"({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f})"
)
