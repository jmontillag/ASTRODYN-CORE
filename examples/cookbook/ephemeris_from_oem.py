#!/usr/bin/env python
"""Create a BoundedPropagator from an OEM file using the ephemeris module.

This example demonstrates the ephemeris module's ability to create Orekit
BoundedPropagator objects from standard CCSDS OEM files.  It uses a real
Artemis I Orion post-TLI trajectory file shipped in ``examples/data/``.

Run with:
    conda run -n astrodyn-core-env python examples/cookbook/ephemeris_from_oem.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import init_orekit

init_orekit()

from orekit.pyhelpers import absolutedate_to_datetime

from astrodyn_core import AstrodynClient, EphemerisSpec
from astrodyn_core.ephemeris.parser import parse_oem

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

app = AstrodynClient()
data_dir = Path(__file__).resolve().parent.parent / "data"
oem_path = data_dir / "Post_TLI_Orion_AsFlown_20221213_EPH_OEM.asc"

if not oem_path.exists():
    raise FileNotFoundError(f"Sample OEM file not found: {oem_path}")

print("=" * 72)
print("  Ephemeris Module: OEM Propagator from Artemis I Trajectory")
print("=" * 72)
print(f"  Source file: {oem_path.name}")

# ---------------------------------------------------------------------------
# Step 1: Inspect the OEM file metadata
# ---------------------------------------------------------------------------

oem = parse_oem(oem_path)
satellites = oem.getSatellites()
sat_id = satellites.keySet().iterator().next()
oem_sat = satellites.get(sat_id)
segments = oem_sat.getSegments()

seg0 = segments.get(0)
start_date = seg0.getStart()
stop_date = seg0.getStop()
span_days = stop_date.durationFrom(start_date) / 86400.0

print(f"\n  OEM metadata:")
print(f"    Satellite ID: {sat_id}")
print(f"    Segments:     {segments.size()}")
print(f"    Start:        {absolutedate_to_datetime(start_date)}")
print(f"    Stop:         {absolutedate_to_datetime(stop_date)}")
print(f"    Span:         {span_days:.2f} days")

# ---------------------------------------------------------------------------
# Step 2: Create a BoundedPropagator via the ephemeris module
# ---------------------------------------------------------------------------

spec = EphemerisSpec.for_oem(oem_path)
bounded_prop = app.ephemeris.create_propagator(spec)


from org.orekit.utils import BoundedPVCoordinatesProvider
bounded = BoundedPVCoordinatesProvider.cast_(bounded_prop)
start_date = bounded.getMinDate()
stop_date = bounded.getMaxDate()

init_state = bounded_prop.getInitialState()
print(f"\n  BoundedPropagator created successfully.")
print(f"    Initial epoch: {absolutedate_to_datetime(init_state.getDate())}")

# ---------------------------------------------------------------------------
# Step 3: Sample the propagator at several points across the arc
# ---------------------------------------------------------------------------

from org.orekit.frames import FramesFactory

frame = FramesFactory.getGCRF()

print(f"\n  Sampled positions (GCRF, km):")
print(f"  {'Epoch':^28s}  {'X':>12s}  {'Y':>12s}  {'Z':>12s}")
print(f"  {'-' * 28}  {'-' * 12}  {'-' * 12}  {'-' * 12}")

n_samples = 6
total_span = stop_date.durationFrom(start_date)
step_s = total_span / (n_samples - 1)

for i in range(n_samples):
    # Use exact stop_date for the last sample to avoid floating-point overshoot
    date = stop_date if i == n_samples - 1 else start_date.shiftedBy(i * step_s)
    state = bounded_prop.propagate(date)
    pos = state.getPVCoordinates(frame).getPosition()
    dt_str = str(absolutedate_to_datetime(date))[:26]
    print(
        f"  {dt_str:28s}"
        f"  {pos.getX() / 1e3:+12.3f}"
        f"  {pos.getY() / 1e3:+12.3f}"
        f"  {pos.getZ() / 1e3:+12.3f}"
    )

# ---------------------------------------------------------------------------
# Step 4: Compare start and end positions
# ---------------------------------------------------------------------------

start_state = bounded_prop.propagate(start_date)
end_state = bounded_prop.propagate(stop_date)
start_pos = start_state.getPVCoordinates(frame).getPosition()
end_pos = end_state.getPVCoordinates(frame).getPosition()
dist_km = start_pos.subtract(end_pos).getNorm() / 1e3

print(f"\n  Distance between start and end positions: {dist_km:.1f} km")
print("\nDone.")
