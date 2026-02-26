# How-To: TLE Propagation from Two Lines (Self-Contained)

This recipe is designed for **package users** who installed `astrodyn-core`
with `pip` and do not have the repo `examples/` folder locally.

It shows how to build and run a TLE propagator using only:

- `orekit`
- `astrodyn_core`
- two TLE lines

## When to use this recipe

Use this when you want a first successful propagation run with minimal setup and
without constructing an Orekit initial orbit manually.

## Prerequisites

- `astrodyn-core` installed
- Orekit data configured

This snippet assumes `orekit-data.zip` is in the current working directory and
uses `setup_orekit_curdir()`. If that is not true in your setup, see:

- [Orekit Data Setup](../getting-started/orekit-data.md)

## Run (copy-paste)

```python
import math

import orekit
orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

from org.orekit.frames import FramesFactory
from org.orekit.propagation.analytical.tle import TLE as OrekitTLE

from astrodyn_core import AstrodynClient, BuildContext, PropagatorKind, PropagatorSpec, TLESpec

app = AstrodynClient()

tle = TLESpec(
    line1="1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9002",
    line2="2 25544  51.6400  10.0000 0006000  50.0000 310.0000 15.49000000000000",
)

propagator = app.propagation.build_propagator(
    PropagatorSpec(kind=PropagatorKind.TLE, tle=tle),
    BuildContext(),
)

tle_epoch = OrekitTLE(tle.line1, tle.line2).getDate()
target = tle_epoch.shiftedBy(45.0 * 60.0)

state = propagator.propagate(target)
orbit = state.getOrbit()
pos = state.getPVCoordinates(FramesFactory.getGCRF()).getPosition()

print(f"a = {orbit.getA()/1e3:.2f} km")
print(f"e = {orbit.getE():.6f}")
print(f"i = {math.degrees(orbit.getI()):.3f} deg")
print(f"r = ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f}) km")
```

## What this demonstrates

- `AstrodynClient` facade usage
- typed TLE configuration via `TLESpec`
- propagator selection via `PropagatorSpec(kind=TLE, ...)`
- propagation to a target epoch
- inspection of orbit elements and Cartesian position

## Extended repo example (same workflow)

If you cloned the repo, the corresponding maintained example is:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode tle
```

That script includes shared helpers and is a good reference for the project’s
example conventions.

## Next steps

- [First Propagation](../getting-started/first-propagation.md) (overview and next-step guidance)
- [Propagation Quickstart](../tutorials/propagation-quickstart.md)
- [Scenario + Mission Workflows](../tutorials/scenario-missions.md)
