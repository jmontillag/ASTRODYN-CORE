# First Propagation

This page shows the shortest **self-contained** propagation workflow you can run
after installing the package with `pip`, plus the repo-based path for deeper examples.

## Goal

Learn the core objects and patterns you will use most often:

- `AstrodynClient`
- `PropagatorSpec`
- `BuildContext`
- (optional later) `IntegratorSpec` for numerical/DSST propagators

## Fastest runnable example (pip-friendly): TLE propagation from two lines

This example is self-contained and does **not** require the repo `examples/`
folder. It uses a TLE propagator because that avoids manually constructing an
Orekit initial orbit for the very first run.

It still requires:

- `orekit` installed (pulled in by `astrodyn-core`)
- Orekit data configured (this snippet assumes `orekit-data.zip` is available in
  the current working directory and uses `setup_orekit_curdir()`)

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
target = tle_epoch.shiftedBy(45.0 * 60.0)  # +45 min
state = propagator.propagate(target)

orbit = state.getOrbit()
pos = state.getPVCoordinates(FramesFactory.getGCRF()).getPosition()

print(f"a = {orbit.getA()/1e3:.2f} km")
print(f"e = {orbit.getE():.6f}")
print(f"i = {math.degrees(orbit.getI()):.3f} deg")
print(f"r = ({pos.getX()/1e3:.1f}, {pos.getY()/1e3:.1f}, {pos.getZ()/1e3:.1f}) km")
```

What this teaches:

- facade-first usage (`AstrodynClient`)
- typed propagator selection (`PropagatorSpec(kind=TLE, ...)`)
- builder context usage (`BuildContext()`)
- propagation to a target epoch and basic orbit/state inspection

## Same workflow from the repo examples (extended version)

If you cloned the repo and want the maintained script version of this flow:

```bash
conda run -n astrodyn-core-env python examples/quickstart.py --mode tle
```

The repo example adds:

- shared helpers (`examples/_common.py`)
- nicer output formatting
- consistent example conventions used across the project

## Next step: numerical propagation (more realistic force models)

After the TLE example, the next learning step is a numerical propagator built
from typed specs and an Orekit initial orbit. That pattern is explained in:

- [Propagation Quickstart](../tutorials/propagation-quickstart.md)

and demonstrated in the repo script:

- `examples/quickstart.py --mode numerical`

## Why this page starts with TLE instead of numerical

Numerical propagation is more representative for many mission-analysis use
cases, but it requires extra setup to construct an Orekit orbit object.

Starting with TLE gives package users a fully self-contained, first-success
example that still exercises the ASTRODYN-CORE propagation facade.

## Next steps

- Read [Orekit data setup](orekit-data.md) if the snippet cannot find data
- Read [Run Examples](examples.md) for repo-based workflows
- Continue to [Propagation Quickstart](../tutorials/propagation-quickstart.md)
