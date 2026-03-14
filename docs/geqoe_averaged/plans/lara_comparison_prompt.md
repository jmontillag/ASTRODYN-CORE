# Prompt: Implement Lara-Brouwer Analytical Propagator and Compare Against GEqOE

Implement a first-order Brouwer/Lara analytical satellite propagator for J2-J5 zonal
harmonics and run a head-to-head comparison against the existing GEqOE averaged theory.

Read the full corrected plan first:
  docs/geqoe_averaged/plans/lara_comparison_plan.md

Then read the geqoe-averaged skill for codebase context:
  .claude/skills/geqoe-averaged/SKILL.md

## What this is

At first order, ALL classical analytical theories (Brouwer 1959, Lara 2021, Coffey-Deprit
1982) produce identical short-period corrections. We are implementing the standard
first-order Brouwer closed-form theory from scratch in Python, then comparing its
position accuracy against the GEqOE averaged theory across 12 test orbits.

The scientific point: the GEqOE theory uses generalized equinoctial elements (non-singular
at e=0, i=0) and averages in the generalized eccentric longitude K. The Brouwer theory
uses classical Keplerian/Delaunay elements (singular at e=0) and averages in the mean
anomaly. Same physics, different mathematical framework -> different numerical conditioning.

## Existing infrastructure to reuse

All files below are relative to the repo root.

Constants (MUST use these -- import directly, do NOT hardcode):
  src/astrodyn_core/geqoe_taylor/constants.py
  -> MU = 398600.4354360959  (km^3/s^2)
  -> RE = 6378.1366  (km)
  -> J2 = 1.08262617385222e-3
  -> J3 = -2.53265648533224e-6
  -> J4 = -1.61989759991697e-6
  -> J5 = -2.27296082868698e-7

Existing kepler-to-cartesian conversion (reuse as-is):
  docs/geqoe_averaged/geqoe_mean/coordinates.py
  -> kepler_to_rv(a_km, e, inc_deg, raan_deg, argp_deg, M_deg) returns (r_vec, v_vec)
  -> rv_to_classical(r_vec, v_vec) returns (a, e, i_deg) -- only 3 elements

Existing validation script (the template -- study its structure):
  docs/geqoe_averaged/scripts/extended_validation.py
  -> CASES list (12 test orbits at line ~92), OrbitCase dataclass
  -> build_cowell_integrator() + propagate_cowell_grid() -- heyoka Cowell truth
  -> run_brouwer() -- Orekit BrouwerLyddanePropagator (our cross-check target)
  -> run_single_case() -- the main loop that runs all methods
  -> compute_errors() from geqoe_mean.validation

GEqOE mean propagation (already works, produces comparison data):
  docs/geqoe_averaged/geqoe_mean/short_period.py
  docs/geqoe_averaged/geqoe_mean/heyoka_compiled.py
  docs/geqoe_averaged/geqoe_mean/validation.py -> integrate_mean, compute_position_errors

Reference papers (readable with pymupdf in the astrodyn-core-env conda env):
  docs/geqoe_averaged/references/lara2021.pdf -- J2 first-order theory, Eq. 15 = da
  docs/geqoe_averaged/references/lara2024.pdf -- J3-J5 generating functions, Tables A.2-A.10
  docs/geqoe_averaged/references/spacetrack_report3_1980.pdf -- SGP4/Brouwer formulas
  docs/geqoe_averaged/references/vallado2006.pdf -- Brouwer implementation details
  docs/geqoe_averaged/references/brouwer1959.pdf -- original Brouwer paper

To read a PDF page:
  conda run -n astrodyn-core-env python3 -c "import fitz; doc = fitz.open('path.pdf'); print(doc[PAGE_NUM].get_text())"

## Test orbit grid (exact definitions from extended_validation.py)

```python
#  Name         a[km]   e      i[deg]  RAAN  argp  M0   orbits
("LEO-circ",    6878,  0.001, 51.6,    30,   0,    0,   50),
("LEO-mod-e",   7000,  0.05,  40,      25,   60,   90,  50),
("SSO",         7078,  0.001, 97.8,    30,   0,    0,   50),
("Near-equat",  7200,  0.01,  5,       0,    45,   90,  50),
("Crit-low-e",  7500,  0.01,  63.435,  30,   90,   0,   100),
("Crit-mod-e",  12000, 0.15,  63.435,  30,   90,   0,   100),
("Molniya",     26554, 0.74,  63.4,    40,   270,  90,  30),
("GTO",         24500, 0.73,  7,       0,    180,  0,   20),
("MEO-GPS",     26560, 0.01,  55,      30,   0,    0,   50),
("Polar",       7200,  0.005, 90,      30,   0,    0,   50),
("HEO-45",      15000, 0.4,   45,      30,   120,  90,  30),
("Retrograde",  7500,  0.01,  120,     30,   0,    0,   50),
```

64 samples/orbit. RAAN/argp/M0 are in degrees.

## File layout to create

```
docs/geqoe_averaged/
  lara_theory/
    __init__.py
    coordinates.py      # Cartesian<->Keplerian<->Delaunay, solve_kepler, f_from_E
    mean_elements.py    # Secular rates + mean ODE integration
    short_period.py     # First-order SP corrections J2-J5
    propagator.py       # Full pipeline class
  scripts/
    lara_comparison.py  # Comparison harness
```

## Phase-by-phase implementation

### Phase 1: coordinates.py

Implement (all vectorized -- accept scalar or ndarray):

- solve_kepler(M, e) -- Newton-Raphson, tol=1e-15
- eccentric_to_true(E, e) -- use atan2(sqrt(1-e^2)*sinE, cosE - e)
- true_to_eccentric(f, e) -- inverse
- cartesian_to_keplerian(r_vec, v_vec, mu) -- full 6-element: (a, e, i, Omega, omega, M)
  (rv_to_classical in the existing code only returns 3 elements -- you need all 6)
- keplerian_to_cartesian(a, e, i, Omega, omega, f, mu) -- takes true anomaly f
- keplerian_to_delaunay / delaunay_to_keplerian -- trivial wrappers

Validation: round-trip Cartesian -> Keplerian -> Cartesian < 1e-13 relative, tested on
all 12 orbit types. Run: conda run -n astrodyn-core-env pytest -q

### Phase 2: mean_elements.py

J2 secular rates (these are exact -- implement directly):

  gamma2 = J2 * Re^2 / (2*p^2),  p = a*(1-e^2),  eta = sqrt(1-e^2),  c = cos(i),  n = sqrt(mu/a^3)

  l_dot = n * [1 + (3/2)*gamma2*eta*(3*c^2 - 1)]
  g_dot = n * (3/2)*gamma2*(5*c^2 - 1)
  h_dot = -3*n*gamma2*c
  L_dot = G_dot = H_dot = 0  (for J2 secular)

For J3-J5 mean rates: implement numerical one-revolution averaging. For each Jn,
evaluate the disturbing function Rn = (mu/r)*Jn*(Re/r)^n*Pn(sin phi) and its partial
derivatives w.r.t. (l,g,L,G,H) via finite differences, then average over l in [0,2pi]
using 128-point Gauss-Legendre quadrature. This gives the J3-J5 contributions to all
6 Delaunay rates.

Integrate the mean ODE with scipy solve_ivp (DOP853, rtol=1e-12, atol=1e-14).

Validation: propagate mean elements 10 orbits (J2 only), compare omega drift rate
against analytical formula. Agreement < 1e-10 rad/s.

### Phase 3: short_period.py -- THE CORE

The J2 first-order short-period corrections. These come from the generating function
W1 via Poisson brackets: delta_xi = J2 * {xi, W1}.

The da formula from Lara (2021) Eq. (15):

  da = a*(Re/p)^2*(1/(4*eta^2)) * sum over i=0..1 of:
       B_i(s) * sum over j of A_{i,j}(eta) * e^|j-2i| * cos(j*f + 2*i*g)

  where B0 = 1 - 3*s^2/2,  B1 = 3*s^2/4,  s = sin(i),  g = omega

  For i=0 (j=0..3):
    A_{0,0} = 10 - 6*eta^2 - 4*eta^3
    A_{0,1} = 15 - 3*eta^2
    A_{0,2} = 6
    A_{0,3} = 1

  For i=1 (j=-1..5):
    A_{1,-1} = 1
    A_{1,0} = 6
    A_{1,1} = 15 - 3*eta^2
    A_{1,3} = 15 - 3*eta^2
    A_{1,4} = 6
    A_{1,5} = 1
    (A_{1,2} -- read from the paper, page 10)

For the remaining 5 corrections (de, di, dOmega, domega, dM), use one of:

**Approach A (recommended)**: Read the Brouwer formulas from
spacetrack_report3_1980.pdf or vallado2006.pdf using pymupdf. These give all 6
Keplerian corrections in closed form. The Spacetrack Report's initialization section
has them in pseudocode-like notation.

**Approach B**: Read Lara (2021) for the generating function W1 (fully specified on
pages 5-6, Eqs. 6+13), then compute {xi, W1} symbolically with SymPy for each
Delaunay variable, and convert to Keplerian corrections. More reliable, slower to set up.

For J3-J5 short-period corrections: use numerical short-period extraction. At each
evaluation point:
1. Compute the Jn disturbing function at the current (mean) state
2. Subtract its orbit-averaged value (128-pt Gauss-Legendre quadrature)
3. Integrate the residual to get the periodic generating function contribution
4. Evaluate the Poisson bracket numerically

This avoids transcribing ~50 inclination polynomial entries from Lara (2024) Tables
A.2-A.10. It is slower but correct. Once validated, the analytical formulas from
Lara (2024) can be substituted for speed.

Implement:

```python
def short_period_corrections(a, e, i, Om, om, M, mu, Re, j_coeffs):
    """First-order SP corrections evaluated at the given (mean) state.
    Returns (da, de, di, dOm, dom, dM).
    """

def mean_to_osculating(mean_kep, mu, Re, j_coeffs):
    """osc = mean + SP(mean)"""

def osculating_to_mean(osc_kep, mu, Re, j_coeffs):
    """mean = osc - SP(osc)  [first-order inverse]"""
```

### CRITICAL CROSS-CHECK after Phase 3

For each of the 12 test orbits, compute:
  1. Our propagator: osculating IC -> mean (inverse) -> osculating (forward) -> Cartesian
  2. Orekit BrouwerLyddanePropagator: position at t=0

These should agree to within ~1-50 meters (both are first-order Brouwer theories).
If they disagree by > 100 m for LEO cases, there is a bug. Debug before proceeding.

### Phase 4: propagator.py

Assemble the full pipeline:

```python
class LaraBrouwerPropagator:
    def initialize(self, r0, v0, t0, mu, Re, j_coeffs):
        osc_kep = cartesian_to_keplerian(r0, v0, mu)
        mean_kep = osculating_to_mean(osc_kep, mu, Re, j_coeffs)
        # Breakwell-Vagners energy calibration (Lara 2021 Eq. 23):
        # E0 = exact total energy from (r0, v0) including zonal potential
        # H01 = averaged J2 Hamiltonian at mean elements
        #      = -(mu/(2*a_bar)) * (Re/p_bar)^2 * eta_bar * (1 - 3/2*sin^2(i_bar))
        # a_calibrated = mu / (2*(-E0 + J2*H01))
        # Replace a_bar with a_calibrated in mean elements
        # This makes secular mean motion accurate to O(J2^3)

    def propagate(self, t_array):
        # 1. Integrate mean Delaunay ODE -> mean states at each epoch
        # 2. At each epoch: mean_to_osculating -> keplerian_to_cartesian
        # Return (N,3) positions, (N,3) velocities
```

Validation: propagate LEO-circ 10 orbits, compare against Cowell truth. Position RMS
should be ~1-5 km (comparable to Orekit Brouwer).

### Phase 5: lara_comparison.py

Extend extended_validation.py's structure. For each of the 12 CASES:

```python
from geqoe_mean.coordinates import kepler_to_rv
from astrodyn_core.geqoe_taylor.constants import MU, RE, J2, J3, J4, J5

# Build IC
r0, v0 = kepler_to_rv(case.a_km, case.e, case.inc_deg,
                        case.raan_deg, case.argp_deg, case.M0_deg)

# Truth: heyoka Cowell
ta = build_cowell_integrator(pert, r0, v0, tol=1e-15)
truth_cart = propagate_cowell_grid(ta, t_grid)

# GEqOE mean+SP (reuse existing)
geqoe_cart, _ = run_geqoe_meansp(case, r0, v0, pert, t_grid)

# Lara-Brouwer (NEW)
prop = LaraBrouwerPropagator()
prop.initialize(r0, v0, 0.0, MU, RE, {2: J2, 3: J3, 4: J4, 5: J5})
lara_cart, _ = prop.propagate(t_grid)

# Orekit Brouwer (cross-check)
brouwer_cart = run_brouwer(case, t_grid, orekit_orbit, epoch, frame)

# Errors
err_geqoe  = compute_errors(truth_cart, geqoe_cart, "GEqOE")
err_lara   = compute_errors(truth_cart, lara_cart, "Lara-Brouwer")
err_brouwer = compute_errors(truth_cart, brouwer_cart, "Orekit-Brouwer")

# Sanity check: Lara vs Orekit-Brouwer should agree within ~50m
lara_vs_brouwer = np.sqrt(np.mean(np.sum((lara_cart - brouwer_cart)**2, axis=1)))
```

Print a summary table:
```
Case         | GEqOE pos_rms | Lara pos_rms | Brouwer pos_rms | Lara-vs-Brouwer
LEO-circ     |   1.27 km     |   ???  km    |   10.6 km       |   ??? m
...
```

Save results to docs/geqoe_averaged/lara_comparison_results.json.

### Phase 6: Commit

After all 12 cases run successfully and the Lara-vs-Brouwer sanity check passes
(< 100 m for most cases), commit everything.

## Key warnings

1. CONSTANTS: Import from constants.py. Do NOT hardcode. A 6th-digit difference in J2
   invalidates the comparison at the theory's truncation error level.

2. OREKIT CROSS-CHECK: The Orekit Brouwer propagator is your ground truth for "is my
   Brouwer implementation correct." Run it at every phase. If you disagree by > 100 m,
   you have a bug.

3. NEAR-CIRCULAR ORBITS: The Keplerian omega is undefined at e=0. Use atan2-based
   eccentricity vector components to avoid NaN. For e < 1e-6, set omega=0. This
   limitation (vs GEqOE's singularity-free equinoctial elements) is part of the
   scientific comparison.

4. DO NOT derive formulas from scratch. Transcribe from papers or compute with SymPy.

5. DO NOT optimize before validating. Pure NumPy first.

6. DO NOT implement second-order corrections. First-order vs first-order only.

7. Run tests after each phase: conda run -n astrodyn-core-env pytest -q
   Expect 371 existing tests to keep passing (you are only adding new files).

8. The lara_theory/ package lives under docs/geqoe_averaged/ (same level as geqoe_mean/).
   Scripts run from docs/geqoe_averaged/scripts/ with sys.path including the parent dir.

9. For the Orekit Brouwer cross-check, reuse the existing run_brouwer() function from
   extended_validation.py. It already handles the Orekit initialization, constant
   matching, and coordinate conversion.

10. The comparison script should be runnable standalone:
    cd docs/geqoe_averaged && conda run -n astrodyn-core-env python scripts/lara_comparison.py
