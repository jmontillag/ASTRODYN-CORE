# Corrected Plan: Lara First-Order Analytical Theory for J₂–J₅ Comparison

## Purpose

Implement a Python version of the first-order Brouwer/Lara closed-form analytical
satellite theory for J₂–J₅ zonal harmonics and compare it head-to-head against the
GEqOE first-order averaged theory. The comparison isolates **element set and
short-period extraction method** — everything else (gravity model, constants, truth
reference, test orbits, output epochs) is held identical.

---

## Critical Corrections to the Original Plan

The original plan had several errors identified by cross-referencing the actual papers:

1. **Lara (2021) is J₂-only.** The paper "Brouwer's Satellite Solution Redux" treats
   only the main problem (J₂) via a single Lie transformation. It does NOT contain
   J₃–J₅ formulas. The original plan incorrectly treated it as the primary reference
   for J₂–J₅.

2. **Lara (2024) is the essential reference for J₂–J₅.** The paper by Lara, Fantino,
   Susanto, Flores provides the three-stage generating functions (parallax + Delaunay
   normalization + long-period) with explicit J₃–J₅ terms in Tables A.2–A.10.
   The original plan listed this as "optional optimization" — it is in fact required.

3. **No Poincaré non-singular formulas exist in Lara (2021).** The paper works
   entirely in Delaunay variables. Lara (2025) demonstrates reformulation into
   semi-equinoctial variables (χ, ζ, λ), but only for J₂. The plan's Section 4.4
   about "Poincaré non-singular" corrections from "Lara (2021) Section X" is
   fabricated.

4. **At first order, all approaches are equivalent.** Lara (2021) p.2: "To the first
   order, the construction of Brouwer's closed-form solution by means of a single
   transformation amounts to the sum of the two transformations computed by
   Brouwer." This means Brouwer = Lara = Coffey-Deprit at first order. The elaborate
   three-stage discussion was irrelevant.

5. **Constants were wrong.** The plan listed μ = 398600.4418, Rₑ = 6378.137.
   The actual GEqOE constants (from `src/astrodyn_core/geqoe_taylor/constants.py`)
   are μ = 398600.4354360959, Rₑ = 6378.1366.

---

## Reference Papers

| Ref | Paper | Role | Content |
|-----|-------|------|---------|
| **[L24]** | Lara, Fantino, Susanto, Flores (2024). "Higher-order composition of short- and long-period effects." *CNSNS.* | **PRIMARY** | Complete J₂–J₅ first-order theory. Generating functions W^P (parallax, Eqs. 12–13), W^D (Delaunay, Eq. 18) with explicit tables (A.2–A.10). Composed single-transformation form. |
| **[L21]** | Lara (2021). "Brouwer's satellite solution redux." *CMDA* 133, art. 47. | Secondary | J₂-only single-transformation theory. First- and second-order corrections in Delaunay. Breakwell-Vagners energy calibration (Eq. 23) — **essential for initialization.** |
| **[L20]** | Lara (2020). "Solution to the main problem by reverse normalization." *Nonlinear Dynamics* 101. | Background | Alternative two-step approach (long-period first). J₂-only, up to sixth order. |
| **[L25]** | Lara (2025). "Purely periodic second-order terms in closed form and arbitrary variables." *J. Astronaut. Sci.* 72:35. | Background | Shows variable reformulation from Delaunay to semi-equinoctial. Only J₂. |
| **[D81]** | Deprit (1981). "Elimination of the parallax." *Celest. Mech.* 24. | Supporting | Original parallax elimination in polar-nodal variables. |

All five are available in `docs/geqoe_averaged/references/`.

Note: Coffey & Deprit (1982) "Third-order solution" is NOT in the repo and is not
needed — [L24] subsumes it.

---

## Constants (MUST match exactly)

From `src/astrodyn_core/geqoe_taylor/constants.py`:

```python
MU = 398600.4354360959        # km³/s²  (Baù et al. 2021)
RE = 6378.1366                # km
J2 = 1.08262617385222e-3      # EGM2008, unnormalized
J3 = -2.53265648533224e-6
J4 = -1.61989759991697e-6
J5 = -2.27296082868698e-7
```

Import these directly: `from astrodyn_core.geqoe_taylor.constants import MU, RE, J2, J3, J4, J5`.

---

## Architecture

```
docs/geqoe_averaged/
├── lara_theory/
│   ├── __init__.py
│   ├── coordinates.py        # Keplerian ↔ Cartesian ↔ Delaunay conversions
│   ├── mean_elements.py      # Secular rates + mean-element ODE
│   ├── short_period.py       # First-order short-period corrections (THE core)
│   └── propagator.py         # Full pipeline: init → propagate → reconstruct
├── scripts/
│   └── lara_comparison.py    # Comparison harness (extends extended_validation.py)
```

---

## Layer 1: Coordinate Conversions (`coordinates.py`)

### 1.1 Variable Sets Needed

Only **two** variable sets are needed for a first-order implementation in Keplerian
elements:

1. **Keplerian elements**: (a, e, i, Ω, ω, M)
2. **Delaunay variables**: (ℓ, g, h, L, G, H) where ℓ=M, g=ω, h=Ω,
   L=√(μa), G=L√(1−e²), H=G cos i

Poincaré variables are NOT needed. At first order in Delaunay/Keplerian variables
the corrections are well-defined for e > ~1e-4 and i not near 0 or π. For
near-circular orbits (e < 1e-4), the comparison results will show the coordinate
singularity effects — this is part of what we're comparing.

### 1.2 Conversions to Implement

```python
def cartesian_to_keplerian(r_vec, v_vec, mu):
    """Returns (a, e, i, Omega, omega, M). Standard algorithm (Vallado Ch. 4)."""

def keplerian_to_cartesian(a, e, i, Omega, omega, f, mu):
    """Returns (r_vec, v_vec). Takes true anomaly f, not M."""

def keplerian_to_delaunay(a, e, i, Omega, omega, M, mu):
    """Returns (ell, g, h, L, G, H)."""
    # Trivial: ell=M, g=omega, h=Omega, L=sqrt(mu*a), G=L*sqrt(1-e²), H=G*cos(i)

def delaunay_to_keplerian(ell, g, h, L, G, H, mu):
    """Returns (a, e, i, Omega, omega, M)."""
    # Inverse of above. Use atan2 for inclination.

def solve_kepler(M, e, tol=1e-15, max_iter=50):
    """Solve M = E - e sin E. Newton-Raphson with standard starter."""

def eccentric_to_true(E, e):
    """True anomaly from eccentric anomaly."""
    # tan(f/2) = sqrt((1+e)/(1-e)) * tan(E/2)
```

**Vectorized versions** (operating on arrays) should be provided from the start, since
the comparison runs over time grids of ~3200 points.

### 1.3 Validation

Round-trip test: Cartesian → Keplerian → Delaunay → Keplerian → Cartesian.
Relative error < 1e-13 for all non-degenerate test orbits.

---

## Layer 2: Mean-Element Propagation (`mean_elements.py`)

### 2.1 Secular Rates

At first order in J₂ (with J₃–J₅ at first order), the mean Delaunay elements evolve:

- L̄˙ = 0 (energy conserved)
- Ḡ˙ = long-period from odd zonals J₃, J₅ (depends on ḡ)
- H̄˙ = 0 (axisymmetry)
- ℓ̄˙ = n̄ + secular J₂ rate + long-period rates from J₃–J₅
- ḡ˙ = secular J₂ rate + long-period rates from J₃–J₅
- h̄˙ = secular J₂ rate + long-period rates from J₃–J₅

**Pure J₂ secular rates** (standard, e.g. Vallado Eq. 9-41):

```python
def secular_rates_j2(L, G, H, mu, Re, J2):
    a = L**2 / mu
    e = sqrt(1 - (G/L)**2)
    eta = G / L  # = sqrt(1-e²)
    p = a * eta**2
    cos_i = H / G
    n = sqrt(mu / a**3)
    gamma2 = J2 * Re**2 / (2 * p**2)

    dl_dt = n * (1 + 1.5 * gamma2 * eta * (3*cos_i**2 - 1))
    dg_dt = n * 1.5 * gamma2 * (5*cos_i**2 - 1)
    dh_dt = -n * 3.0 * gamma2 * cos_i

    return dl_dt, dg_dt, dh_dt, 0.0, 0.0, 0.0
```

**J₃–J₅ mean rates** are functions of ḡ (argument of perigee) and involve harmonics
of ω. At first order these come from averaging the disturbing function
R_n = (μ/r)(Rₑ/r)ⁿ Jₙ Pₙ(sin φ) over the mean anomaly ℓ. For the first
implementation, compute these via **numerical one-revolution averaging** (integrate
R_n over ℓ = [0, 2π] using 128-point Gauss-Legendre quadrature). This avoids
transcription errors in the long explicit expressions and is exact to machine precision.

Later optimization: replace with explicit closed-form expressions from [L24]
Hamiltonian terms (Eq. 19, Table A.4).

### 2.2 Mean-Element ODE

```python
def mean_element_rhs(t, y, mu, Re, j_coeffs):
    """RHS of mean Delaunay ODE.
    y = [ell, g, h, L, G, H]
    j_coeffs = {2: J2, 3: J3, 4: J4, 5: J5}
    """
    ell, g, h, L, G, H = y
    rates = secular_rates_j2(L, G, H, mu, Re, j_coeffs[2])
    # Add J3-J5 numerically averaged rates
    for n in [3, 4, 5]:
        rates_n = numerically_averaged_rates(L, G, H, g, mu, Re, j_coeffs[n], n)
        rates = [r + rn for r, rn in zip(rates, rates_n)]
    return rates
```

Integrate with `scipy.integrate.solve_ivp(method='DOP853', rtol=1e-12, atol=1e-14)`.

### 2.3 Breakwell-Vagners Energy Calibration

This is critical for a fair comparison. From [L21] Eq. (23):

After computing mean elements from osculating via the first-order inverse map
(Section 4.3), the mean semi-major axis ā (equivalently L̄) is **calibrated** using
the exact energy conservation:

```
E₀ = H(ℓ₀, g₀, h₀, L₀, G₀, H₀)     # exact osculating energy

L̂ = μ / √(2) * 1/√(-E₀ + Σ_{m=1}^{k} (J₂ᵐ/m!) H₀,ₘ(L', G', H'))^{1/2}
```

where H₀,₁ = -μ/(2a') · (Rₑ/p')² · η' · (1 - 3/2 s'²) is the J₂ averaged
Hamiltonian evaluated at mean elements.

This replaces L' with L̂ in the secular frequency computation, effectively giving
third-order-accurate secular terms with only first-order periodic corrections.

---

## Layer 3: Short-Period Corrections (`short_period.py`) — THE CORE

### 3.1 Theory

At first order, the short-period corrections are obtained from the Lie-transform
generating function via Poisson brackets: δξ = {ξ, W₁}. Since Lara (2021) proved
that the single-transformation W₁ is the sum of the three generating functions at
first order, the corrections are **identical** to classical Brouwer's first-order theory.

The corrections are expressed as trigonometric polynomials in the mean true anomaly
f̄ and mean argument of perigee ω̄, with coefficients that are closed-form functions
of the mean eccentricity ē and inclination ī.

### 3.2 J₂ Short-Period Corrections

These are the classical Brouwer first-order corrections. The generating function is
([L21] Eq. 6 without C₁; equivalently [L24] Eq. 12 for the parallax part plus
Eq. 17–18 for the short-period Delaunay part):

```
W₁ = -(G Rₑ²)/(2p²) [B₀ φ + Σ_{i=0}^{1} B_i Σ_j (2-j*)^i / j · e^|j-2i| sin(jf + 2ig)]
```

where B₀ = 1 - 3s²/2, B₁ = 3s²/4, φ = f - ℓ (equation of center), s = sin i.

The Poisson bracket δξ = {ξ, W₁} for each Keplerian element produces explicit
formulas. For the semi-major axis, [L21] Eq. (15) gives:

```
Δa = a(Rₑ/p)² · (1/4η²) Σ_{i=0}^{1} B_i(s) Σ_j A_{i,j}(η) e^|j-2i| cos(jf + 2ig)
```

with explicit A_{i,j} coefficients listed after Eq. (15).

**Implementation strategy**: Transcribe the explicit first-order corrections from
standard references. The cleanest source is Vallado (2013) Chapter 9, which gives
all six element corrections (Δa, Δe, Δi, ΔΩ, Δω, ΔM) from Brouwer's theory.
Cross-check against [L21] Eq. (15) for Δa.

The corrections involve:
- sin(kf + 2mω), cos(kf + 2mω) for k = 0..3, m = 0..1
- φ = f - ℓ (equation of center, only in Δω and ΔM)
- Powers of e, η = √(1-e²)
- Inclination polynomials in s = sin i, c = cos i

```python
def short_period_j2(a_bar, e_bar, i_bar, omega_bar, M_bar, mu, Re, J2):
    """First-order J₂ short-period corrections.

    All inputs are MEAN elements. Returns (da, de, di, dOmega, domega, dM).
    Formulas from Brouwer (1959) / Lara (2021) Eq. 15 structure.
    """
    E_bar = solve_kepler(M_bar, e_bar)
    f_bar = eccentric_to_true(E_bar, e_bar)

    eta = sqrt(1 - e_bar**2)
    p = a_bar * (1 - e_bar**2)
    s = sin(i_bar)
    c = cos(i_bar)
    theta = omega_bar + f_bar   # argument of latitude
    gamma = J2 * Re**2 / (2 * p**2)

    # da: Eq. (15) of [L21]
    # de, di, dOmega, domega, dM: standard Brouwer formulas
    # (see Vallado Ch. 9 or Lyddane 1963)
    ...
```

### 3.3 J₃–J₅ Short-Period Corrections

From [L24], the J₃–J₅ contributions enter through the generating functions
W²_P (parallax, Eq. 13) and W²_D (Delaunay normalization, part of Eq. 18).

The parallax generating function for J₃–J₅ is ([L24] Eq. 13):

```
W^P₂ = -G (Rₑ³/p³) J̃₃ Σ_{i,j,k} e²ᵏ s^{2i+1} e^|j-2i-1| Q_{i,j,k} cos(jf + (2i+1)ω)
      + G (Rₑ⁴/p⁴) Σ_{i,j,k} e²ᵏ s^{2i} e^|j-2i| P_{i,j,k} sin(jf + 2iω)
```

where J̃ₙ = Jₙ/J₂² and the inclination polynomials Q_{i,j,k} and P_{i,j,k} are
in Tables A.2 and A.3.

The Delaunay normalization part for J₃–J₅ involves similar structure with
inclination polynomials from Table A.8–A.10.

**Implementation strategy for J₃–J₅**:

**Option A (recommended for first version)**: Numerical short-period extraction.
For each output epoch, compute the short-period correction as the difference between
the osculating disturbing function and its orbit-averaged value. This is done by:

1. Evaluate the J₃–J₅ part of the disturbing function at the current (mean) state
2. Subtract its mean value (computed by numerical quadrature over one orbit)
3. Integrate the residual over the true anomaly to get the generating function
4. Evaluate Poisson brackets numerically

This avoids transcribing the ~50 inclination polynomial entries from Tables A.2–A.10.

**Option B (optimized)**: Transcribe the explicit formulas from [L24]. Compute
{ξ, W²_P + W²_D} using the explicit polynomial tables. This is faster but
error-prone.

**Option C (hybrid)**: Use Brouwer's known explicit J₃–J₅ first-order corrections
from standard references (e.g., Hoots & Roehrich 1980 / Spacetrack Report #3,
which gives J₃–J₅ corrections used in SGP4). These are the same formulas at
first order.

**Recommended**: Start with Option C — the Brouwer-Lyddane J₃–J₅ short-period
corrections are well-documented in Vallado (2013) Chapter 9 and Hoots & Roehrich
(1980) Spacetrack Report #3 (available in `references/spacetrack_report3_1980.pdf`
and `references/vallado2006.pdf`). These are exactly what Orekit's
BrouwerLyddanePropagator implements, so cross-validation is easy.

### 3.4 Handling the Equation of Center

The short-period corrections for ΔM and Δω contain the equation of center
φ = f - ℓ. This is NOT a problem for evaluation (just compute f - M at the
mean state). It IS a problem for symbolic manipulation at higher orders, but at
first order it's trivial.

### 3.5 Forward and Inverse Maps

```python
def mean_to_osculating(mean_kep, mu, Re, j_coeffs):
    """mean Keplerian → osculating Keplerian (first-order forward map)."""
    a, e, i, Om, om, M = mean_kep
    da, de, di, dOm, dom, dM = short_period_corrections(a, e, i, Om, om, M,
                                                          mu, Re, j_coeffs)
    return (a+da, e+de, i+di, Om+dOm, om+dom, M+dM)

def osculating_to_mean(osc_kep, mu, Re, j_coeffs):
    """osculating Keplerian → mean Keplerian (first-order inverse map).

    At first order: mean ≈ osc - SP(osc). The O(ε²) error is the same order
    as the theory's truncation error. This is identical to what GEqOE does
    (paper Eqs. 66-67).
    """
    a, e, i, Om, om, M = osc_kep
    da, de, di, dOm, dom, dM = short_period_corrections(a, e, i, Om, om, M,
                                                          mu, Re, j_coeffs)
    return (a-da, e-de, i-di, Om-dOm, om-dom, M-dM)
```

---

## Layer 4: Full Propagation Pipeline (`propagator.py`)

```python
class LaraAnalyticalPropagator:
    def __init__(self, mu, Re, j_coeffs):
        self.mu = mu
        self.Re = Re
        self.j_coeffs = j_coeffs  # {2: J2, 3: J3, 4: J4, 5: J5}

    def initialize(self, r0_vec, v0_vec, t0):
        """Initialize from osculating Cartesian state."""
        # 1. Cartesian → osculating Keplerian
        self.osc_kep_0 = cartesian_to_keplerian(r0_vec, v0_vec, self.mu)

        # 2. Osculating → mean (first-order inverse map)
        self.mean_kep_0 = osculating_to_mean(self.osc_kep_0, self.mu,
                                              self.Re, self.j_coeffs)

        # 3. Breakwell-Vagners energy calibration
        self._calibrate_energy(r0_vec, v0_vec)

        self.t0 = t0

    def _calibrate_energy(self, r0, v0):
        """Breakwell-Vagners calibration of L̄ from energy conservation.

        Replaces ā (from first-order inverse map) with calibrated â that makes
        the secular mean motion accurate to O(J₂³). From [L21] Eq. (23).
        """
        r = np.linalg.norm(r0)
        v = np.linalg.norm(v0)
        E0 = 0.5 * v**2 - self.mu / r  # exact osculating energy

        # Add zonal potential at initial position
        # sin(phi) = z/r, where z is the Cartesian z-coordinate
        sin_phi = r0[2] / r
        for n, Jn in self.j_coeffs.items():
            # U_n = (mu/r) * Jn * (Re/r)^n * Pn(sin_phi)
            from numpy.polynomial.legendre import legval
            Pn = legval(sin_phi, [0]*n + [1])  # P_n(sin_phi)
            E0 += (self.mu / r) * Jn * (self.Re / r)**n * Pn

        # H_{0,1} = averaged J₂ Hamiltonian at mean elements
        a_bar, e_bar, i_bar = self.mean_kep_0[:3]
        eta_bar = np.sqrt(1 - e_bar**2)
        p_bar = a_bar * (1 - e_bar**2)
        s_bar = np.sin(i_bar)
        H01 = (-self.mu / (2*a_bar)) * (self.Re/p_bar)**2 * eta_bar * (
            1 - 1.5 * s_bar**2)

        # Calibrated semi-major axis
        J2 = self.j_coeffs[2]
        a_cal = self.mu / (2 * (-E0 + J2 * H01))  # simplified for first order
        # More precisely: solve -mu/(2*L_hat²) + J2*H01 = E0 for L_hat
        L_hat = np.sqrt(self.mu * a_cal)
        self.mean_kep_0 = (a_cal, *self.mean_kep_0[1:])

    def propagate(self, t_array):
        """Propagate to array of times. Returns (N,3) positions, (N,3) velocities."""
        # 1. Propagate mean elements
        mean_states = self._propagate_mean(t_array)

        # 2. Mean → osculating → Cartesian at each epoch
        positions = np.empty((len(t_array), 3))
        velocities = np.empty((len(t_array), 3))

        for idx, (t, mean_kep) in enumerate(zip(t_array, mean_states)):
            osc_kep = mean_to_osculating(mean_kep, self.mu, self.Re, self.j_coeffs)
            a, e, i, Om, om, M = osc_kep
            E = solve_kepler(M, e)
            f = eccentric_to_true(E, e)
            r, v = keplerian_to_cartesian(a, e, i, Om, om, f, self.mu)
            positions[idx] = r
            velocities[idx] = v

        return positions, velocities

    def _propagate_mean(self, t_array):
        """Integrate mean Delaunay ODE, return Keplerian elements at each t."""
        ...
```

**Vectorization**: The `propagate` method should be vectorized (batch Kepler solve,
batch kep→cart) from the start, since the comparison runs N ≈ 3200 epochs.

---

## Layer 5: Comparison Harness (`scripts/lara_comparison.py`)

### 5.1 Reuse Existing Infrastructure

The comparison script should extend `extended_validation.py`. The key pieces to
reuse:

- `CASES` list (12 test orbits, exact definitions)
- `build_cowell_integrator()` + `propagate_cowell_grid()` (truth reference)
- `run_brouwer()` (Orekit Brouwer-Lyddane — cross-validation target)
- `compute_errors()` (position error metrics)
- `_build_orekit_orbit()` (orbit setup helper)

### 5.2 Test Orbit Grid (exact from `extended_validation.py`)

```python
CASES = [
    OrbitCase("LEO-circ",   6878,  0.001, 51.6,   30, 0,   0,   50),
    OrbitCase("LEO-mod-e",  7000,  0.05,  40,     25, 60,  90,  50),
    OrbitCase("SSO",        7078,  0.001, 97.8,   30, 0,   0,   50),
    OrbitCase("Near-equat", 7200,  0.01,  5,      0,  45,  90,  50),
    OrbitCase("Crit-low-e", 7500,  0.01,  63.435, 30, 90,  0,  100),
    OrbitCase("Crit-mod-e",12000,  0.15,  63.435, 30, 90,  0,  100),
    OrbitCase("Molniya",   26554,  0.74,  63.4,   40, 270, 90,  30),
    OrbitCase("GTO",       24500,  0.73,  7,      0,  180, 0,   20),
    OrbitCase("MEO-GPS",   26560,  0.01,  55,     30, 0,   0,   50),
    OrbitCase("Polar",      7200,  0.005, 90,     30, 0,   0,   50),
    OrbitCase("HEO-45",    15000,  0.4,   45,     30, 120, 90,  30),
    OrbitCase("Retrograde", 7500,  0.01,  120,    30, 0,   0,   50),
]
```

RAAN, argp, M0 are in degrees. 64 samples/orbit.

### 5.3 Comparison Flow

For each test case:

```
1. Build Cartesian IC from Keplerian elements
2. Cowell truth (heyoka, tol=1e-15) → truth_cart [N,3]
3. GEqOE mean+SP → geqoe_cart [N,3]  (already cached in extended results)
4. GEqOE equinoctial → eqnoc_cart [N,3]  (already cached)
5. Lara propagator → lara_cart [N,3]  (NEW)
6. Orekit Brouwer → brouwer_cart [N,3]  (cross-check for Lara)
7. Compute errors for all against Cowell truth
```

### 5.4 Cross-Validation: Lara vs Orekit Brouwer

Since Orekit's BrouwerLyddanePropagator implements essentially the same first-order
Brouwer theory, Lara and Orekit-Brouwer should agree to within O(ε²) ≈ O(J₂²)
≈ 10⁻⁶ relative. If they disagree by more, there's a bug. Test this explicitly:

```python
# For each case:
lara_vs_brouwer_rms = rms(lara_cart - brouwer_cart)
assert lara_vs_brouwer_rms < 0.01  # Should agree to ~1-10 m
```

Note: exact agreement is not expected because Orekit Brouwer uses slightly different
internal algorithms (Lyddane's non-singular form for near-circular cases). The
agreement test is a sanity check, not an exact match.

### 5.5 Output Format

Extend the existing results JSON with a `lara` entry:

```json
{
    "lara_time": 0.05,
    "lara": {
        "label": "Lara-Brouwer",
        "pos_rms_km": 1.27,
        "pos_max_km": 2.55,
        "rad_rms_km": 0.75,
        "rad_max_km": 1.80
    }
}
```

Generate a comparison table with columns:
```
Case | GEqOE polar | GEqOE eqnoc | Lara-Brouwer | Orekit-Brouwer | DSST
```

---

## Validation Phases (STRICT ORDER)

### Phase 1: Coordinate Round-Trips (Day 1)
- Cartesian ↔ Keplerian ↔ Delaunay round-trip < 1e-13 relative
- Test all 12 orbit types (circular, eccentric, equatorial, polar, retrograde)
- Kepler equation solver: verify against known solutions

### Phase 2: J₂-Only Secular Rates (Day 1)
- Compare analytical J₂ secular rates (ℓ̇, ġ, ḣ) against Vallado Table 9-1
- Propagate mean elements 100 orbits (J₂ only), compare mean-element drift
  against Cowell orbit-averaged drift

### Phase 3: J₂ Short-Period Round-Trip (Day 2)
- osc → mean → osc for single epoch
- Position round-trip error should be O(J₂²) ≈ few meters for LEO
- Cross-check: our Δa against [L21] Eq. (15) at a test point

### Phase 4: J₂-Only Full Propagation vs Cowell (Day 2)
- 10-orbit propagation, J₂ only
- Position RMS against Cowell truth
- Cross-check against Orekit Brouwer (J₂-only mode if available)

### Phase 5: J₂–J₅ Full Propagation vs Cowell (Day 3)
- Activate J₃–J₅ short-period corrections
- Compare all 12 test orbits against Cowell truth
- Errors should be in the same ballpark as GEqOE theory (Table 5 in paper)

### Phase 6: Head-to-Head Comparison (Day 3)
- Run GEqOE + Lara + Orekit-Brouwer + DSST on all 12 cases
- Generate comparison table and figures
- Record wall-clock times

---

## Technical Notes

### The Critical Inclination (i ≈ 63.435°)

Both theories should handle this identically at first order. The critical inclination
is a physical resonance where ω̇ → 0, not a coordinate singularity. The
(5cos²i - 1) divisor that appears in Brouwer's long-period corrections does NOT
appear in the first-order short-period corrections (it appears in the generating
function C₁ that eliminates long-period terms, but C₁ itself doesn't enter the
first-order corrections).

Verify: no denominator in the J₂ short-period formulas vanishes at i = 63.435°.

### Near-Circular Orbits (e < 0.01)

The Delaunay variable g = ω is undefined for circular orbits. This means:
- The short-period corrections involving sin(f + 2ω) etc. are numerically sensitive
- The osculating-to-mean inversion is ill-conditioned for ω

This is an **expected limitation** of the Keplerian/Delaunay formulation. The GEqOE
theory avoids this via its equinoctial element set. **Documenting this contrast is one
of the scientific points of the comparison.**

For the implementation: use `ω = atan2(e sin ω, e cos ω)` computed from the
eccentricity vector components to avoid the e→0 singularity. This is essentially
Lyddane's trick.

### Sign Conventions

- J₂ > 0, J₃ < 0, J₄ < 0, J₅ < 0 (our constants.py)
- Orekit uses C₂₀ = -J₂, C₃₀ = -J₃, etc. (already handled in `run_brouwer()`)
- The Lie transform convention: osculating = mean + J₂·{mean, W₁} (direct)
- The Brouwer convention: osculating = mean + corrections (additive)
- These are equivalent at first order

### Comparison Fairness Checklist

- [ ] Same μ, Rₑ, J₂–J₅ (import from `constants.py`)
- [ ] Same initial Cartesian (r₀, v₀) to 15 digits (compute from Keplerian via
      same `keplerian_to_cartesian`)
- [ ] Same Cowell truth (heyoka, tol=1e-15)
- [ ] Same output epochs (same t_grid)
- [ ] Same arc lengths (same n_orbits per case)
- [ ] Both theories at first order only
- [ ] Breakwell-Vagners calibration applied (improves Lara's in-track accuracy)

---

## What NOT to Do

1. **Do not derive formulas from scratch.** Transcribe from papers/Vallado.
2. **Do not optimize before validating.** Pure NumPy first.
3. **Do not implement second-order corrections.** First-order vs first-order.
4. **Do not add tesseral harmonics, drag, or third-body.** Zonal-only.
5. **Do not use Orekit as the Lara implementation.** Build from scratch to know
   exactly what's being compared. Use Orekit Brouwer only for cross-validation.
6. **Do not implement Poincaré variables.** Not needed for first-order Keplerian
   corrections. Accept the e→0 limitation as a scientific finding.
7. **Do not skip the Orekit cross-check.** If our Lara implementation disagrees with
   Orekit Brouwer by more than ~10 m, there's a bug.
