# GEqOE Taylor Propagator with Automatic STM — Implementation Plan

## Goal

Build a high-performance orbit propagator using **Generalized Equinoctial Orbital Elements (GEqOE)** with the **heyoka.py** Taylor integrator. The propagator must:

1. Propagate the 6-element GEqOE state using adaptive-order Taylor stepping (AD-generated coefficients)
2. Automatically compute the State Transition Matrix (STM) via `var_ode_sys`
3. Support arbitrary conservative perturbations (initially J2-only, then extensible)
4. Provide dense output (polynomial evaluation at any sub-step time)
5. Match or exceed the performance of an existing hand-derived analytical Taylor-4 J2-only implementation

**Reference paper**: Bau, Hernando-Ayuso & Bombardelli (2021), "A generalization of the equinoctial orbital elements", Celestial Mechanics and Dynamical Astronomy, 133:50. All equation numbers below refer to this paper.

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundations (constants, conversions, perturbation interface, J2 model) | DONE |
| 2 | Symbolic RHS construction (`rhs.py`) | DONE |
| 3 | heyoka integrators (state-only, STM, propagation helpers) | DONE |
| 4 | Validation tests (14 tests passing) | DONE |
| 5 | Dense output + demo script | DONE |
| -- | Cowell ground truth reference propagator | DONE |
| 6 | General perturbations (third-body, non-conservative, higher geopotential) | DONE (core) |
| 7a | Zonal harmonics (J2–Jn via auto-gradient) | DONE |
| 7b | Post-review hardening + formal implementation note | DONE |
| 8 | Continuous thrust and maneuver characterization framework | IN PROGRESS (8a done, 8b core + smooth law) |

### Achieved Performance (J2-only, tol=1e-15)

- **12-day propagation**: 3.9 ms, ~8 steps/orbit, Taylor order 19
- **1-year propagation**: 98 ms
- **Accuracy vs paper Appendix C**: 1.68e-7 km (0.17 mm) position error
- **Accuracy vs heyoka Cowell ground truth**: 4.6e-8 km position error
- **STM vs finite differences**: relative error ~1.5e-6
- **Conversion round-trip**: <1e-10 km position, <1e-13 km/s velocity

### Phase 6 Results (J2 + Sun + Moon)

- **General equations vs J2-only**: position agreement < 1e-5 km (12-day)
- **J2+Sun+Moon vs Cowell ground truth**: < 1e-3 km (12-day)
- **Third-body effect size**: measurable but < 10 km for 12-day LEO
- **JIT compile time**: ~2-4s (coarse ephemeris), ~minutes (full precision)
- **25 GEqOE tests passing** (14 original + 11 Phase 6)

### Phase 7a Results (Zonal Harmonics)

- **ZonalPerturbation({2: J2}) vs paper reference**: < 1e-5 km (12-day)
- **J2+J3+J4 vs Cowell ground truth**: < 1e-4 km (12-day)
- **J3+J4 effect size**: measurable (0.01–100 km range for 12-day LEO)
- **nu conservation**: preserved to machine precision (E_dot=0)
- **Auto-gradient (diff_tensors+subs) vs finite differences**: relative error < 5e-5
- **Optimized zonal fast path**: no Cartesian detour, Euler identity for 2U-rFr,
  F_h from dU/dzhat (because r·eZ=0 in the orbital plane)
- **51 GEqOE tests passing** (14 original + 11 Phase 6 + 22 Phase 7a + 4 hardening regressions)

### Phase 7b Results (Post-Review Hardening)

- **J2 fast path is now explicit opt-in** via `_j2_fast_path`, avoiding silent
  misrouting of arbitrary static conservative models into the pure-J2 equations
- **General path no longer requires hidden perturbation attributes** (`A`);
  `mu` falls back to the package constant when absent
- **Standalone third-body propagation works** without wrapping in a composite model
- **Time-dependent perturbations are invariant to integrator `t0` offsets** via
  symbolic relative time `hy.time - t0`
- **Formal LaTeX implementation note added** in `docs/geqoe_taylor/paper/`

### Phase 8a Results (Continuous-Thrust Core)

- **7-state GEqOE propagation implemented** with mass-augmented state
  $(\nu, p_1, p_2, K, q_1, q_2, m)$
- **Continuous-thrust control layer added** via `ContinuousThrustLaw`,
  `ConstantRTNThrustLaw`, and `ContinuousThrustPerturbation`
- **Control coefficients exposed through `hy.par[i]`** so thrust magnitude and
  `I_{sp}` can be changed without rebuilding the symbolic graph
- **Dedicated thrust integrator builders added**:
  `build_thrust_state_integrator()` and `build_thrust_stm_integrator()`
- **Mass depletion implemented** with
  $\dot{m} = -T / (g_0 I_{sp})$ using thrust magnitude in newtons and mass in kg
- **GEqOE vs Cowell validation added** through a generic heyoka Cowell path for
  arbitrary perturbation models, including propagated mass
- **62 GEqOE tests passing** (`test_geqoe_taylor.py`,
  `test_geqoe_taylor_general.py`, `test_geqoe_taylor_zonal.py`,
  `test_geqoe_taylor_thrust.py`)

### Phase 8b Results (Sensitivity Core)

- **State and parameter sensitivities implemented** for the 7-state thrust
  system using `hy.var_args.vars | hy.var_args.params`
- **Dedicated builder added**: `build_thrust_sensitivity_integrator()`
- **Variational extraction helper added**:
  `extract_variational_matrices()` returning `(y, Phi_x, Phi_p, param_names)`
- **Runtime parameter ordering exposed** via `parameter_names_from_map()`
- **Endpoint Jacobian selector added** via `extract_endpoint_jacobian()`
- **Single-arc smooth spline law added** via `CubicHermiteRTNThrustLaw`
- **Finite-difference regression added** for thrust-parameter endpoint
  sensitivities

### Commits

- `c101e50`: feat: Add heyoka-based GEqOE Taylor propagator with automatic STM (10 files, 849 lines)
- `6921a53`: feat: Add Cowell ground truth reference and interactive demo script (3 files, 429 lines)

---

## Critical Design Decision: Use K Instead of L

The GEqOE as defined in the paper use the generalized mean longitude $L$ as the fast variable. Computing intermediate quantities ($r$, $\dot{r}$, $X$, $Y$, $\cos L$, $\sin L$) from $L$ requires **solving the generalized Kepler equation** (Eq. 30):

$$L = K + p_1 \cos K - p_2 \sin K$$

This is an implicit equation for $K$ given $L$, requiring iterative Newton-Raphson — **incompatible with heyoka's symbolic expression system**.

**Solution**: Integrate $K$ (the generalized eccentric longitude) instead of $L$. The time derivative $\dot{K}$ is given explicitly in Eq. 75 of the paper:

$$\dot{K} = \frac{w}{r} + \frac{h - c}{r^2} - w_h + \frac{1}{c}\left(1 + \alpha\left(1 - \frac{r}{a}\right)\right)(2\mathcal{U} - r F_r) - \frac{r\dot{r}\alpha}{\mu w}\left(1 - \frac{r}{a\beta}\right)\dot{\mathcal{E}}$$

where $w = \sqrt{\mu/a}$. All quantities on the RHS ($r$, $\dot{r}$, $X$, $Y$, $h$, $c$, etc.) can be computed from $(\nu, p_1, p_2, K, q_1, q_2)$ using only elementary functions — no implicit solve needed.

**The integrated state vector is**: $\mathbf{y} = (\nu, p_1, p_2, K, q_1, q_2)$

If $L$ is ever needed (e.g., for output), compute it from $K$ via the forward Kepler equation (Eq. 30), which is explicit.

---

## Repository Structure

```
src/astrodyn_core/geqoe_taylor/
    __init__.py                    # Package init, exports public API
    constants.py                   # Physical constants (mu, J2, Re, A_J2)
    perturbations/
        __init__.py
        base.py                    # PerturbationModel protocol
        j2.py                      # J2 zonal harmonic (symbolic + numeric)
        thrust.py                  # Continuous-thrust wrapper -> Cartesian P + m_dot
        zonal.py                   # Arbitrary zonal harmonics J2-Jn (auto-gradient)
    rhs.py                         # GEqOE RHS builder (symbolic heyoka expressions)
    integrator.py                  # Wrapper: build/run heyoka integrators
    conversions.py                 # Cartesian <-> GEqOE conversions (numerical)
    thrust.py                      # Continuous-thrust law abstractions
    utils.py                       # Helpers (Kepler solve for L<->K)
    cowell.py                      # Cowell ground truth helpers (J2 + general)
tests/
    test_geqoe_taylor.py           # 14 tests: conversions, propagation, STM, Cowell
    test_geqoe_taylor_thrust.py    # 6 tests: thrust core, mass flow, Cowell match
examples/
    geqoe_taylor_demo.py           # Interactive demo (7 sections)
```

---

## Phase 1: Foundations (DONE)

### 1.1 Physical Constants (`constants.py`)

```python
MU = 398600.4354360959        # km^3/s^2 (from paper Eq. 59)
J2 = 1.08262617385222e-3      # (from paper Eq. 59)
RE = 6378.1366                # km (from paper Eq. 59)
A_J2 = MU * J2 * RE**2 / 2   # Convenience constant for J2 potential
```

### 1.2 Coordinate Conversions (`conversions.py`)

Implemented in plain NumPy (not heyoka expressions). These run outside the integrator.

**Cartesian -> GEqOE**: `cart2geqoe(r_vec, v_vec, mu, perturbation)` -> `(nu, p1, p2, K, q1, q2)`
**GEqOE -> Cartesian**: `geqoe2cart(geqoe, mu, perturbation)` -> `(r_vec, v_vec)`

Both accept a `PerturbationModel` for the U-dependent terms (h, c, velocity reconstruction).

### 1.3 Perturbation Model Interface (`perturbations/base.py`)

```python
class PerturbationModel(Protocol):
    def U_expr(self, x, y, z, r_mag, t, pars: dict) -> "heyoka expression": ...
    def U_numeric(self, r_vec, t) -> float: ...
```

### 1.4 J2 Perturbation (`perturbations/j2.py`)

From Eq. 56: $\mathcal{U} = -A/r^3(1 - 3\hat{z}^2)$ where $A = \mu J_2 r_e^2 / 2$.

Both symbolic (heyoka expression) and numeric (NumPy) versions implemented.

---

## Phase 2: Symbolic RHS Construction (DONE)

### 2.1 State Variables and Parameters

State: `[nu, p1, p2, K, q1, q2]` as heyoka variables.
Parameters:
- J2 fast path: `mu = hy.par[0]`, `A_J2 = hy.par[1]`
- General and zonal paths: `mu = hy.par[0]`

### 2.2 Intermediate Quantities

All built as heyoka expressions from state variables:

```
a = (mu / nu^2)^(1/3)           # Semi-major axis (Eq. 21)
g2 = p1^2 + p2^2                # (Eq. 22)
beta = sqrt(1 - g2)             # (Eq. 40)
alpha = 1 / (1 + beta)          # (Eq. 40)
X, Y = ...                      # Position in equinoctial frame (Eq. 42)
r = a * (1 - p1*sinK - p2*cosK) # Orbital distance (Eq. 31)
zhat = 2*(Y*q2 - X*q1) / (r*(1+q1^2+q2^2))   # (Eq. 57)
c = (mu^2/nu)^(1/3) * beta      # Generalized angular momentum (Eq. 23)
h = sqrt(c^2 - 2*r^2*U)         # Physical angular momentum (Eq. 44)
h - c = -2*r^2*U / (h + c)      # Stable computation (avoids cancellation)
```

### 2.3 J2-Only Simplifications (Section 7.1)

- Energy $\mathcal{E}$ is a first integral: $\dot{\nu} = 0$, $\dot{\mathcal{E}} = 0$
- $2\mathcal{U} - rF_r = -\mathcal{U}$ (derived in Section 7.1)

### 2.4 Assembled ODEs (corrected)

**BUG FIX**: The original plan specified `ell/alpha` ($= c^2/(\mu\alpha)$) as the coefficient in $\dot{K}$. This is **wrong** — it gave $\dot{K} \approx 7.4 \times 10^{-3}$ (7x too large). The correct coefficient is `1 + alpha*(1 - r/a)`, matching Eq. 75 with the J2-only substitution $2\mathcal{U} - rF_r = -\mathcal{U}$ and $\dot{\mathcal{E}} = 0$.

Note: The existing L-based code uses $\Gamma = 1/\alpha + \alpha(1 - r/a)$ for $\dot{L}$. For $\dot{K}$, the coefficient differs because the L->K transformation shifts it: the $\dot{K}$ coefficient is $1 + \alpha(1 - r/a)$, not $\Gamma$.

```
nu_dot = 0                                                # Energy integral
p1_dot = p2*(d - w_h) - (1/c)*(X/a + 2*p2)*U             # Eq. 47 simplified
p2_dot = p1*(w_h - d) + (1/c)*(Y/a + 2*p1)*U             # Eq. 48 simplified
K_dot  = w/r + d - w_h - (1/c)*(1 + alpha*(1 - r/a))*U   # Eq. 75 (CORRECTED)
q1_dot = -I * sinL                                         # Eq. 50 simplified
q2_dot = -I * cosL                                         # Eq. 51 simplified
```

where:
- `d = (h - c) / r^2` with stable h-c computation
- `w_h = I * zhat`
- `I = 3*A*zhat*(1 - q1^2 - q2^2) / (h*r^3)` (Eq. 58)
- `1 - r/a = p1*sinK + p2*cosK` (Eq. 31, for numerical stability)

---

## Phase 3: heyoka Integrators (DONE)

### 3.1 State-Only Integrator (6 DOF)

`build_state_integrator(perturbation, ic, tol=1e-15)` -> `(ta, par_map)`

### 3.2 State + STM Integrator (42 DOF)

`build_stm_integrator(perturbation, ic, tol=1e-15)` -> `(ta, par_map)`

Uses `hy.var_ode_sys(sys, hy.var_args.vars, order=1)` for automatic variational equations. Compact mode enabled by default for the 42-DOF system.

### 3.3 Propagation Helpers

- `propagate(ta, t_final)` -> `(times, states)` — step-by-step with adaptive stepping
- `propagate_grid(ta, t_grid)` -> states array — dense output at specified times
- `extract_stm(state_aug)` -> `(y, phi)` — extract 6x6 STM from 42-element state

---

## Phase 4: Validation (DONE — 14 tests passing)

### Test Classes

| Class | Tests | Description |
|-------|-------|-------------|
| `TestConversions` | 4 | Round-trip (case a, elliptical, retrograde), paper Table 3 values |
| `TestKeplerEquation` | 2 | K->L->K round-trip, K=0 edge case |
| `TestPropagation` | 4 | 12-day accuracy, nu conservation, step vs propagate_until, grid output |
| `TestSTM` | 2 | STM vs finite differences (<1e-5 rel err), identity at t=0 |
| `TestCowellGroundTruth` | 2 | GEqOE vs heyoka Cowell (<1e-6 km), Cowell vs paper (<1e-5 km) |

Run: `conda run -n astrodyn-core-env pytest tests/test_geqoe_taylor.py -q`

---

## Phase 5: Dense Output and Demo (DONE)

### 5.1 Dense Output

Uses heyoka's built-in `propagate_grid()` for sub-step polynomial evaluation. No custom Taylor coefficient handling needed.

### 5.2 Cowell Ground Truth (`cowell.py`)

Two independent Cartesian J2 propagators for validation:
- `propagate_cowell()` — scipy DOP853 (rtol=atol=1e-14)
- `propagate_cowell_heyoka()` — heyoka Taylor (tol=1e-15, highest accuracy)

Ground truth comparison (12 days, J2-only):
```
Method                     pos error (km)    pos error (m)
GEqOE Taylor (heyoka)      4.6e-08           0.0000
Paper Dromo (App. C)        2.1e-07           0.0002
Scipy DOP853                9.6e-07           0.0010
```

### 5.3 Interactive Demo (`examples/geqoe_taylor_demo.py`)

Seven sections: conversions, 12-day propagation, step history, dense output grid, STM computation, ground truth comparison, 1-year performance summary.

Run: `conda run -n astrodyn-core-env python examples/geqoe_taylor_demo.py`

---

## Phase 6: General Perturbations (DONE — core)

This phase extends the propagator beyond J2-only to support arbitrary conservative and non-conservative perturbations, using the full equations of motion (Eqs. 45-51).

### 6.1 Implemented: Full Equations of Motion

**IMPORTANT CORRECTION**: The q-dot equations listed in the original plan were wrong. The correct forms from the paper (Eqs. 50-51) are:

$$\dot{q}_1 = \frac{\gamma}{2} w_Y, \quad \dot{q}_2 = \frac{\gamma}{2} w_X$$

where $\gamma = 1 + q_1^2 + q_2^2$, and from Eq. 14/52:

$$w_X = \frac{X}{h} F_h, \quad w_Y = \frac{Y}{h} F_h, \quad w_h = w_X q_1 - w_Y q_2$$

Here $F_h = \mathbf{F} \cdot \mathbf{e}_Z$ is the total out-of-plane perturbation force projection, where $\mathbf{F} = \mathbf{P} - \nabla\mathcal{U}$ combines conservative and non-conservative forces (Eq. 3).

The p-dot equations use the compact paper form (Eqs. 47-48) with combined $F_r = \mathbf{F} \cdot \mathbf{e}_r$ and the energy derivative $\dot{\mathcal{E}}$ term:

$$\dot{p}_1 = p_2(d - w_h) + \frac{1}{c}\left(\frac{X}{a} + 2p_2\right)(2\mathcal{U} - rF_r) + \frac{1}{c^2}\left[Y(r + \varrho) + r^2 p_1\right]\dot{\mathcal{E}}$$

$$\dot{p}_2 = p_1(w_h - d) - \frac{1}{c}\left(\frac{Y}{a} + 2p_1\right)(2\mathcal{U} - rF_r) + \frac{1}{c^2}\left[X(r + \varrho) + r^2 p_2\right]\dot{\mathcal{E}}$$

where $\varrho = c^2/\mu$ and $d = (h-c)/r^2$.

For $\dot{K}$, instead of the expanded Eq. 75, the implementation derives it from $\dot{\mathcal{L}}$ (Eq. 49):

$$\dot{K} = \frac{a}{r}\left(\dot{\mathcal{L}} - \dot{p}_1 \cos K + \dot{p}_2 \sin K\right)$$

This avoids a separate formula and is algebraically exact (from $\mathcal{L} = K + p_1 \cos K - p_2 \sin K$).

### 6.2 Implemented Architecture

**Auto-detection**: `build_geqoe_system()` first checks `_zonal_fast_path`, then `_j2_fast_path`, and otherwise falls back to the general equations. The J2 shortcut is opt-in because it hard-codes the Eq. 56 potential and Section 7.1 simplifications.

**Perturbation protocol** (`perturbations/base.py`): `GeneralPerturbationModel` extends `PerturbationModel` with `grad_U_expr`, `P_expr`, `U_t_expr`.

**Key identity for $2\mathcal{U} - rF_r$**: Using the Euler theorem for homogeneous functions:
$$2\mathcal{U} - rF_r = 2\mathcal{U} + \mathbf{r} \cdot \nabla\mathcal{U} - r P_r$$

The term $\mathbf{r} \cdot \nabla\mathcal{U} = x \frac{\partial\mathcal{U}}{\partial x} + y \frac{\partial\mathcal{U}}{\partial y} + z \frac{\partial\mathcal{U}}{\partial z}$. For any spherical harmonic term of degree $n$, this equals $-(n+1)\mathcal{U}_n$, giving $2\mathcal{U}_n - rF_{r,n} = (1-n)\mathcal{U}_n$.

### 6.3 Implemented: Third-Body Gravity

`perturbations/third_body.py` — Sun via `hy.model.vsop2013_cartesian` (heliocentric EMB, negated), Moon via `hy.model.elp2000_cartesian_e2000`. Both converted from ecliptic J2000 to equatorial J2000 via obliquity rotation ($\varepsilon = 23.4393°$). Time-dependent models consume the relative symbolic time `hy.time - t0`, so `epoch_jd` always refers to the epoch of the initial state even when the integrator uses a nonzero external time origin. Ephemeris truncation threshold controls JIT compile time:

| Threshold (Sun / Moon) | JIT time | Position accuracy (12-day) |
|------------------------|----------|---------------------------|
| 1e-9 / 1e-6 (default) | ~minutes | sub-km |
| 1e-4 / 1e-2 (coarse) | ~2-4s | adequate for most LEO |

### 6.4 Implemented: Composite Perturbation

`perturbations/composite.py` — Separates conservative (U) and non-conservative (P) contributions. Conservative models define the non-osculating ellipse (h, c); non-conservative models are external forcing.

### 6.5 Implemented: Cowell Ground Truth

`cowell.py` extended with `propagate_cowell_heyoka_full()` supporting J2 + Sun + Moon via the same ephemeris models.

### 6.6 Commits and Test Results

- **25 GEqOE tests passing** (14 original + 11 Phase 6)
- **292 total project tests passing**
- General equations match J2-only: position error < 1e-5 km (12-day)
- J2 + Sun + Moon vs Cowell ground truth: < 1e-3 km (12-day)

---

## Phase 7: Higher-Order Geopotential (Zonal DONE, Tesseral TODO)

Extend the conservative potential $\mathcal{U}$ to support arbitrary zonal ($J_n$) and tesseral ($C_{nm}$, $S_{nm}$) harmonics.

### 7.1 Key Enabler: Automatic Gradient via `hy.diff_tensors` + `hy.subs`

heyoka provides symbolic differentiation (`hy.diff_tensors`) and expression substitution (`hy.subs`). This eliminates manual gradient derivation for new potential models:

```python
import heyoka as hy

# 1. Build U using placeholder Cartesian variables
_x, _y, _z = hy.make_vars("_x", "_y", "_z")
_r = hy.sqrt(_x*_x + _y*_y + _z*_z)
U_placeholder = build_potential(_x, _y, _z, _r, ...)

# 2. Auto-differentiate
dt = hy.diff_tensors([U_placeholder], [_x, _y, _z], diff_order=1)
dUdx_ph = dt[0, 1]  # dU/d_x
dUdy_ph = dt[0, 2]  # dU/d_y
dUdz_ph = dt[0, 3]  # dU/d_z

# 3. Substitute GEqOE-derived Cartesian positions
smap = {"_x": x_cart_expr, "_y": y_cart_expr, "_z": z_cart_expr}
dUdx = hy.subs(dUdx_ph, smap)
dUdy = hy.subs(dUdy_ph, smap)
dUdz = hy.subs(dUdz_ph, smap)
U_val = hy.subs(U_placeholder, smap)
```

This approach works for **any** potential that can be expressed symbolically in heyoka. The gradient is exact (not numerical), and CSE (common subexpression elimination) during JIT compilation keeps the compiled code efficient.

### 7.2 Zonal Harmonics ($J_n$, arbitrary degree)

Zonal harmonics are **conservative** and **time-independent**, so $\dot{\mathcal{E}} = 0$ and $\dot{\nu} = 0$. They extend $\mathcal{U}$ directly.

**Potential** (individual term, degree $n$):

$$\mathcal{U}_{J_n} = \frac{\mu J_n R_e^n}{r^{n+1}} P_n(\hat{z})$$

where $\hat{z} = z/r$ and $P_n$ is the Legendre polynomial of degree $n$.

**Euler identity**: Since $\mathcal{U}_{J_n}$ is homogeneous of degree $-(n+1)$ in $(x,y,z)$:

$$2\mathcal{U}_{J_n} - r F_{r,J_n} = (1-n)\,\mathcal{U}_{J_n}$$

For $n=2$: $-\mathcal{U}$ (current J2). For $n=3$: $-2\mathcal{U}_{J_3}$. For $n=4$: $-3\mathcal{U}_{J_4}$.

**Legendre polynomial recurrence** (builds heyoka expressions):

```python
def _legendre_P(n, x):
    """P_n(x) as a heyoka expression via Bonnet recurrence."""
    if n == 0: return 1.0
    if n == 1: return x
    P_prev, P_curr = 1.0, x
    for k in range(2, n + 1):
        P_next = ((2*k - 1) * x * P_curr - (k - 1) * P_prev) / k
        P_prev, P_curr = P_curr, P_next
    return P_curr
```

**Gradient**: Use `hy.diff_tensors` + `hy.subs` (Section 7.1) — no manual derivation needed. Alternatively, the analytical gradient for degree $n$ is:

$$\frac{\partial \mathcal{U}_{J_n}}{\partial x} = \frac{C_n \, x}{r^{n+3}}\left[-(n+1)P_n(\hat{z}) - \hat{z}\,P_n'(\hat{z})\right]$$

$$\frac{\partial \mathcal{U}_{J_n}}{\partial z} = \frac{C_n}{r^{n+2}}\left[-(n+1)\hat{z}\,P_n(\hat{z}) + (1-\hat{z}^2)P_n'(\hat{z})\right]$$

where $C_n = \mu J_n R_e^n$ and $P_n'$ is built alongside $P_n$ via the differentiated recurrence.

**Implementation plan**: `perturbations/zonal.py`

```python
class ZonalHarmonicsPerturbation:
    """Zonal harmonics J2 through Jn_max."""
    is_conservative = True
    is_time_dependent = False

    def __init__(self, j_coeffs: dict[int, float], mu=MU, re=RE):
        """j_coeffs: {2: J2, 3: J3, 4: J4, ...}"""
        ...

    def U_expr(self, x, y, z, r_mag, t, pars):
        zhat = z / r_mag
        total = 0.0
        for n, Jn in self.j_coeffs.items():
            Cn = self.mu * Jn * self.re**n
            total = total + Cn / r_mag**(n+1) * _legendre_P(n, zhat)
        return total

    def grad_U_expr(self, ...):
        # Option A: auto-diff via hy.diff_tensors + hy.subs
        # Option B: analytical formula using P_n and P_n'
        ...
```

**Standard J-coefficients** (EGM2008, unnormalized):
| n | $J_n$ |
|---|-------|
| 2 | 1.08262617385222e-3 |
| 3 | -2.53265648533224e-6 |
| 4 | -1.61989759991697e-6 |
| 5 | -2.27296082868698e-7 |
| 6 | 5.40681239107085e-7 |

### 7.3 Full Geopotential: Tesseral and Sectoral Harmonics ($C_{nm}$, $S_{nm}$)

Tesseral harmonics ($m \neq 0$) introduce **time dependence** because the potential is defined in the Earth-fixed (body) frame, which rotates.

**Potential** (degree $n$, order $m$):

$$\mathcal{U}_{nm} = \frac{\mu R_e^n}{r^{n+1}} \bar{P}_{nm}(\sin\varphi) \left[C_{nm}\cos(m\lambda) + S_{nm}\sin(m\lambda)\right]$$

where $\varphi$ is geocentric latitude ($\sin\varphi = z/r$) and $\lambda$ is geographic (body-fixed) longitude.

**Earth rotation**: The body-fixed longitude requires the Greenwich Sidereal Time:

$$\theta(t) = \theta_0 + \omega_\oplus \, t$$

where $\omega_\oplus = 7.2921150 \times 10^{-5}$ rad/s. The body-fixed coordinates are:

$$x_\text{bf} = x\cos\theta + y\sin\theta, \quad y_\text{bf} = -x\sin\theta + y\cos\theta, \quad z_\text{bf} = z$$

**Longitude trigonometry** (avoids `atan2` — built recursively):

$$\cos\lambda = x_\text{bf}/r_{xy}, \quad \sin\lambda = y_\text{bf}/r_{xy}, \quad r_{xy} = \sqrt{x^2 + y^2}$$

Higher-order terms via Chebyshev recurrence:

$$\cos(m\lambda) = 2\cos\lambda\cos((m-1)\lambda) - \cos((m-2)\lambda)$$

**Associated Legendre functions** (fully normalized $\bar{P}_{nm}$):

$$\bar{P}_{mm}(u) = \sqrt{\frac{(2m+1)!!}{(2m)!!}} (1-u^2)^{m/2}$$
$$\bar{P}_{nm}(u) = u \sqrt{\frac{2n-1}{n-m}} \bar{P}_{n-1,m}(u) - \sqrt{\frac{(n+m-1)(n-m-1)(2n-1)}{(n-m)(n+m)(2n-3)}} \bar{P}_{n-2,m}(u)$$

All built as heyoka symbolic expressions.

**Gradient**: Use `hy.diff_tensors` + `hy.subs` (Section 7.1). This is the strongly preferred approach for tesseral harmonics — the manual gradient is complex but auto-differentiation handles it exactly.

**Time derivative $\mathcal{U}_t$** (for $\dot{\mathcal{E}}$ in the general equations): Since time enters only through $\theta(t) = \theta_0 + \omega_\oplus t$, and $\cos\theta$, $\sin\theta$ are heyoka expressions of `hy.time`, the time derivative can be computed by:

1. Introduce a temporary variable `_t` in place of `hy.time`
2. Build $\mathcal{U}$ with this variable
3. Differentiate: $\partial\mathcal{U}/\partial t$ via `hy.diff_tensors`
4. Substitute `_t → hy.time` via `hy.subs`

Or analytically: $\mathcal{U}_t = -m\omega_\oplus \times$ (same expression with $\cos(m\lambda) \to \sin(m\lambda)$ and $\sin(m\lambda) \to -\cos(m\lambda)$).

### 7.4 Design Choices: U vs P for Higher Harmonics

| Approach | Pros | Cons |
|----------|------|------|
| **Add to U** (conservative) | More accurate h, c; preserves element structure | Larger expression trees |
| **Treat as P** (non-conservative) | Simpler; same code path as third-body | h, c don't reflect higher-order terms |

**Recommendation**:
- **J2 always in U** (defines the non-osculating ellipse)
- **J3–J6 in U** (small expression overhead, meaningful improvement for MEO/GEO)
- **J7+ and tesseral in P** (expression complexity grows quadratically with degree; P treatment is adequate for small perturbations)
- For research/high-accuracy: everything in U is possible up to degree ~20-30

### 7.5 Expression Size and JIT Compilation Limits

The number of spherical harmonic terms grows as $\sim N(N+1)/2$ for max degree $N$:

| Max degree $N$ | Terms | JIT compile (est.) | Practical? |
|----------------|-------|-------------------|------------|
| 4 | 15 | < 5s | Yes |
| 10 | 66 | ~10-20s | Yes |
| 20 | 231 | ~1-2 min | Yes (compact_mode) |
| 50 | 1326 | ~5-10 min | Marginal |
| 70+ | 2556+ | Very slow | Use Cowell instead |

For high-degree models ($N > 30$), a **Cowell formulation** with compiled acceleration routines (e.g., `heyoka.cfunc` for vectorized evaluation) is more appropriate than GEqOE.

### 7.6 Implementation Plan

**Step 1**: `perturbations/zonal.py` — Zonal harmonics $J_2$ through $J_{n_\text{max}}$ using `hy.diff_tensors` for auto-gradient. Conservative, time-independent.

**Step 2**: `perturbations/geopotential.py` — Full spherical harmonics with Earth rotation model. Conservative, time-dependent. Uses `hy.diff_tensors` for gradient and time derivative.

**Step 3**: Validation against Cowell ground truth with the same geopotential model. Compare paper cases (b)-(d).

**Step 4** (optional): Atmospheric drag via `hy.model.nrlmsise00_tn` (thermoNET density model built into heyoka) + `hy.model.cart2geo` (Cartesian → geodetic). Non-conservative P.

### 7.7 Validation Plan

- J2+J3+J4 vs Cowell ground truth (12-day LEO)
- Full 4×4 geopotential vs Cowell (12-day, paper cases b-d)
- Energy drift rate for time-dependent potentials ($\dot{\nu} \neq 0$)
- STM validation with finite differences for composite models

---

## Phase 8: Continuous Thrust and Maneuver Characterization (IN PROGRESS)

This phase adds continuous thrust as a first-class non-conservative component of
the GEqOE Taylor framework and is the recommended path toward maneuver
optimization, estimation, and future maneuver detection / characterization.

### 8.1 Guiding Decision: Implement Smooth Control Infrastructure First

Two independent research notes were reviewed:

- `docs/geqoe_taylor/compass_continuous_markdown.md`
- `docs/geqoe_taylor/deep-research-report-taylor-continuous-optim.md`

**Conclusion**: the best path is **not** to begin with the full generalized
Fourier/TFC-in-K theory. That is a strong research target, but it should come
**after** a robust continuous-thrust propagation and sensitivity framework is in
place.

The implementation priority should be:

1. **Continuous thrust as a smooth non-conservative force model** in the current
   GEqOE equations
2. **Parameter sensitivities with respect to thrust-law coefficients** using
   heyoka's variational system
3. **Direct optimization-friendly transcription** (multiple shooting first)
4. **Research extension** to generalized Fourier coefficients in the GEqOE fast
   angle $K$
5. **Uncertainty / maneuver characterization** using STMs, higher derivatives,
   and eventually control-distance-like metrics

This ordering best matches the current codebase and optimizes for performance,
flexibility, and derivative consistency.

### 8.2 Why This Is the Best Path

From the research review, the strongest engineering conclusions are:

- **Multiple shooting is the practical sweet spot** for GEqOE + heyoka when the
  control is smooth but discontinuities or active constraints may exist
- **Smooth thrust parameterizations** (cubic splines / B-splines first, then
  Fourier) are preferable to piecewise-constant controls because Taylor
  integrators lose efficiency on non-smooth controls
- **Control should enter through the existing non-conservative term**
  $\mathbf{P}$, preserving the GEqOE separation $\mathbf{F}=\mathbf{P}-\nabla U$
- **Sensitivities with respect to control coefficients** are essential from day
  one, because they unlock both optimization and future uncertainty workflows
- **Generalized TFC/Fourier-in-K is a Phase-2 research extension**, not the
  lowest-risk initial implementation

### 8.3 Implemented Phase 8a Architecture

The recommended architecture is:

#### A. Extend the propagated state with mass

Use the augmented state:

$$\mathbf{y}_{thrust} = (\nu, p_1, p_2, K, q_1, q_2, m)$$

with

$$\dot{m} = -\frac{T}{g_0 I_{sp}}$$

or a more general power-coupled electric-propulsion law if required.

Rationale:
- Continuous thrust without mass flow is incomplete for maneuver
  characterization / optimization
- The added scalar state is cheap relative to the GEqOE geometry already being
  evaluated
- The 7-state system still fits naturally into heyoka variational propagation

#### B. Add a control-law abstraction on top of `P_expr`

Implemented layer:

```python
class ContinuousThrustLaw(Protocol):
    def thrust_rtn_expr(self, state, t, pars, prefix) -> tuple:
        """Return (T_r, T_t, T_n, T_mag, Isp) in RTN coordinates."""
```

and a perturbation wrapper:

```python
class ContinuousThrustPerturbation:
    """Map a thrust law into GEqOE-compatible non-conservative force projections."""
```

This wrapper should:
- support at least RTN / orbital-frame thrust components
- optionally support inertial-frame thrust definitions
- expose thrust magnitude, throttle, and $I_{sp}$ / power relations
- return Cartesian $\mathbf{P}$ for compatibility with the existing general path

The current implementation delivers:

- `ContinuousThrustLaw` in `src/astrodyn_core/geqoe_taylor/thrust.py`
- `ConstantRTNThrustLaw` as the first validation law
- `CubicHermiteRTNThrustLaw` as the first smooth single-arc spline law
- `ContinuousThrustPerturbation` in
  `src/astrodyn_core/geqoe_taylor/perturbations/thrust.py`
- `build_thrust_state_integrator()` / `build_thrust_stm_integrator()` as the
  explicit 7-state public API
- `build_thrust_sensitivity_integrator()` and
  `extract_endpoint_jacobian()` for direct endpoint Jacobian access
- `geqoe2cart()` support for 7-state inputs by ignoring the trailing mass entry

Not yet implemented inside Phase 8:

- multi-segment spline / B-spline control laws
- optimization / multiple-shooting transcription utilities

#### C. Treat control coefficients as differentiable parameters

The thrust-law coefficients should be mapped to `hy.par[i]`, not hard-coded as
Python literals. This enables:

- cheap coefficient changes without rebuilding the symbolic graph
- variational equations with respect to both state and control coefficients
- endpoint Jacobians for direct optimization

**Recommended heyoka configuration**:

```python
hy.var_ode_sys(sys, hy.var_args.vars | hy.var_args.params, order=1)
```

This is the key enabler for performance: the same symbolic ODE defines both the
state dynamics and the exact control sensitivities.

### 8.4 Control Parameterization Strategy

#### Phase 8a core (implemented): constant RTN law as the validation baseline

The first shipped control law is intentionally simple:

- `ConstantRTNThrustLaw` with runtime parameters for `(T_r, T_t, T_n, I_{sp})`
- smooth in the Taylor sense (no switching / piecewise-constant control)
- enough to validate mass flow, energy growth, and GEqOE-vs-Cowell agreement

This is a validation-oriented control law, not the final optimization-facing
parameterization.

#### Implemented next step: cubic Hermite spline in normalized arc time

The current smooth parameterization is:

- `CubicHermiteRTNThrustLaw(duration_s, ...)`
- RTN thrust components represented as cubic Hermite polynomials in
  $\tau = t / \text{duration}_s$
- endpoint values and endpoint slopes exposed as differentiable runtime
  parameters

This gives a smooth, single-arc control law that is compatible with Taylor
integration and the current parameter-sensitivity machinery.

#### Still pending after the Hermite law: multi-segment cubic splines / B-splines

This remains the next parameterization to add on top of the current Hermite
law for richer arc modeling.

Why:
- smooth enough for Taylor integration
- local support (better than global polynomials)
- flexible but not over-parameterized
- easy to use in multiple shooting
- robust under eclipse windows, arc segmentation, and active constraints

Suggested controls per arc:
- throttle $\sigma(\tau)$
- in-plane angle $\gamma(\tau)$
- out-of-plane angle $\delta(\tau)$

or directly:
- $P_r(\tau)$, $P_f(\tau)$, $P_h(\tau)$

with $\tau \in [0,1]$ per arc.

#### Phase 8b: low-order Fourier per revolution or per arc

Add truncated Fourier expansions once the spline-based pipeline is stable.

Why:
- useful for smooth multi-revolution thrust patterns
- closer to the TFC literature
- lower parameter counts for quasi-periodic solutions

But this should remain secondary to splines because:
- Fourier is less natural with eclipses and local discontinuities
- generalized Fourier in the GEqOE fast angle $K$ requires separate theory work

#### Explicit non-recommendation for the first version

Do **not** start with piecewise-constant thrust as the primary control
parameterization. It is simple, but it is a poor fit for Taylor propagation and
likely to degrade both performance and derivative quality.

### 8.5 Recommended Optimization Strategy

#### Phase 8a: multiple shooting first

This is the recommended first transcription.

Why:
- compatible with existing propagator wrappers
- allows arc boundaries aligned with discontinuities or eclipse transitions
- robust under strong nonlinearities
- leverages exact endpoint sensitivities from variational equations

Decision variables should include:
- control coefficients per arc
- interior node states
- optionally final time

Constraints should include:
- arc continuity
- terminal orbit conditions
- thrust / power / mass constraints
- optional path constraints via node checks and event-aligned segmentation

#### Phase 8b: add collocation / pseudospectral only if needed

These are worthwhile only after the propagation and sensitivity layer is stable.
They may become attractive for:
- very large control dimensions
- many active path constraints
- problems where sparse NLP structure dominates overall performance

### 8.6 Uncertainty-Driven Design Choices

Because uncertainty propagation is a priority, the continuous-thrust
implementation should be designed from the start to support:

1. **STM wrt initial state and control coefficients**
2. **Covariance propagation in the presence of parametric thrust uncertainty**
3. **Control-law estimation / characterization via endpoint sensitivities**
4. **Future higher-order maps** for nonlinear uncertainty propagation

This implies two non-negotiable requirements:

- control coefficients must be represented as heyoka parameters
- the public API should expose sensitivities wrt both state and control

### 8.7 Proposed Phase Breakdown

#### Phase 8a: Continuous-thrust propagation core

Delivered:
- 7-state GEqOE + mass dynamics
- `ContinuousThrustLaw` abstraction
- `ContinuousThrustPerturbation`
- constant RTN thrust validation model (`ConstantRTNThrustLaw`)
- dedicated 7-state integrator builders
- state-only propagation with continuous thrust
- 7-state STM propagation wrt the initial augmented state

Validation completed:
- constant tangential thrust sanity case
- GEqOE vs Cartesian Cowell under the same thrust history
- mass-flow consistency checks

Deferred within the broader Phase 8 roadmap:

- multi-segment spline / B-spline control laws
- direct optimization transcription

#### Phase 8b: Variational sensitivities wrt control coefficients

Delivered in the current 8b core:
- STM wrt initial state and thrust parameters
- endpoint Jacobian extraction utility
- endpoint Jacobian selector for chosen outputs / parameters
- regression tests against finite differences

Validation completed:
- directional derivative tests
- sensitivity agreement vs finite differences for thrust coefficients

Deferred within Phase 8b:

- higher-level endpoint Jacobian assembly helpers for shooting / NLP layers
- optional custom variational argument lists beyond `vars | params`

#### Phase 8c: Multiple-shooting optimization prototype

Deliverables:
- multi-arc transcription helper
- objective / constraint assembly
- interface to a sparse NLP solver

Recommended first objectives:
- minimum propellant
- minimum time with bounded thrust
- weighted smoothness-regularized objective

#### Phase 8d: Generalized Fourier/TFC research extension

Deliverables:
- derive thrust expansions in generalized eccentric longitude $K$
- identify surviving averaged coefficients and relation to classical TFCs
- compare against spline-based direct transcription

This is the main research novelty path and should be pursued once the direct
continuous-thrust backend is already working.

#### Phase 8e: Maneuver characterization and uncertainty

Deliverables:
- parameter-estimation view of continuous thrust
- uncertainty propagation wrt thrust coefficients
- initial exploration of control-distance-like metrics in GEqOE space

Longer-term:
- second-order / higher variational equations
- polynomial maps and nonlinear uncertainty transport

### 8.8 Validation Plan for Continuous Thrust

Recommended test campaign:

1. **Two-body + constant tangential thrust**
   - analytic sanity / monotonic semimajor-axis growth
2. **GEqOE vs Cartesian Cowell with the same thrust law**
   - state agreement over short and medium arcs
3. **Mass depletion tests**
   - consistency with $\int T/(g_0 I_{sp})\,dt$
4. **Gradient verification**
   - endpoint sensitivities wrt control coefficients
5. **Multiple-shooting continuity tests**
   - segment matching and event-aligned arc transitions
6. **Initial uncertainty tests**
   - covariance transport with uncertain thrust coefficients

### 8.9 Final Recommendation

The best path forward is:

1. implement continuous thrust as a **smooth, parameterized non-conservative
   perturbation with mass flow**
2. expose **exact state-and-parameter sensitivities** via heyoka variational
   equations
3. build a **multiple-shooting optimization layer** on top of that
4. only then pursue the **generalized Fourier/TFC-in-K research program**

This path is the best compromise between:
- **performance**: smooth controls preserve Taylor efficiency
- **flexibility**: splines + multi-arc support many mission types
- **uncertainty readiness**: parameter sensitivities are available from the same backend
- **research value**: the generalized TFC formulation remains an open,
  publishable extension rather than a prerequisite blocker

---

## Key Numerical Values for Testing

From the paper, case (a) — LEO circular i=45deg:

**Initial Cartesian** (Table 2):
```
r = [7178.1366, 0, 0] km
v = [0, 5.269240572916780, 5.269240572916780] km/s
```

**Initial GEqOE** (Table 3):
```
nu = 1.0395e-3 rad/s
p1 = 0
p2 = -8.5476e-4
K  = 0 rad
q1 = 0
q2 = 0.41421
```

**Reference final state after 12 days, J2 only** (Appendix C):
```
r_f = [-5398.929377366906, -390.257240638229, -4693.719111636971] km
v_f = [2.214482567493, -6.845637008953, -1.977748618717] km/s
```

---

## Dependencies

```
heyoka >= 6.0.0    # Taylor integrator with var_ode_sys support (tested with 7.9.2)
numpy >= 1.24
scipy              # For DOP853 Cowell reference
pytest
```

Install: `conda install -c conda-forge heyoka-py` (recommended) or `pip install heyoka`

---

## Pitfalls and Lessons Learned

1. **K_dot coefficient**: The coefficient of $\mathcal{U}$ in $\dot{K}$ is $(1/c)(1 + \alpha(1-r/a))$, NOT $(1/c)(\ell/\alpha + \alpha(1-r/a))$ as might be naively derived from $\dot{L}$. The L->K transformation changes the coefficient structure. Always verify against Eq. 75 directly.

2. **Numerical precision of $h - c$**: Always compute as $-2r^2\mathcal{U}/(h+c)$ rather than $\sqrt{c^2 - 2r^2\mathcal{U}} - c$.

3. **$1 - r/a$ stability**: Use $p_1 \sin K + p_2 \cos K$ (from Eq. 31) instead of computing $r/a$ separately.

4. **Cowell J2 sign**: The J2 perturbation acceleration is $-\nabla\mathcal{U}$ (force = $-$grad of potential energy). At the equator, J2 adds inward acceleration (extra attraction). Getting the sign wrong gives ~0.6 km error at 12 days.

5. **heyoka LLVM caching**: First build of an expression graph takes ~1.2s (LLVM JIT compilation). Subsequent builds with the same structure take ~9ms (per-process cache). The cache does not persist across Python sessions.

6. **Units**: The paper uses km and seconds throughout. This implementation follows the same convention (no normalization). This differs from the existing code in `src/astrodyn_core/propagation/geqoe/` which normalizes with $L=R_e$, $T=\sqrt{R_e^3/\mu}$.

7. **`var_ode_sys` state layout**: The 42-element augmented state stores the 6 base state variables followed by 36 STM entries in row-major order. Use `state[6:].reshape(6, 6)` to extract the STM.

8. **Thread safety**: Each `taylor_adaptive` instance is independent. Do not share instances across threads.
