# GEqOE Taylor Propagator with Automatic STM — Implementation Plan

## Goal

Build a high-performance orbit propagator using **Generalized Equinoctial Orbital Elements (GEqOE)** with the **heyoka.py** Taylor integrator. The propagator must:

1. Propagate the 6-element GEqOE state using adaptive-order Taylor stepping (AD-generated coefficients)
2. Automatically compute the State Transition Matrix (STM) via `var_ode_sys`
3. Support arbitrary conservative perturbations (initially J2-only, then extensible)
4. Provide dense output (polynomial evaluation at any sub-step time)
5. Match or exceed the performance of an existing hand-derived analytical Taylor-4 J2-only implementation

**Reference paper**: Baù, Hernando-Ayuso & Bombardelli (2021), "A generalization of the equinoctial orbital elements", Celestial Mechanics and Dynamical Astronomy, 133:50. All equation numbers below refer to this paper.

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
geqoe_taylor/
├── README.md
├── pyproject.toml
├── src/
│   └── geqoe_taylor/
│       ├── __init__.py
│       ├── constants.py              # Physical constants (mu, J2, Re)
│       ├── perturbations/
│       │   ├── __init__.py
│       │   ├── base.py               # PerturbationModel protocol
│       │   ├── j2.py                 # J2 zonal harmonic
│       │   └── j2_thirdbody.py       # J2 + Sun/Moon (later)
│       ├── rhs.py                    # GEqOE RHS builder (symbolic)
│       ├── integrator.py             # Wrapper: build/run heyoka integrators
│       ├── conversions.py            # Cartesian <-> GEqOE conversions (numerical, not symbolic)
│       └── utils.py                  # Helpers (Kepler solve for L<->K, etc.)
├── tests/
│   ├── test_conversions.py           # Round-trip Cartesian <-> GEqOE
│   ├── test_rhs_eval.py             # RHS evaluation against finite differences
│   ├── test_propagation.py          # Propagation accuracy vs reference
│   ├── test_stm.py                  # STM vs finite-difference Jacobian
│   └── test_paper_cases.py          # Reproduce Table 2/3 and Figs 3-7 from paper
└── examples/
    ├── propagate_leo.py             # Basic propagation example
    └── compare_cowell.py            # GEqOE vs Cartesian comparison
```

---

## Phase 1: Foundations (No heyoka yet)

### 1.1 Physical Constants (`constants.py`)

```python
MU = 398600.4354360959        # km^3/s^2 (from paper Eq. 59)
J2 = 1.08262617385222e-3      # (from paper Eq. 59)
RE = 6378.1366                # km (from paper Eq. 59)
A_J2 = MU * J2 * RE**2 / 2   # Convenience constant for J2 potential
```

### 1.2 Coordinate Conversions (`conversions.py`)

Implement in plain NumPy (not heyoka expressions). These run outside the integrator.

**Cartesian → GEqOE** (Section 3 of paper):
- Input: `r_vec` (3,), `v_vec` (3,), `mu`, `perturbation: PerturbationModel`
- Output: `(nu, p1, p2, K, q1, q2)`
- Steps: compute orbital frame vectors (Eq. 4), `r`, `rdot`, `h`, total energy `E` (using `perturbation.U_numeric`), `nu` (Eq. 16), `q1/q2` (Eq. 35-36), equinoctial frame `eX/eY/eZ` (Eq. 37), generalized angular momentum `c` (Eq. 6), generalized velocity `upsilon` (Eq. 7), generalized Laplace vector `g` (Eq. 10), `p1/p2` as projections of `g`, then `X/Y`, `sin K/cos K` (Eq. 39), `K = atan2(sinK, cosK)`
- **Important**: Accept `PerturbationModel` from day one (not a bare `U` function). This avoids refactoring conversions when moving from J2-only to general perturbations in Phase 6.

**GEqOE → Cartesian** (Section 4 of paper):
- Input: `(nu, p1, p2, K, q1, q2)`, `mu`, `perturbation: PerturbationModel`
- Output: `r_vec` (3,), `v_vec` (3,)
- Steps: `eX/eY` from `q1/q2` (Eq. 37), `a` from `nu` (Eq. 21), `alpha/beta` (Eq. 40), `X/Y` (Eq. 42), `r_vec = X*eX + Y*eY`, `r/rdot` (Eq. 31-32), `cos L/sin L` from `X/Y/r`, then `h` from `c` and `U` (Eq. 44), velocity components `Xdot/Ydot` (Eq. 43), `v_vec = Xdot*eX + Ydot*eY`

**Validation**: Use Table 2 → Table 3 from the paper (Appendix B). The paper provides step-by-step numerical values for case (a). Implement a test that reproduces these exact numbers to 10+ significant figures.

### 1.3 Perturbation Model Interface (`perturbations/base.py`)

```python
from typing import Protocol

class PerturbationModel(Protocol):
    """Interface for perturbation models.
    
    A model must provide:
    - U_expr(r, z, r_mag, t, pars): heyoka expression for potential energy U
    - U_numeric(r_vec, t): numeric evaluation of U for coordinate conversions
    - grad_U_numeric(r_vec, t): numeric gradient of U for coordinate conversions
    """
    
    def U_expr(self, x, y, z, r_mag, t, pars) -> "heyoka expression":
        """Return U as a heyoka symbolic expression.
        
        Args:
            x, y, z: heyoka expressions for Cartesian position components
            r_mag: heyoka expression for |r|
            t: heyoka time expression
            pars: dict mapping parameter names to par[i] indices
        """
        ...
    
    def U_numeric(self, r_vec, t) -> float:
        """Evaluate U numerically (for coordinate conversions)."""
        ...
    
    def grad_U_numeric(self, r_vec, t):
        """Evaluate grad(U) numerically (for coordinate conversions). Returns (3,) array."""
        ...
```

### 1.4 J2 Perturbation (`perturbations/j2.py`)

From Eq. 56 of the paper:

$$\mathcal{U} = -\frac{A}{r^3}(1 - 3\hat{z}^2)$$

where $A = GM J_2 r_e^2 / 2$ and $\hat{z} = z/r$.

Implement both symbolic (heyoka expressions) and numeric versions.

For the symbolic version, express $z$ in terms of equinoctial quantities using Eq. 57:
$$\hat{z} = \frac{2(Y q_2 - X q_1)}{r(1 + q_1^2 + q_2^2)}$$

where $X = r \cos L$, $Y = r \sin L$. Since we integrate $K$, compute $\cos L$ and $\sin L$ from Eq. 60.

**Key derived quantities needed in the RHS** (specific to J2-only, from Section 7.1):
- Total energy $\mathcal{E}$ is a first integral → $\dot{\nu} = 0$, $\dot{\mathcal{E}} = 0$
- $2\mathcal{U} - r F_r = -\mathcal{U}$ (derived in Section 7.1)
- $F_h = -6A \hat{z} \cos i / r^4$
- $w_h = I\hat{z}$ with $I = 3A\hat{z}(1 - q_1^2 - q_2^2)/(hr^3)$ (Eq. 58)

---

## Phase 2: Symbolic RHS Construction (`rhs.py`)

This is the core and hardest part. Build the GEqOE equations of motion as heyoka symbolic expressions.

### 2.1 State Variables and Parameters

```python
import heyoka as hy

def build_geqoe_system(perturbation_model, use_par=True):
    """Build the GEqOE ODE system as heyoka expressions.
    
    State: [nu, p1, p2, K, q1, q2]
    
    Returns:
        sys: list of (var, rhs) tuples for heyoka.taylor_adaptive
        state_vars: list of heyoka variables
        par_map: dict mapping parameter names to par[] indices
    """
    # State variables
    nu, p1, p2, K, q1, q2 = hy.make_vars("nu", "p1", "p2", "K", "q1", "q2")
    
    # Runtime parameters (changeable without recompilation)
    if use_par:
        mu = hy.par[0]
        # Additional params depend on perturbation model
        par_map = {"mu": 0}
        # Let perturbation model register its parameters starting at index 1
    else:
        mu = MU  # compile-time constant (faster but less flexible)
        par_map = {}
```

### 2.2 Intermediate Quantities from (nu, p1, p2, K, q1, q2)

Build these as heyoka expressions in order. Each depends only on state variables and parameters.

```
# Shape quantities
a = (mu / nu**2)**(1./3)                          # Eq. 21
g2 = p1**2 + p2**2                                 # Eq. 22
beta = hy.sqrt(1 - g2)                             # Eq. 40
alpha = 1 / (1 + beta)                             # Eq. 40

# Position from K (Eq. 42)
sinK = hy.sin(K)
cosK = hy.cos(K)
X = a * (alpha*p1*p2*sinK + (1 - alpha*p1**2)*cosK - p2)
Y = a * (alpha*p1*p2*cosK + (1 - alpha*p2**2)*sinK - p1)

# Orbital distance and radial velocity (Eq. 31-32)
r = a * (1 - p1*sinK - p2*cosK)
rdot = hy.sqrt(mu*a) / r * (p2*sinK - p1*cosK)

# True longitude trig functions
cosL = X / r
sinL = Y / r

# Equinoctial frame unit vectors (Eq. 37) — needed for z-component
gamma_inv = 1 / (1 + q1**2 + q2**2)
# eX = gamma_inv * [1-q1^2+q2^2, 2*q1*q2, -2*q1]  (but we only need z-components for zhat)
# eY = gamma_inv * [2*q1*q2, 1+q1^2-q2^2, 2*q2]

# z-hat (Eq. 57)
zhat = 2 * (Y*q2 - X*q1) * gamma_inv / r

# Generalized angular momentum (Eq. 23)
c = (mu**2 / nu)**(1./3) * hy.sqrt(1 - g2)

# Physical angular momentum h from c and U (Eq. 44)
# Need U as function of position — this comes from perturbation_model
# h = sqrt(c**2 - 2*r**2*U)
```

### 2.3 Perturbation-Dependent Terms

For the **J2-only** case (Section 7.1 of paper), the RHS simplifies because:
- $\dot{\mathcal{E}} = 0$ (energy is conserved) → $\dot{\nu} = 0$
- $2\mathcal{U} - rF_r = -\mathcal{U}$

The simplified J2-only equations are given explicitly in Section 7.1. **Implement this case first.**

For the **general** case (Eqs. 45-51), the full terms $\dot{\mathcal{E}}$ (Eq. 46), $2\mathcal{U} - rF_r$, and $w_h$ must be computed from the perturbation model's symbolic expressions.

### 2.4 Assembling the ODEs

For **J2-only** (from Section 7.1), the system is:

```
nu_dot = 0   # (energy integral)

p1_dot = p2 * ((h-c)/r**2 - I*zhat) - (1/c) * (X/a + 2*p2) * U_val

p2_dot = p1 * (I*zhat - (h-c)/r**2) + (1/c) * (Y/a + 2*p1) * U_val

K_dot  = w/r + (h-c)/r**2 - I*zhat - (1/c) * (ell/alpha + alpha*(1-r/a)) * U_val
         # where w = sqrt(mu/a), ell = c**2/mu
         # Note: this is Eq. 75 with E_dot=0 and 2U-rFr=-U

q1_dot = -I * Y / r

q2_dot = -I * X / r
```

where:
- `U_val` = $\mathcal{U}(r, \hat{z})$ from the J2 model (Eq. 56)
- `I` = $3A\hat{z}(1 - q_1^2 - q_2^2)/(hr^3)$ (Eq. 58)
- `h` = $\sqrt{c^2 - 2r^2\mathcal{U}}$ (Eq. 44)

### 2.5 Important Implementation Notes

1. **Substitution hints from Section 7.1 of the paper**: For numerical stability in the $\dot{K}$ expression, use:
   - $1 - r/a = p_1 \sin K + p_2 \cos K$ (from Eq. 31)
   - $r\dot{r}/c = (p_2 \sin K - p_1 \cos K)/\beta$ (derived from Eq. 32 and definitions)

2. **The quantity $h - c$**: This is small (of order $J_2$). Computing it as `sqrt(c**2 - 2*r**2*U) - c` loses precision. Instead, use:
   $$h - c = \frac{c^2 - 2r^2\mathcal{U} - c^2}{h + c} = \frac{-2r^2\mathcal{U}}{h + c}$$
   In heyoka: `h_minus_c = -2*r**2*U_val / (h + c)`

3. **The semi-latus rectum**: $\ell = c^2/\mu = a(1 - g^2)$ (Eq. 19-20). Use whichever avoids cancellation.

4. **Avoid dividing by small quantities**: The expressions involve $1/c$, $1/r$, $1/h$. These are all well-behaved for non-degenerate orbits, but watch for $\beta = \sqrt{1 - g^2}$ near $g = 1$ (highly eccentric).

---

## Phase 3: Build heyoka Integrators (`integrator.py`)

### 3.1 State-Only Integrator

```python
import heyoka as hy

def build_state_integrator(perturbation_model, ic, t0=0.0, tol=1e-15, par_values=None):
    """Build a state-only GEqOE Taylor integrator.
    
    Args:
        perturbation_model: PerturbationModel instance
        ic: initial conditions [nu, p1, p2, K, q1, q2]
        t0: initial time
        tol: integrator tolerance
        par_values: list of parameter values matching par_map
    
    Returns:
        ta: heyoka.taylor_adaptive integrator
        par_map: dict mapping names to par[] indices
    """
    sys, state_vars, par_map = build_geqoe_system(perturbation_model)
    
    ta = hy.taylor_adaptive(
        sys,
        state=list(ic),
        time=t0,
        pars=par_values or [],
        tol=tol,
        compact_mode=False  # OK for 6-DOF
    )
    return ta, par_map
```

### 3.2 State + STM Integrator (42 DOF)

```python
def build_stm_integrator(perturbation_model, ic, t0=0.0, tol=1e-15, par_values=None):
    """Build a GEqOE integrator with automatic 1st-order variational equations (STM).
    
    The augmented system has 6 + 36 = 42 state variables.
    """
    sys, state_vars, par_map = build_geqoe_system(perturbation_model)
    
    # Generate variational equations automatically
    vsys = hy.var_ode_sys(sys, args=hy.var_args.vars, order=1)
    
    # Initial conditions: state + identity matrix (flattened)
    ic_aug = list(ic) + [1 if i == j else 0 for i in range(6) for j in range(6)]
    
    ta = hy.taylor_adaptive(
        vsys,
        state=ic_aug,
        time=t0,
        pars=par_values or [],
        tol=tol,
        compact_mode=True  # Essential for 42-DOF variational system
    )
    return ta, par_map
```

### 3.3 Propagation Interface

```python
def propagate(ta, t_final, max_delta_t=None, dense=False):
    """Propagate to t_final, optionally with step clamping and dense output.
    
    Args:
        ta: heyoka integrator
        t_final: target time
        max_delta_t: maximum step size (None = unclamped)
        dense: if True, store Taylor coefficients for dense output
    
    Returns:
        times: list of step boundary times
        states: list of states at step boundaries
        tc: Taylor coefficients if dense=True (for sub-step evaluation)
    """
    times = [ta.time]
    states = [ta.state.copy()]
    tcs = [] if dense else None
    
    while ta.time < t_final:
        if max_delta_t is not None:
            remaining = t_final - ta.time
            step_limit = min(max_delta_t, remaining)
            outcome = ta.step(max_delta_t=step_limit, write_tc=dense)
        else:
            # Let heyoka choose the step adaptively
            # Clamp to not overshoot t_final
            remaining = t_final - ta.time
            outcome = ta.step(max_delta_t=remaining, write_tc=dense)
        
        times.append(ta.time)
        states.append(ta.state.copy())
        if dense:
            tcs.append(ta.tc.copy())
    
    return times, states, tcs


def extract_stm(state_aug):
    """Extract the 6x6 STM from the 42-element augmented state.
    
    The variational state is stored after the 6 base state variables.
    The layout follows heyoka's var_ode_sys convention.
    """
    y = state_aug[:6]
    phi = state_aug[6:].reshape(6, 6)
    return y, phi
```

---

## Phase 4: Validation Tests

### 4.1 Conversion Round-Trip (`test_conversions.py`)

- For each of the 4 initial conditions in Table 1 of the paper:
  1. Convert Keplerian → Cartesian (Table 2)
  2. Convert Cartesian → GEqOE (Table 3)
  3. Convert GEqOE → Cartesian
  4. Assert round-trip error < 1e-12 in position and velocity
- For case (a), verify intermediate values against Appendix B step-by-step values

### 4.2 RHS Evaluation (`test_rhs_eval.py`)

- At a given state, evaluate the heyoka RHS by doing a tiny step ($\Delta t = 10^{-10}$ s) and computing $(y_1 - y_0)/\Delta t$
- Compare against a separate numerical implementation of Eqs. 45-51 (plain NumPy)
- Agreement to at least 8 significant figures

### 4.3 Propagation Accuracy (`test_propagation.py`)

Reproduce Figures 3 and 5 from the paper:

- Case (a): LEO, J2 only, 12-day propagation
  - Reference final state from Appendix C:
    ```
    x_f = -5398.929377366906 km    xdot_f = 2.214482567493 km/s
    y_f = -390.257240638229 km     ydot_f = -6.845637008953 km/s
    z_f = -4693.719111636971 km    zdot_f = -1.977748618717 km/s
    ```
  - These are obtained with Dromo(PC) + DOPRI5(4)7FM at tolerance 1e-13

- Cases (b), (c), (d): same but with J2 + third-body (Phase 6)

**Test procedure**: Propagate with GEqOE+heyoka, convert final state to Cartesian, compare against reference. Position error should be < 1e-6 km for default tolerance.

### 4.4 STM Validation (`test_stm.py`)

For a single step of size $h$:

1. Propagate nominal state → $\mathbf{y}(t_0 + h)$, extract $\Phi$ from variational integrator
2. For each component $j = 0..5$:
   - Perturb: $\mathbf{y}_0^+ = \mathbf{y}_0$, $y_{0,j}^+ += \delta$
   - Perturb: $\mathbf{y}_0^- = \mathbf{y}_0$, $y_{0,j}^- -= \delta$
   - Propagate both for same $h$
   - FD column: $\Phi_{:,j}^{FD} = (\mathbf{y}^+ - \mathbf{y}^-) / (2\delta)$
3. Assert $\|\Phi - \Phi^{FD}\| / \|\Phi\| < 10^{-8}$

Use $\delta = 10^{-7}$ (relative to element magnitude) for central differences.

**Important**: Use the same integrator tolerance for nominal and perturbed propagations. Build fresh integrator instances for each perturbation (do NOT reuse — heyoka integrators are stateful).

### 4.5 Paper Reproduction (`test_paper_cases.py`)

Long-term propagation tests matching the paper's figures:

- Case (a), 365 days, J2 only, 1-minute fixed step
- Verify position error growth matches Figure 4 (top panel) qualitatively
- The GEqOE curve should show ~1e-4 km error at 365 days with 60s step

---

## Phase 5: Dense Output and Post-Processing

### 5.1 Dense Output for Sub-Step Evaluation

After `ta.step(write_tc=True)`, the Taylor coefficients are available in `ta.tc`. For a step from $t_n$ to $t_{n+1}$ with step size $h$:

$$y_i(t_n + \tau) = \sum_{k=0}^{p} c_{i,k} \tau^k, \quad \tau \in [0, h]$$

where $p$ is the Taylor order (determined adaptively by heyoka). The coefficients $c_{i,k}$ are stored in `ta.tc` with layout: variable index varies fastest, then coefficient order.

```python
def dense_eval(tc, order, n_vars, tau):
    """Evaluate dense output polynomial at time offset tau.
    
    tc: Taylor coefficients from ta.tc (flat array)
    order: Taylor order (ta.order)
    n_vars: number of state variables (6 or 42)
    tau: time offset from step start
    """
    # tc layout: [c_{0,0}, c_{1,0}, ..., c_{n-1,0}, c_{0,1}, ..., c_{n-1,p}]
    result = np.zeros(n_vars)
    for k in range(order, -1, -1):  # Horner's method
        result = result * tau + tc[k*n_vars:(k+1)*n_vars]
    return result
```

### 5.2 L Recovery from K

For output or comparison purposes, compute $L$ from $K$:
$$L = K + p_1 \cos K - p_2 \sin K$$
This is explicit (Eq. 30 read left-to-right).

### 5.3 Output Coordinate Conversion

After propagation, convert GEqOE state to Cartesian using the numerical conversion from Phase 1.2. This is done outside heyoka.

---

## Phase 6: Generalization to Full Perturbations (Later)

### 6.1 Non-Conservative Forces

For drag, SRP, or empirical accelerations, the force $\mathbf{P}$ appears through projections $P_r$, $P_f$ in the RHS (Eqs. 46-49). These are *not* absorbed into $\mathcal{U}$.

In heyoka, express $\mathbf{P}$ components in the orbital frame:
- $P_r = \mathbf{P} \cdot \mathbf{e}_r$ where $\mathbf{e}_r = \mathbf{e}_X \cos L + \mathbf{e}_Y \sin L$
- $P_f = \mathbf{P} \cdot \mathbf{e}_f$ where $\mathbf{e}_f = \mathbf{e}_Y \cos L - \mathbf{e}_X \sin L$

The full equations (Eqs. 45-51) include $\dot{\mathcal{E}}$ (Eq. 46) which is no longer zero:
$$\dot{\mathcal{E}} = \mathcal{U}_t + \dot{r}P_r + \frac{h}{r}P_f$$

And $\dot{\nu} \neq 0$ when energy is not conserved (Eq. 45):
$$\dot{\nu} = -3\left(\frac{\nu}{\mu^2}\right)^{1/3}\dot{\mathcal{E}}$$

### 6.2 Third-Body Gravity

Following the paper's Section 7.2 recommendation: treat third-body forces as non-conservative ($\mathbf{P}$), **not** absorbed into $\mathcal{U}$, for Earth-bound orbits. This avoids numerical instabilities.

Third-body positions come from ephemerides. In heyoka, use `hy.time` as the independent variable and either:
- Analytical ephemerides (e.g., `hy.model.vsop2013` for Sun, simple lunar theory for Moon)
- Chebyshev polynomial approximations of JPL ephemerides (pre-computed for the propagation window, loaded as heyoka expressions)

### 6.3 Higher-Order Geopotential

For $J_n$ zonal harmonics beyond $J_2$: extend the potential $\mathcal{U}$ with additional Legendre polynomial terms. The symbolic expression becomes longer but the structure is unchanged. Only $J_2$ goes into $\mathcal{U}$; higher-order terms can be treated as non-conservative to avoid complexity (at the cost of slower element evolution).

---

## Implementation Order (What To Build First)

**Step 1**: `constants.py` + `conversions.py` + `test_conversions.py`
- Pure NumPy, no heyoka dependency
- Validate against Appendix B of the paper
- This catches any misunderstanding of the formulation early

**Step 2**: `perturbations/base.py` + `perturbations/j2.py` (numeric methods only)
- J2 potential evaluation in NumPy
- Used by conversions for the U-dependent terms

**Step 3**: `rhs.py` — J2-only symbolic RHS
- Start with the simplified J2-only equations from Section 7.1
- Build heyoka expressions step by step
- Test by constructing an integrator and checking a single tiny step

**Step 4**: `integrator.py` — state-only propagator
- Build and run the Taylor integrator
- Propagate case (a) for 12 days
- Compare final Cartesian state against Appendix C reference
- This is the **first milestone**: if this matches, the RHS is correct

**Step 5**: `test_stm.py` + STM integrator
- Add `var_ode_sys` wrapper
- Validate STM against finite differences
- **Second milestone**: automatic STM that matches FD to ~1e-8

**Step 6**: Dense output + helper utilities
- Taylor coefficient extraction
- Sub-step evaluation

**Step 7** (later): General perturbations, third-body, non-conservative forces

---

## Dependencies

```
heyoka >= 6.0.0    # Taylor integrator with var_ode_sys support
numpy >= 1.24
pytest
```

Install: `pip install heyoka numpy pytest`

---

## Key Numerical Values for Testing

From the paper, case (a) — LEO circular i=45°:

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
L  = 0 rad  →  K = 0 rad (since L=K when p1=0, p2 small, at L=0)
q1 = 0
q2 = 0.41421
```

**Reference final state after 12 days, J2 only** (Appendix C):
```
r_f = [-5398.929377366906, -390.257240638229, -4693.719111636971] km
v_f = [2.214482567493, -6.845637008953, -1.977748618717] km/s
```

---

## Pitfalls to Watch For

1. **heyoka expression tree depth**: The GEqOE RHS involves deeply nested expressions. If compilation is extremely slow (>5 minutes), try `compact_mode=True` even for the 6-DOF system.

2. **Numerical precision of $h - c$**: Always compute as $-2r^2\mathcal{U}/(h+c)$ rather than $\sqrt{c^2 - 2r^2\mathcal{U}} - c$.

3. **Initial K vs L**: At $t=0$, compute $K_0$ from $L_0$ by solving the Kepler equation (Eq. 30) with Newton-Raphson in the conversion routine. For the paper's test cases where $L_0 = 0$ and $p_1 = 0$, $K_0 \approx 0$ to high precision.

4. **Units**: The paper uses km and seconds throughout. Keep these units.

5. **heyoka `var_ode_sys` state layout**: The augmented state for first-order variational equations has the base state in positions 0..5, then the 36 STM entries. Use `ta.get_vslice(order=1, component=i)` to extract columns of the STM, or simply reshape `state[6:]` into `(6,6)`. Verify the convention (row-major vs column-major) against finite differences.

6. **Step clamping**: When reproducing paper figures with fixed step sizes, use `ta.step(max_delta_t=h)`. heyoka may take shorter steps than `h` if the Taylor series doesn't converge, but it will never exceed `h`.

7. **Thread safety**: Each `taylor_adaptive` instance is independent. Do not share instances across threads.
