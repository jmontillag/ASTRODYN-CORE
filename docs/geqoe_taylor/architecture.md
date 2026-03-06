# GEqOE Taylor Propagator — Architecture & Design Decisions

Complete documentation of all architectural decisions, simplifications, and mathematical derivations in the `src/astrodyn_core/geqoe_taylor/` implementation.

**Reference paper**: Baù, Hernando-Ayuso & Bombardelli (2021), "A generalization of the equinoctial orbital elements", *Celest. Mech. Dyn. Astr.* 133:50. All equation numbers refer to this paper unless stated otherwise.

---

## Table of Contents

1. [State Vector Choice: K Instead of L](#1-state-vector-choice-k-instead-of-l)
2. [Units and Physical Constants](#2-units-and-physical-constants)
3. [Perturbation Model Protocol](#3-perturbation-model-protocol)
4. [Three RHS Code Paths](#4-three-rhs-code-paths)
5. [Shared Intermediate Quantities](#5-shared-intermediate-quantities)
6. [J2-Only Fast Path](#6-j2-only-fast-path)
7. [Zonal Fast Path](#7-zonal-fast-path)
8. [General Path (Full Equations)](#8-general-path-full-equations)
9. [Path Selection Logic](#9-path-selection-logic)
10. [Runtime Parameters](#10-runtime-parameters)
11. [Coordinate Conversions](#11-coordinate-conversions)
12. [Integrator Wrappers](#12-integrator-wrappers)
13. [Perturbation Implementations](#13-perturbation-implementations)
14. [Numerical Stability Techniques](#14-numerical-stability-techniques)
15. [Sign Conventions](#15-sign-conventions)
16. [Validation Strategy](#16-validation-strategy)
17. [Known Pitfalls and Lessons Learned](#17-known-pitfalls-and-lessons-learned)

---

## 1. State Vector Choice: K Instead of L

**State vector**: `[nu, p1, p2, K, q1, q2]`

| Element | Symbol | Description |
|---------|--------|-------------|
| `nu`    | ν      | Generalized mean motion (rad/s) |
| `p1`    | p₁     | Eccentricity-like (projection of generalized Laplace vector onto eY) |
| `p2`    | p₂     | Eccentricity-like (projection onto eX) |
| `K`     | K      | Generalized eccentric longitude |
| `q1`    | q₁     | Orientation (eₕ·eₓ / (1 + eₕ·eᵤ)) |
| `q2`    | q₂     | Orientation (-eₕ·eᵧ / (1 + eₕ·eᵤ)) |

**Why K instead of L?** The paper's original formulation uses the generalized mean longitude L as the fast variable. Computing intermediate quantities (r, ṙ, X, Y, cosL, sinL) from L requires solving the **generalized Kepler equation** (Eq. 30):

```
L = K + p₁ cos K - p₂ sin K
```

This is an implicit equation for K given L, requiring iterative Newton-Raphson — **incompatible with heyoka's symbolic expression system**. heyoka builds a DAG (directed acyclic graph) of expressions that must be composed from elementary operations (+, -, ×, ÷, sin, cos, sqrt, pow). An iterative solver cannot be expressed as a finite symbolic expression tree.

**Solution**: Integrate K directly. The paper provides K̇ explicitly in Eq. 75. All RHS quantities (r, ṙ, X, Y) can be computed from K using only elementary functions. If L is ever needed for output, it is recovered via the **forward** Kepler equation (Eq. 30), which is explicit.

**Files**: `rhs.py` (all three code paths use K), `utils.py` (provides `K_to_L` and `solve_kepler_gen` for L↔K conversion).

---

## 2. Units and Physical Constants

**Units**: Physical (km, s) throughout — no normalization.

This differs from the existing hand-derived propagator in `src/astrodyn_core/propagation/geqoe/` which normalizes with L=R_e, T=sqrt(R_e³/μ). The choice of physical units was made to:
- Match the paper's convention (Eq. 59)
- Avoid normalization/denormalization overhead at boundaries
- Produce directly interpretable state values

**Constants** (`constants.py`):

| Constant | Value | Source |
|----------|-------|--------|
| `MU` | 398600.4354360959 km³/s² | Paper Eq. 59 |
| `J2` | 1.08262617385222e-3 | Paper Eq. 59 |
| `RE` | 6378.1366 km | Paper Eq. 59 |
| `A_J2` | MU·J2·RE²/2 | Derived convenience constant |
| `J3`–`J6` | EGM2008 unnormalized | Standard geopotential |
| `GM_SUN` | 1.32712440041279419e11 km³/s² | IAU |
| `GM_MOON` | 4902.800066 km³/s² | IAU |
| `AU_KM` | 149597870.7 km | IAU |
| `OBLIQUITY_J2000` | 23.439291111° | J2000 ecliptic obliquity |
| `JD_J2000` | 2451545.0 | J2000 epoch Julian date |

---

## 3. Perturbation Model Protocol

**File**: `perturbations/base.py`

Two protocol levels, using Python `Protocol` with `runtime_checkable`:

### 3.1 PerturbationModel (Minimal)

Required for all code paths:

```python
class PerturbationModel(Protocol):
    def U_expr(self, x, y, z, r_mag, t, pars: dict): ...
    def U_numeric(self, r_vec: np.ndarray, t: float = 0.0) -> float: ...
```

- `U_expr`: Returns the disturbing potential U as a heyoka symbolic expression. Arguments are heyoka expressions (or floats) for Cartesian coordinates, scalar distance, time, and a parameter dict.
- `U_numeric`: Returns U as a Python float for a given NumPy position vector. Used by coordinate conversions (which run outside the integrator and need numerical evaluation).

### 3.2 GeneralPerturbationModel (Extended)

Required for the general code path (non-J2-only):

```python
class GeneralPerturbationModel(PerturbationModel, Protocol):
    is_conservative: bool
    is_time_dependent: bool
    def grad_U_expr(self, x, y, z, r_mag, t, pars: dict) -> tuple: ...
    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars: dict) -> tuple: ...
    def U_t_expr(self, x, y, z, r_mag, t, pars: dict): ...
```

- `grad_U_expr`: Returns `(dU/dx, dU/dy, dU/dz)` as heyoka expressions.
- `P_expr`: Returns non-conservative acceleration `(Px, Py, Pz)` in Cartesian. Zero for conservative models.
- `U_t_expr`: Returns ∂U/∂t. Zero for time-independent models.
- `is_conservative`: True if P = 0.
- `is_time_dependent`: True if U or P depends explicitly on time.

Optional extensions used by the continuous-thrust stack:

- `parameter_defaults()`: exposes runtime parameters that should be mapped to
  `hy.par[i]`
- `requires_mass = True`: flags models that need the 7-state mass-augmented
  GEqOE system
- `mass_flow_expr(...)`: returns the mass derivative for the augmented system
- `P_and_mass_flow_expr(...)`: optional fused path used to avoid rebuilding the
  same thrust expressions twice

### 3.3 Additional Flags

Models may set optional flags that control code path selection:

| Flag | Default | Meaning |
|------|---------|---------|
| `_zonal_fast_path` | False | Model supports `zonal_quantities()` interface |
| `_force_general` | False | Model bypasses J2-only path even if conservative + time-independent |

### 3.4 Design Rationale

**Why separate U and P?** The GEqOE equations distinguish between conservative and non-conservative perturbations at a fundamental level:

- **Conservative (U)**: Enters the definition of the physical angular momentum h = sqrt(c² - 2r²U) and the generalized angular momentum c. This means U affects the non-osculating reference ellipse.
- **Non-conservative (P)**: Enters as external forcing only (appears in Ė, F_r, F_f, F_h projections). Does not affect h or c definitions.

Mixing a conservative perturbation into P would give h = sqrt(c²) = c, losing the U contribution to h. This degrades accuracy for perturbations that affect the orbit shape (geopotential), though it's acceptable for small perturbations (third-body for short arcs).

**Why `U_numeric` separate from `U_expr`?** Coordinate conversions (`cart2geqoe`, `geqoe2cart`) run outside the integrator and need numerical evaluation of U to compute h and c. They cannot use heyoka expressions because they operate on NumPy arrays.

---

## 4. Three RHS Code Paths

**File**: `rhs.py`

The system provides three code paths, auto-selected based on the perturbation model's properties:

| Path | When | Simplifications | Parameters |
|------|------|-----------------|------------|
| **J2-only** | `is_conservative`, not `is_time_dependent`, not `_force_general` | Ė=0, ν̇=0, 2U-rF_r=-U, inline J2 potential | hy.par[0]=μ, hy.par[1]=A |
| **Zonal fast** | `_zonal_fast_path`, `is_conservative`, not `is_time_dependent` | Ė=0, ν̇=0, Euler identity for 2U-rF_r, no Cartesian | hy.par[0]=μ |
| **General** | Everything else | Full Eqs. 45-51 | hy.par[0]=μ (+ model params) |

All three paths share `_build_intermediates()` for computing orbital geometry from the state vector.

The continuous-thrust mass-augmented builder (`build_geqoe_mass_system`) reuses
the general path directly and simply appends the scalar mass state to it. There
is intentionally no separate thrust-specific force decomposition.

---

## 5. Shared Intermediate Quantities

**Function**: `_build_intermediates(mu, nu, p1, p2, K, q1, q2)` in `rhs.py`

Computes the following heyoka expressions, used by all code paths:

| Quantity | Expression | Reference |
|----------|------------|-----------|
| sinK, cosK | sin(K), cos(K) | — |
| a | (μ/ν²)^(1/3) | Eq. 21: generalized semi-major axis |
| g² | p₁² + p₂² | Eq. 22 |
| β | sqrt(1 - g²) | Eq. 40 |
| α | 1/(1 + β) | Eq. 40 |
| X | a·(α·p₁·p₂·sinK + (1-α·p₁²)·cosK - p₂) | Eq. 42: in-plane position (eX component) |
| Y | a·(α·p₁·p₂·cosK + (1-α·p₂²)·sinK - p₁) | Eq. 42: in-plane position (eY component) |
| r | a·(1 - p₁·sinK - p₂·cosK) | Eq. 31: orbital distance |
| cosL, sinL | X/r, Y/r | True longitude trig |
| γ | 1 + q₁² + q₂² | Orientation factor |
| γ⁻¹ | 1/γ | — |
| ẑ | 2(Y·q₂ - X·q₁)·γ⁻¹/r | Eq. 57: z/r via equinoctial frame |
| c | (μ²/ν)^(1/3)·β | Eq. 23: generalized angular momentum |
| w | sqrt(μ/a) | Keplerian velocity scale |
| ṙ | w·a/r·(p₂·sinK - p₁·cosK) | Eq. 32: radial velocity |

**Key insight about ẑ**: The quantity ẑ = z/r is computed without explicitly constructing Cartesian z. Instead, it uses the equinoctial frame projection:

```
z = X·eZ_X + Y·eZ_Y = X·(2q₁/γ) + Y·(-2q₂/γ) → z/r is built from X, Y, q₁, q₂ directly
```

Wait — more precisely, from the frame definition:
```
z = r_vec · ê_z = (X·eX + Y·eY) · ê_z
eX_z = -2q₁/γ,  eY_z = 2q₂/γ
z = X·(-2q₁/γ) + Y·(2q₂/γ) = 2(Yq₂ - Xq₁)/γ
ẑ = z/r = 2(Yq₂ - Xq₁)/(γr)
```

This avoids constructing the full 3D position vector for the J2-only and zonal paths.

---

## 6. J2-Only Fast Path

**Function**: `_build_j2_only_system()` in `rhs.py`

### 6.1 Simplifications

For J2-only (conservative, time-independent, degree-2 zonal):

1. **Ė = 0, ν̇ = 0**: The generalized energy E is a first integral of the J2-perturbed motion (Section 7.1). This means the generalized mean motion ν is constant.

2. **2U - rF_r = -U**: For J2 (degree n=2), the Euler homogeneity theorem gives:
   - r·∇U = -(n+1)·U = -3U (since U is homogeneous of degree -(n+1) in r)
   - 2U - rF_r = 2U + r·∇U·ê_r - r·P_r
   - For conservative (P=0): 2U + r·∇U = 2U - 3U = -U

3. **No Cartesian coordinates**: The potential U = -A/r³·(1 - 3ẑ²) and all force projections are expressed directly in orbital quantities (r, ẑ, X, Y, sinL, cosL).

4. **Inline J2 potential**: U is hard-coded, not obtained from the perturbation model's `U_expr`.

### 6.2 Out-of-Plane Force (J2-Only)

For J2, the quantity `I` (Eq. 58) encapsulates the out-of-plane force:

```
I = 3A·ẑ·δ / (h·r³)
```

where δ = 1 - q₁² - q₂². The angular rates are:
```
w_h = I·ẑ
q̇₁ = -I·sinL    (Eq. 50 simplified for J2)
q̇₂ = -I·cosL    (Eq. 51 simplified for J2)
```

Note: This simplified form of q̇ is equivalent to the general form `q̇₁ = (γ/2)·w_Y`, `q̇₂ = (γ/2)·w_X` when specialized to J2.

### 6.3 Assembled ODEs

```
ν̇  = 0
ṗ₁ = p₂·(d - w_h) - (1/c)·(X/a + 2p₂)·U
ṗ₂ = p₁·(w_h - d) + (1/c)·(Y/a + 2p₁)·U
K̇  = w/r + d - w_h - (1/c)·(1 + α·(1 - r/a))·U
q̇₁ = -I·sinL
q̇₂ = -I·cosL
```

where d = (h - c)/r².

### 6.4 Parameters

Uses two runtime parameters: `hy.par[0]` = μ, `hy.par[1]` = A_J2. Both are runtime-changeable without expression recompilation.

---

## 7. Zonal Fast Path

**Function**: `_build_zonal_system()` in `rhs.py`
**Model**: `perturbations/zonal.py`

### 7.1 Motivation

The zonal fast path generalizes the J2-only path to arbitrary zonal harmonics (J2 through Jn) while avoiding the overhead of the general path (Cartesian coordinates, frame vectors, 3D gradient). It exploits the axial symmetry of zonal harmonics.

### 7.2 Zonal Potential

The zonal potential for degree n is:

```
U_n = C_n / r^(n+1) · P_n(ẑ)
```

where C_n = μ·J_n·R_e^n and P_n is the Legendre polynomial. The total zonal potential is:

```
U = Σ_n U_n
```

### 7.3 Legendre Polynomial Computation

**Bonnet recurrence** (works with both heyoka expressions and floats):

```python
def _legendre_P(n, x):
    if n == 0: return 1.0
    if n == 1: return x
    P_prev, P_curr = 1.0, x
    for k in range(2, n + 1):
        P_next = ((2k-1) * x * P_curr - (k-1) * P_prev) / k
        P_prev, P_curr = P_curr, P_next
    return P_curr
```

**Differentiated Bonnet recurrence** (simultaneous P_n and P'_n):

```python
def _legendre_P_and_deriv(n, x):
    # ...base cases...
    for k in range(2, n + 1):
        P_next  = ((2k-1) * x * P_curr  - (k-1) * P_prev)  / k
        dP_next = ((2k-1) * (P_curr + x * dP_curr) - (k-1) * dP_prev) / k
```

This builds the derivative P'_n alongside P_n in a single pass, avoiding a separate differentiation step.

### 7.4 The `zonal_quantities()` Interface

The `ZonalPerturbation` class provides a dedicated method for the fast path:

```python
def zonal_quantities(self, r, zhat) -> (U, dU_dzhat, euler_term):
```

Returns three quantities in a single Legendre pass:
- **U**: Total zonal potential
- **dU/dẑ**: Derivative of U with respect to ẑ (needed for F_h)
- **euler_term**: Σ_n (1-n)·U_n (Euler identity for 2U - rF_r)

### 7.5 Key Mathematical Simplifications

#### 7.5.1 Euler Homogeneity for 2U - rF_r

For a zonal harmonic of degree n, U_n is homogeneous of degree -(n+1) in (x,y,z). By Euler's theorem for homogeneous functions:

```
r · ∇U_n = -(n+1) · U_n
```

Therefore:
```
2U_n - rF_{r,n} = 2U_n + r·(∇U_n · ê_r) = 2U_n + (1/r)·r·∇U_n = 2U_n - (n+1)U_n = (1-n)·U_n
```

This generalizes the J2-only result: for n=2, (1-2)·U = -U. For n=3, -2U₃. For n=4, -3U₄.

The combined Euler term is simply: `Σ_n (1-n)·U_n`.

**Impact**: Eliminates the need to compute the full 3D gradient for the `2U - rF_r` term in the p-dot and K-dot equations.

#### 7.5.2 r·eZ = 0 and the Out-of-Plane Force

The out-of-plane force component F_h = F · eZ (projection of total force onto orbit normal) is needed for w_X, w_Y, w_h, and ultimately the q-dot equations. For a conservative potential F = -∇U, computing F_h naively requires the full 3D gradient.

**Key identity**: The position vector r lies in the orbital plane (spanned by eX and eY), so:

```
r · eZ = 0
```

This is true by construction: r = X·eX + Y·eY (Eq. 41).

**Derivation of F_h from dU/dẑ**: The potential U depends on position only through r and ẑ = z/r (for zonal harmonics). The out-of-plane force is:

```
F_h = -∇U · eZ = -(∂U/∂r)(∂r/∂r_vec · eZ) - (∂U/∂ẑ)(∂ẑ/∂r_vec · eZ)
```

For the ∂r term: ∂r/∂r_vec = r_vec/r, so (∂r/∂r_vec)·eZ = r·eZ/r = 0.

For the ∂ẑ term: ẑ = z/r = (r_vec · ê_z)/r, so:
```
∂ẑ/∂r_vec = ê_z/r - z·r_vec/r³ = (ê_z - ẑ·ê_r)/r
```

Projecting onto eZ:
```
(∂ẑ/∂r_vec) · eZ = (ê_z · eZ - ẑ·(ê_r · eZ)) / r
```

Since ê_r = r_vec/r = (X·eX + Y·eY)/r lies in the orbital plane, ê_r · eZ = 0. And ê_z · eZ = eZ_z = (1 - q₁² - q₂²)/γ = δ/γ. Therefore:

```
F_h = -(dU/dẑ) · (δ/γ) / r
```

**Impact**: The out-of-plane force is computed directly from dU/dẑ (a scalar derivative already available from the Legendre recurrence), without any 3D gradient computation.

#### 7.5.3 Combined Simplification

The zonal fast path needs only `(U, dU/dẑ, euler_term)` from the perturbation model — three scalar quantities computed in a single Legendre polynomial pass. It avoids:
- Constructing 3D Cartesian position (x, y, z)
- Computing equinoctial frame vectors (eX, eY, eZ as 3-vectors)
- Building 3D gradient expressions (∂U/∂x, ∂U/∂y, ∂U/∂z)
- Computing Cartesian velocity
- Projecting forces onto orbital frame unit vectors

### 7.6 Assembled ODEs (Zonal)

```
ν̇  = 0
ṗ₁ = p₂·(d - w_h) + (1/c)·(X/a + 2p₂)·euler_val
ṗ₂ = p₁·(w_h - d) - (1/c)·(Y/a + 2p₁)·euler_val
K̇  = w/r + d - w_h + (1/c)·(1 + α·(1 - r/a))·euler_val
q̇₁ = (γ/2)·w_Y
q̇₂ = (γ/2)·w_X
```

where:
```
δ = 2 - γ = 1 - q₁² - q₂²
F_h = -dU_dẑ · δ / (γ·r)
w_X = (X/h)·F_h
w_Y = (Y/h)·F_h
w_h = w_X·q₁ - w_Y·q₂
d = (h - c) / r²
```

**Sign difference from J2-only**: In the p-dot and K-dot equations, the J2-only path uses `-U` (the 2U-rFr simplification gives -U when 2U-rFr = -U). The zonal path uses `+euler_val` where euler_val = Σ(1-n)·U_n. For n=2, euler_val = -U, so the signs agree. But for higher n, the euler_val includes contributions from all harmonics.

### 7.7 General Path Fallback

The `ZonalPerturbation` class also provides the standard interface (`U_expr`, `grad_U_expr`, `P_expr`, `U_t_expr`) for compatibility with the general path. This is used when a `ZonalPerturbation` is wrapped in a `CompositePerturbation`.

The gradient is computed via `hy.diff_tensors` + `hy.subs`:
1. Build U with placeholder variables `_zx, _zy, _zz`
2. Auto-differentiate: `dt = hy.diff_tensors([U], [_zx, _zy, _zz], diff_order=1)`
3. Cache the gradient expressions
4. On each call, substitute actual coordinate expressions via `hy.subs`

This is lazy-evaluated (`_ensure_cart_grad()`) — the gradient is only built if the general path actually needs it.

---

## 8. General Path (Full Equations)

**Function**: `_build_general_system()` in `rhs.py`

### 8.1 When Used

The general path handles arbitrary perturbations, including:
- Non-conservative forces (drag, solar radiation pressure)
- Time-dependent potentials (tesseral harmonics, third-body gravity)
- Composite perturbations (J2 + Sun + Moon)

### 8.2 Equinoctial Frame Vectors

The equinoctial frame (eX, eY, eZ) is built from q₁, q₂ (Eq. 37):

```
eX = γ⁻¹ · [1-q₁²+q₂², 2q₁q₂, -2q₁]
eY = γ⁻¹ · [2q₁q₂, 1+q₁²-q₂², 2q₂]
eZ = γ⁻¹ · [2q₁, -2q₂, 1-q₁²-q₂²]
```

### 8.3 Cartesian State Reconstruction

**Position** (Eq. 41):
```
r_vec = X·eX + Y·eY
```
(Expanded into three Cartesian components x, y, z as heyoka expressions.)

**Velocity** (Eq. 43):
```
Ẋ = ṙ·cosL - (h/r)·sinL
Ẏ = ṙ·sinL + (h/r)·cosL
v_vec = Ẋ·eX + Ẏ·eY
```

Note: Computing velocity requires h, which requires U, which requires position. The dependency chain is: state → intermediates → position → U → h → velocity.

### 8.4 Force Decomposition

The total perturbation force is F = P - ∇U (Eq. 3), where:
- P = non-conservative acceleration (from `P_expr`)
- ∇U = gradient of conservative potential (from `grad_U_expr`)

Force projections onto the orbital frame:
```
F_h = F · eZ    (out-of-plane)
P_r = P · ê_r   (radial, non-conservative only — for Ė)
P_f = P · ê_f   (along-track, non-conservative only — for Ė)
```

where ê_r = eX·cosL + eY·sinL and ê_f = eY·cosL - eX·sinL.

**Important**: The 2U - rF_r term uses the full expression:
```
2U - rF_r = 2U + r·∇U·ê_r - r·P_r = 2U + (x·dU/dx + y·dU/dy + z·dU/dz) - r·P_r
```

The Euler homogeneity identity `r·∇U = -(n+1)·U` is NOT used here because the general path handles arbitrary U (not necessarily homogeneous). The full dot product `x·dU/dx + y·dU/dy + z·dU/dz` is computed explicitly.

### 8.5 Energy Derivative

```
Ė = ∂U/∂t + ṙ·P_r + (h/r)·P_f    (Eq. 46)
```

For conservative + time-independent perturbations: Ė = 0.

### 8.6 L̇ and K̇ Derivation

**L̇** is given directly by Eq. 49:
```
L̇ = ν + d - w_h + (1/c)·(1/α + α·(p₁·sinK + p₂·cosK))·(2U - rF_r) + (ṙ·α)/(μ·c)·(r + ρ)·Ė
```

where ρ = c²/μ.

**K̇** is derived from L̇ rather than using Eq. 75 directly. From the generalized Kepler equation L = K + p₁·cosK - p₂·sinK, differentiating:

```
L̇ = K̇·(1 + p₁·sinK + p₂·cosK)/1 + ṗ₁·cosK - ṗ₂·sinK
```

Wait — more carefully:
```
L̇ = K̇ + K̇·(p₁·sinK + p₂·cosK)/? ...
```

Actually, the derivative of L = K + p₁ cos K - p₂ sin K is:
```
dL/dt = dK/dt · (1 + p₁ sin K + p₂ cos K) + dp₁/dt · cos K - dp₂/dt · sin K
```

Wait, let me be precise. L = K + p₁·cos(K) - p₂·sin(K). Taking the total derivative:
```
L̇ = K̇ + ṗ₁·cosK - ṗ₂·sinK + K̇·(-p₁·sinK - p₂·cosK) ... no
```

Actually: L = K + p₁ cos K - p₂ sin K
```
dL/dt = dK/dt + (dp₁/dt) cos K + p₁ (-sin K)(dK/dt) - (dp₂/dt) sin K - p₂ (cos K)(dK/dt)
      = dK/dt (1 - p₁ sin K - p₂ cos K) + ṗ₁ cos K - ṗ₂ sin K
      = K̇ · (r/a) + ṗ₁ cos K - ṗ₂ sin K
```

since r/a = 1 - p₁ sin K - p₂ cos K (Eq. 31). Therefore:

```
K̇ = (a/r) · (L̇ - ṗ₁ cos K + ṗ₂ sin K)
```

This is the form used in the implementation. It avoids a separate K̇ formula and is algebraically exact.

### 8.7 Assembled ODEs (General)

```
ν̇  = -3·(ν/μ²)^(1/3) · Ė
ṗ₁ = p₂·(d - w_h) + (1/c)·(X/a + 2p₂)·(2U - rF_r) + (1/c²)·[Y·(r + ρ) + r²·p₁]·Ė
ṗ₂ = p₁·(w_h - d) - (1/c)·(Y/a + 2p₁)·(2U - rF_r) + (1/c²)·[X·(r + ρ) + r²·p₂]·Ė
K̇  = (a/r)·(L̇ - ṗ₁·cosK + ṗ₂·sinK)
q̇₁ = (γ/2)·w_Y
q̇₂ = (γ/2)·w_X
```

### 8.8 q-dot Equations: Paper vs Plan Correction

The original plan had incorrect q-dot equations. The correct forms from the paper (Eqs. 50-51) are:

```
q̇₁ = (γ/2)·w_Y
q̇₂ = (γ/2)·w_X
```

where:
```
w_X = (X/h)·F_h
w_Y = (Y/h)·F_h
```

These are NOT the same as `-I·sinL`, `-I·cosL` for general perturbations (those are J2-only specializations).

---

## 9. Path Selection Logic

**Functions**: `_can_use_zonal_path()`, `_is_j2_only()` in `rhs.py`

Priority order (checked sequentially):

```
1. Zonal fast path?
   _zonal_fast_path=True AND is_conservative=True AND is_time_dependent=False
   → _build_zonal_system()

2. J2-only fast path?
   _j2_fast_path=True
   → _build_j2_only_system()

3. General path (default)
   → _build_general_system()
```

**Why this priority?** The zonal path is the most optimized for multi-degree zonal harmonics. The J2-only path is optimized for the special case n=2 only. The general path handles everything else.

**Flag interactions**:

| Model | `_zonal_fast_path` | `_j2_fast_path` | `_force_general` | Path |
|-------|-------------------|-----------------|------------------|------|
| `J2Perturbation` | False | True | False | J2-only |
| `ZonalPerturbation` | True | False | True | Zonal fast |
| `ThirdBodyPerturbation` | False | False | False | General |
| `CompositePerturbation(conservative=[J2])` | False | True | False | J2-only |
| `CompositePerturbation(conservative=[Zonal])` | False | False | True | General |

Note: the J2 path is opt-in because it hard-codes the Eq. 56 potential and the Section 7.1 simplifications. Static conservative is not enough to qualify. `ZonalPerturbation` still sets `_force_general = True` so wrapped zonal models cannot fall through to the J2 shortcut.

---

## 10. Runtime Parameters

heyoka supports runtime parameters (`hy.par[i]`) that can be changed without recompiling the expression graph. The implementation uses them for:

| Path | par[0] | par[1] | par_map |
|------|--------|--------|---------|
| J2-only | μ | A_J2 | `{"mu": 0, "A_J2": 1}` |
| Zonal fast | μ | — | `{"mu": 0}` |
| General | μ | — | `{"mu": 0}` |

**Why A_J2 is not a parameter in general/zonal paths**: The J2-only path uses `hy.par[1]` for A_J2 because it's a simple scalar coefficient in the hard-coded J2 potential. The general and zonal paths do not require `perturbation.A` at all.

**mu source**: `par[0]` is filled from `perturbation.mu` when that attribute exists, otherwise it falls back to the package constant `MU`. This keeps the public perturbation protocol minimal while still allowing models to override the central body's gravitational parameter.

Only models that explicitly enable `_j2_fast_path` need to expose an `A` coefficient.

**Parameter building** (`integrator.py`):

```python
def _build_par_values(perturbation, par_map):
    if not par_map: return []
    n_pars = max(par_map.values()) + 1
    par_values = [0.0] * n_pars
    if "mu" in par_map:
        par_values[par_map["mu"]] = getattr(perturbation, "mu", MU)
    if "A_J2" in par_map:
        if not hasattr(perturbation, "A"):
            raise AttributeError("J2 fast-path perturbations must define an 'A' coefficient.")
        par_values[par_map["A_J2"]] = perturbation.A
    return par_values
```

This builds the parameter array dynamically from the par_map returned by `build_geqoe_system()`.

---

## 11. Coordinate Conversions

**File**: `conversions.py`

### 11.1 cart2geqoe

Converts Cartesian (r_vec, v_vec) to GEqOE [ν, p₁, p₂, K, q₁, q₂]:

1. Compute geometric quantities: r, v², ṙ, h_vec, h
2. Compute disturbing potential: U = perturbation.U_numeric(r_vec)
3. Compute total energy: E = v²/2 - μ/r + U (Eq. 16)
4. Generalized mean motion: ν = (1/μ)·(-2E)^(3/2)
5. Orientation parameters: q₁, q₂ from angular momentum direction (Eqs. 35-36)
6. Equinoctial frame: eX, eY from q₁, q₂ (Eq. 37)
7. Effective angular momentum: c = r·sqrt(h² + 2U·r²)... wait, more precisely:
   - U_eff = h²/(2r²) + U → c = r·sqrt(2·U_eff)
8. Generalized Laplace vector: g = (υ × (r × υ))/μ - ê_r where υ = ṙ·ê_r + (c/r)·ê_f
9. Eccentricity projections: p₁ = g·eY, p₂ = g·eX
10. K from sinK, cosK via Eq. 42 inverse

### 11.2 geqoe2cart

Converts GEqOE to Cartesian:

1. Build equinoctial frame from q₁, q₂
2. Compute shape quantities (a, β, α)
3. Position from K (Eq. 42): X, Y → r_vec = X·eX + Y·eY
4. Distance r and radial velocity ṙ from K
5. Compute c from ν, β
6. Compute U = perturbation.U_numeric(r_vec)
7. Physical angular momentum: h = sqrt(c² - 2r²U)
8. Velocity: Ẋ, Ẏ → v_vec = Ẋ·eX + Ẏ·eY

### 11.3 Consistency Requirement

**Critical**: The same perturbation model must be used for coordinate conversion and propagation. The conversion computes h and c from U, which depend on the potential model. If you convert with J2-only but propagate with J2+J3+J4, the initial GEqOE state will represent a slightly different Cartesian state when converted back through the propagation model.

Example: `cart2geqoe(r0, v0, mu, j2_pert)` vs `cart2geqoe(r0, v0, mu, zonal_pert)` give different GEqOE states because h and c differ. Both convert back to the same r0, v0 through their respective models, but cross-model conversion introduces errors.

---

## 12. Integrator Wrappers

**File**: `integrator.py`

### 12.1 build_state_integrator

Creates a 6-DOF heyoka `taylor_adaptive` integrator:

```python
build_state_integrator(perturbation, ic, t0=0.0, tol=1e-15, compact_mode=False)
```

Steps:
1. Call `build_geqoe_system(perturbation, use_par=True, time_origin=t0)` to get the ODE system and par_map
2. Build parameter values from par_map
3. Create `hy.taylor_adaptive(sys, state=ic, pars=par_values, tol=tol)`
4. Return (ta, par_map)

For time-dependent perturbations, `build_geqoe_system()` receives `time_origin=t0`, so the symbolic time passed into perturbation models is `hy.time - t0`. This makes `epoch_jd` correspond to the physical epoch of the initial state even when the integrator's external time coordinate starts at a nonzero value.

### 12.2 build_stm_integrator

Creates a 42-DOF integrator with automatic variational equations:

```python
build_stm_integrator(perturbation, ic, t0=0.0, tol=1e-15, compact_mode=True)
```

Uses `hy.var_ode_sys(sys, hy.var_args.vars, order=1)` to automatically derive the 6×6 variational equations. The augmented state is [6 state + 36 STM entries]. Compact mode is enabled by default for the 42-DOF system to reduce expression tree size and compilation time.

**STM layout**: The 36 STM entries are stored in row-major order. Use `state[6:].reshape(6, 6)` to extract the 6×6 matrix. The initial condition is the 6×6 identity matrix (flattened).

### 12.3 Mass-Augmented and Parameter-Sensitivity Builders

Phase 8 adds three thrust-oriented wrappers on top of the same symbolic RHS:

- `build_thrust_state_integrator(...)`: 7-state propagation for
  `(nu, p1, p2, K, q1, q2, m)`
- `build_thrust_stm_integrator(...)`: 7-state STM wrt the augmented initial
  state only
- `build_thrust_sensitivity_integrator(...)`: 7-state variational system wrt
  both the augmented initial state and all runtime parameters

The last builder uses:

```python
hy.var_ode_sys(sys, hy.var_args.vars | hy.var_args.params, order=1)
```

For a state dimension `n` and `k` runtime parameters, the augmented Jacobian
has shape `n x (n + k)`. heyoka stores it in row-major order after the
original state. The columns are ordered as:

1. state variables in propagation order
2. runtime parameters in increasing `par[i]` index order

### 12.4 Propagation Helpers

- `propagate(ta, t_final, max_delta_t=None)`: Step-by-step propagation collecting times and states at each step boundary.
- `propagate_grid(ta, t_grid)`: Dense output at specified time grid points via heyoka's built-in `propagate_grid`.
- `extract_stm(state_aug)`: Extract 6-element state and 6×6 STM from 42-element augmented state.
- `extract_variational_matrices(state_aug, state_dim, par_map)`: Extract the
  propagated state, state STM, parameter sensitivity matrix, and ordered
  parameter names from a `vars | params` augmented system.
- `extract_endpoint_jacobian(...)`: Select output rows and parameter columns
  from the full variational matrices for endpoint-constraint assembly.
- `parameter_names_from_map(par_map)`: Recover runtime parameter names ordered
  by the underlying heyoka parameter indices.

---

## 13. Perturbation Implementations

### 13.1 J2Perturbation (`perturbations/j2.py`)

The simplest model. Provides all protocol methods:

- **U_expr**: `-A/r³·(1 - 3ẑ²)` where A = μ·J2·Re²/2 (Eq. 56)
- **U_numeric**: Same formula with NumPy
- **grad_U_expr**: Hand-derived analytical gradient:
  - dU/dx = 3A·x/r⁵·(1 - 5ẑ²)
  - dU/dy = 3A·y/r⁵·(1 - 5ẑ²)
  - dU/dz = 3A·z/r⁵·(3 - 5ẑ²)
- **P_expr**: (0, 0, 0)
- **U_t_expr**: 0

Flags: `is_conservative = True`, `is_time_dependent = False`, `_j2_fast_path = True`.

### 13.2 ZonalPerturbation (`perturbations/zonal.py`)

Supports arbitrary zonal harmonics J2 through Jn.

**Constructor**: `ZonalPerturbation(j_coeffs, mu=MU, re=RE)` where `j_coeffs = {2: J2, 3: J3, 4: J4}`.

**Pre-computed coefficients**: `self._Cn = {n: mu * Jn * re^n for n, Jn in j_coeffs.items()}`

**Validation**: Empty j_coeffs raises ValueError. Degree < 2 raises ValueError.

**Dual interface**:
1. **Fast path** (`zonal_quantities`): Returns (U, dU/dẑ, euler_term) directly from r and ẑ. Used by `_build_zonal_system`.
2. **Standard** (`U_expr`, `grad_U_expr`): Uses placeholder variables + `hy.diff_tensors` + `hy.subs`. Lazy-built (`_ensure_cart_grad`). Used by the general path (when wrapped in CompositePerturbation).

**U_numeric**: Uses `zonal_quantities(r, zhat)` with float inputs — the Legendre recurrence works with both heyoka expressions and floats.

Flags: `_zonal_fast_path = True`, `_force_general = True`, `is_conservative = True`, `is_time_dependent = False`.

### 13.3 ThirdBodyPerturbation (`perturbations/third_body.py`)

Models Sun or Moon gravitational perturbation as non-conservative P.

**Why non-conservative?** Per Section 7.2 of the paper, treating third-body gravity as P (rather than adding to U) avoids numerical instabilities. Adding it to U would change h and c, requiring the reference ellipse to track the combined Earth+third-body potential — which can cause issues when the third-body force varies rapidly relative to the orbital period.

**Ephemeris**:
- Sun: heyoka's built-in VSOP2013 theory (heliocentric Earth-Moon barycenter, negated to get geocentric Sun)
- Moon: heyoka's built-in ELP2000 theory (geocentric Moon)

Both are in ecliptic J2000 coordinates and are rotated to equatorial J2000 via the obliquity rotation:
```
x_eq = x_ecl
y_eq = y_ecl·cos(ε) - z_ecl·sin(ε)
z_eq = y_ecl·sin(ε) + z_ecl·cos(ε)
```

**Time conversion**: The perturbation consumes the relative time passed by the RHS builder, `t_rel = hy.time - t0`, so `epoch_jd` always refers to the epoch of the initial state:
- VSOP2013: Julian millennia from J2000 = (epoch_jd - JD_J2000)/365250 + t_rel/(86400·365250)
- ELP2000: Julian centuries from J2000 = (epoch_jd - JD_J2000)/36525 + t_rel/(86400·36525)

**Truncation threshold**: Controls how many terms from the analytical theory are kept. Larger threshold = fewer terms = faster JIT compilation but less precise ephemeris.

| Threshold | JIT time | Use case |
|-----------|----------|----------|
| Sun 1e-9 / Moon 1e-6 (default) | Minutes | High-accuracy research |
| Sun 1e-4 / Moon 1e-2 | 2-4 seconds | Most practical applications |

**Acceleration formula** (Battin's form):
```
P = μ₃ · ((r₃ - r)/|r₃ - r|³ - r₃/|r₃|³)
```

The second term (indirect part) accounts for the acceleration of Earth's center by the third body.

Flags: `is_conservative = False`, `is_time_dependent = True`. The class also exposes `mu = MU` for the central body so it can be used standalone without wrapping it in a composite model.

### 13.4 CompositePerturbation (`perturbations/composite.py`)

Combines multiple perturbation models, maintaining the separation between conservative (U) and non-conservative (P).

**Constructor**: `CompositePerturbation(conservative=[...], non_conservative=[...])`

**Aggregation**:
- `U_expr`, `U_numeric`, `grad_U_expr`: Sum over conservative models
- `P_expr`: Sum over non-conservative models
- `mass_flow_expr`: Sum over mass-coupled non-conservative models
- `P_and_mass_flow_expr`: Fused acceleration + mass-flow traversal for
  mass-augmented propagation
- `U_t_expr`: Sum over conservative models (if they have it)
- `is_conservative`: True only if no non-conservative models
- `is_time_dependent`: True if any child model is time-dependent
- `requires_mass`: True if any child model requires the mass state
- `_force_general`: True if any child model sets `_force_general`
- `_j2_fast_path`: True only for the exact pure-J2 case (one conservative child with `_j2_fast_path=True` and no non-conservative children)

**mu and A**: Taken from the first conservative model when available, with fallbacks to `MU` and `0.0`. This keeps composites compatible with custom conservative models that implement the public protocol but do not define private bookkeeping attributes.

**Typical usage**:
```python
comp = CompositePerturbation(
    conservative=[J2Perturbation()],
    non_conservative=[
        ThirdBodyPerturbation("sun", epoch_jd, thresh=1e-4),
        ThirdBodyPerturbation("moon", epoch_jd, thresh=1e-2),
    ]
)
```

### 13.5 Continuous-Thrust Stack (`thrust.py`, `perturbations/thrust.py`)

Phase 8a adds a control layer on top of the existing non-conservative
acceleration path rather than introducing a separate thrust solver.

**Control law interface**:

```python
class ContinuousThrustLaw(Protocol):
    def parameter_defaults(self, prefix: str) -> dict[str, float]: ...
    def thrust_rtn_expr(self, state, t, pars, prefix) -> tuple: ...
```

The current shipped laws are:

- `ConstantRTNThrustLaw`, which exposes four runtime parameters:
  - `thrust.r_newtons`
  - `thrust.t_newtons`
  - `thrust.n_newtons`
  - `thrust.isp_s`
- `CubicHermiteRTNThrustLaw`, a smooth single-arc spline law defined over
  normalized time `tau = t / duration_s`, with endpoint values and endpoint
  slopes for each RTN thrust component:
  - `thrust.r0_newtons`, `thrust.r1_newtons`
  - `thrust.t0_newtons`, `thrust.t1_newtons`
  - `thrust.n0_newtons`, `thrust.n1_newtons`
  - `thrust.r0_slope_newtons`, `thrust.r1_slope_newtons`
  - `thrust.t0_slope_newtons`, `thrust.t1_slope_newtons`
  - `thrust.n0_slope_newtons`, `thrust.n1_slope_newtons`
  - `thrust.isp_s`

These parameters are mapped into `hy.par[i]`, so the symbolic graph can be
reused while thrust coefficients are varied or differentiated later.

**Perturbation wrapper**: `ContinuousThrustPerturbation(law, name="thrust")`

Responsibilities:

- reconstruct the RTN basis from Cartesian `(r, v)`
- convert RTN thrust force (newtons) into Cartesian acceleration `P` (km/s²)
  using the propagated mass
- propagate `ṁ = -T / (g_0 I_{sp})`
- declare `requires_mass = True` so users must select the 7-state builders

**Public integrators**:

- `build_thrust_state_integrator()`: 7-state propagation for
  `(nu, p1, p2, K, q1, q2, m)`
- `build_thrust_stm_integrator()`: 7-state STM wrt the augmented initial state
- `build_thrust_sensitivity_integrator()`: 7-state sensitivities wrt the
  augmented initial state and runtime parameters

**Sensitivity extraction**:

- `extract_variational_matrices()` returns:
  - `Phi_x`: sensitivities wrt the initial augmented state
  - `Phi_p`: sensitivities wrt all runtime parameters
  - `param_names`: parameter names ordered consistently with the columns of
    `Phi_p`
- `extract_endpoint_jacobian()` slices these matrices into the exact blocks
  needed for chosen endpoint outputs and chosen runtime-parameter subsets

This is the main Phase 8b bridge to multiple shooting and optimization: the
endpoint Jacobian with respect to control coefficients comes straight from the
same symbolic graph as the propagated state.

The existing 6-state integrators intentionally reject perturbations that set
`requires_mass = True`, so the API fails fast instead of silently dropping mass
depletion.

**Multiple-shooting layer**:

- `ShootingArc`: one 7-state thrust arc with a nominal initial state, duration,
  optional selected runtime parameters, and optional absolute start time
- `MultiArcShootingProblem`: reusable multi-arc transcription helper built on
  top of `build_thrust_sensitivity_integrator()`
- `ArcPropagationResult` / `ShootingEvaluation`: arc-endpoint data plus sparse
  continuity residuals and Jacobians

The flat decision vector is grouped per arc as:

```text
[x_0, p_0, x_1, p_1, ..., x_{N-1}, p_{N-1}]
```

where each `x_i` is the 7-state GEqOE + mass node and each `p_i` is the chosen
subset of arc-local runtime parameters. The first prototype assembles:

- continuity constraints `y_i(tf) - x_{i+1}`
- terminal residuals for selected endpoint outputs
- a minimum-propellant objective `m_0 - m_f`

with exact sparse Jacobians extracted from the same variational propagation
already used for the 7-state endpoint sensitivities. For performance, the
problem object reuses compiled heyoka integrators by resetting state, time, and
runtime parameters between evaluations rather than rebuilding the symbolic
graph.

The first solver-facing helpers are also part of this layer:

- `build_named_bounds(...)`: convert exact decision names or shared suffixes
  such as `m` / `thrust.t_newtons` into SciPy `Bounds`
- `TerminalConstraintSpec`: bounded or equality-constrained terminal outputs on
  the final arc
- `SmoothnessPenaltySpec`: quadratic cross-arc regularization for selected
  control variables
- `ShootingSolveSpec`: explicit solve configuration for the prototype backend
- `solve(...)`: spec-driven SciPy `trust-constr` solve entry point
- `solve_minimum_propellant(...)`: convenience wrapper around `solve(...)`
  for the current minimum-propellant prototype

The objective model is intentionally simple and exact:

- base objective: `m_0 - m_f`
- optional regularization:
  `0.5 * \sum w_j (u_{i+1,j} - u_{i,j})^2`

where the selector-matched control variables are penalized across consecutive
arcs. This gives a useful smoothness-regularized prototype now, while keeping a
clear path to richer multi-term objectives later.

This is intentionally a thin adapter rather than a committed long-term solver
backend. It validates that the GEqOE transcription and exact Jacobians are
usable directly inside a constrained optimizer before adding richer sparse-NLP
plumbing.

An executable reference is included in
`examples/geqoe_heyoka/geqoe_taylor_shooting_demo.py`, which sets up a two-arc thrust
problem, applies terminal constraints and a smoothness penalty, and solves it
with the current SciPy adapter.

The surrounding example set is now broader than that single solve entry point:
`examples/geqoe_heyoka/maneuver_profile_gallery.py` visualizes constant-thrust,
smooth cubic-Hermite, and shooting-based maneuver profiles, while
`examples/geqoe_compare/native_vs_heyoka_j2.py` checks coefficient-level
parity against the earlier GEqOE J2 implementation on the shared state
components. On the estimation side,
`examples/geqoe_reconstruction_lab/position_fit_demo.py` remains the simplest
position-only fit, while
`examples/geqoe_reconstruction_lab/mixed_position_range_fit_demo.py` shows that
the same transcription can already mix inertial position and inertial range
samples without a separate solver path.

That measurement bridge is now part of the same layer rather than living in an
ad hoc example script. The current implementation adds:

- `MeasurementModel`: interface for per-sample measurement evaluation and local
  Jacobians with respect to the propagated 7-state GEqOE + mass sample
- `SampledMeasurement`: observation definition carrying arc selector
  (`arc_index` or `arc_name`), sample time relative to the arc start, observed
  value, and a left-weighting/whitening matrix
- `MeasurementResidualEvaluation`: residual/Jacobian bundle attached to the
  corresponding `ShootingEvaluation`
- `MultiArcShootingProblem.evaluate_measurements(...)`: residual assembly using
  the same compiled 7-state sensitivity integrators already used for
  continuity/terminal derivatives
- `MeasurementObjectiveSpec`: weighted least-squares measurement objective
  configuration, including solve-time Hessian strategy
- `DecisionTrackingPenaltySpec`: quadratic selector-based tracking/effort
  regularization for decision variables
- `solve_measurement_fit(...)`: convenience wrapper that routes measurement
  residuals, continuity constraints, and regularization through the same solve
  entry point
- `estimate_covariance(...)`: constrained local covariance estimate built from
  the same linearized objective and equality constraints

The implemented chain is intentionally simple and future-proof:

1. propagate each arc's augmented state and first-order variational equations
   to the requested sample epochs using dense output,
2. evaluate the measurement model on each propagated 7-state sample,
3. evaluate the model's local Jacobian with respect to that 7-state sample,
4. apply the chain rule through the exact GEqOE sensitivity blocks, and
5. left-multiply by the supplied whitening matrix.

This means the observation stack is still derivative-consistent with the
underlying GEqOE dynamics: the sample-state sensitivities come from the same
`hy.var_args.vars | hy.var_args.params` propagation used elsewhere, and the
measurement layer only adds a small local map on top of that flow.

The first implemented observation type is
`InertialPositionMeasurementModel`, which evaluates Cartesian inertial position
directly from the propagated GEqOE state and uses a compiled symbolic Jacobian
with respect to the 7-state sample. This is the intended toy problem for early
maneuver estimation because it is easy to validate against `geqoe2cart()` and
directly relevant to position-only reconstruction experiments.

The second implemented observation type is
`InertialRangeMeasurementModel`, which reduces the same inertial Cartesian
position to a one-way range against a fixed inertial reference point. Its local
Jacobian is just the line-of-sight row multiplied by the exact inertial-position
Jacobian, so it remains on the same derivative-consistent path as the position
model. Because sampled measurements are stored as a heterogeneous list of
`MeasurementModel` instances, position and range samples can already be mixed in
the same batch without changing the transcription logic.

The weighting API is already compatible with future uncertainty-aware work.
`SampledMeasurement` accepts an arbitrary left-weighting matrix and provides
convenience constructors from diagonal standard deviations or a full
covariance. In other words, the current residual stack already uses whitened
measurement space rather than hard-coding scalar weights.

The current solve path is still intentionally modest. Measurement objectives now
use exact first-order residuals/Jacobians and selector-based quadratic tracking
penalties inside `ShootingSolveSpec`. For constrained multi-arc estimation
problems, the default solve route is exact first-order data plus SciPy's
quasi-Newton curvature updates; an explicit Gauss-Newton objective Hessian is
still available, but the full second-order measurement terms are not yet
propagated. That makes the present interface appropriate for prototype
maneuver-estimation studies, not yet a finished sparse least-squares backend.

That same solve assembly now underpins a first local uncertainty estimate.
`estimate_covariance(...)` builds a constrained covariance from the
Gauss-Newton / quadratic objective Hessian together with the linearized
equalities currently present in the solve model:

- continuity equations,
- terminal constraints only when they are exact equalities, and
- fixed bounds where `lb == ub`.

General active inequalities are still excluded from the covariance model. The
result is already useful for maneuver detection because recovered thrust
coefficients can now be reported with local standard deviations and
significance scores, but it remains a local linearized approximation rather
than a full posterior treatment.

The remaining limitations should also be stated explicitly. Only inertial
position and inertial range are implemented today; velocity, line-of-sight,
angles, and range-rate are still future work. The current decision-tracking
terms are deterministic quadratic penalties rather than a full prior/covariance
model, and there is still no first-class abstraction for mixed observation
families plus priors plus maneuver regularization inside a single uncertainty
model. The architecture boundary is now in place, while the broader estimation
stack remains prototype-level.

The new maneuver-detection demo exercises that stack directly. It fits four
candidate thrust arcs against mixed inertial position and range data, applies
continuity plus fixed-initial-state plus terminal equality constraints, and
classifies each arc by `|T| / sigma_T`. One prototype caveat is worth stating
explicitly: the exact-zero constant-thrust case is still a numerical edge for
the present sensitivity integrator, so the demo keeps "quiet" arcs at a tiny
positive thrust floor while still treating them as zero-thrust for reporting.

By default, arc evaluations use local time `t = 0` unless an explicit
`start_time_s` is provided. This keeps the current arc-local cubic Hermite
thrust law consistent across multiple arcs while still allowing absolute-time
transcription when needed.

---

## 14. Numerical Stability Techniques

### 14.1 h - c Computation

The physical angular momentum h = sqrt(c² - 2r²U) is close to c when the perturbation is small. Computing h - c directly suffers from catastrophic cancellation. Instead:

```
h - c = (h² - c²) / (h + c) = -2r²U / (h + c)
```

This is used everywhere h - c (or d = (h-c)/r²) appears.

### 14.2 1 - r/a Computation

The quantity 1 - r/a appears in K̇. Computing r/a and subtracting from 1 loses precision when the orbit is nearly circular. Instead, from Eq. 31:

```
r = a·(1 - p₁·sinK - p₂·cosK)
→ 1 - r/a = p₁·sinK + p₂·cosK
```

This is exact and avoids the subtraction.

### 14.3 nu_dot = 0 Enforcement

For the J2-only and zonal paths, ν̇ = 0 is enforced by setting `nu_dot = 0.0 * nu` rather than just `0.0`. The multiplication by `nu` ensures heyoka recognizes that `nu_dot` depends on the state variable `nu` (for proper expression graph construction), while still being identically zero.

---

## 15. Sign Conventions

### 15.1 Disturbing Potential U

The disturbing potential U is defined such that the total orbital energy is:

```
E = v²/2 - μ/r + U
```

For J2: U = -A/r³·(1 - 3ẑ²). Note: U is **not** the gravitational potential energy. It represents the perturbation to the Keplerian potential. The sign is chosen so that the total force is F = P - ∇U (Eq. 3), meaning the conservative acceleration is **minus** the gradient of U.

### 15.2 Force = P - ∇U

The total perturbation acceleration F has two parts:
- P: non-conservative acceleration (directly specified, same sign convention as physical force per unit mass)
- -∇U: conservative acceleration (negative gradient of the disturbing potential)

### 15.3 J2 Acceleration Sign

The J2 gradient dU/dx = 3A·x/r⁵·(1 - 5ẑ²). The acceleration is -dU/dx. At the equator (z=0): -dU/dx = -3A·x/r⁵ (pointing inward since A > 0 and x is along position). This is correct: J2 represents the equatorial bulge, adding extra inward pull at the equator.

---

## 16. Validation Strategy

### 16.1 Multi-Level Validation

The implementation is validated at multiple levels:

| Level | Method | Tolerance |
|-------|--------|-----------|
| Conversion round-trip | cart→geqoe→cart | < 1e-10 km pos, < 1e-13 km/s vel |
| Kepler equation | K→L→K | Machine precision |
| ν conservation | ν(t) = ν(0) for conservative/time-independent | < 1e-14 |
| 12-day vs paper reference | Appendix C final state | < 1e-5 km |
| GEqOE vs Cowell (heyoka) | Same force model, Cartesian RHS | < 1e-6 km (J2), < 1e-4 km (zonal) |
| GEqOE vs Cowell (continuous thrust) | Same RTN thrust law + mass flow | < 5e-4 km pos, < 1e-6 km/s vel |
| STM vs finite differences | Perturbed trajectories | relative error < 1e-5 |
| Legendre polynomials | Known values P₀–P₄ | Machine precision |
| Gradient vs finite differences | (U(r+ε) - U(r-ε))/(2ε) | relative < 5e-5 |
| Higher harmonics effect size | J3+J4 difference from J2-only | 0.01–100 km (12-day LEO) |

### 16.2 Cowell Ground Truth

Two independent Cartesian propagators serve as ground truth:

1. **scipy DOP853** (`propagate_cowell`): rtol=atol=1e-14. Lower accuracy but independent of heyoka.
2. **heyoka Taylor** (`propagate_cowell_heyoka`): tol=1e-15. Highest accuracy, same engine.
3. **heyoka Taylor full** (`propagate_cowell_heyoka_full`): J2 + Sun + Moon via same ephemeris.
4. **heyoka Taylor general** (`propagate_cowell_heyoka_general`): arbitrary
   `U/P` perturbation models, optionally with propagated mass.
5. **Zonal Cowell** (in test file): Cowell propagator using `ZonalPerturbation.grad_U_expr` for arbitrary zonal harmonics.

The GEqOE propagator should match the heyoka Cowell propagator to within the element conversion precision, since both use the same underlying Taylor integrator and force model.

### 16.3 Test Organization

| File | Tests | Coverage |
|------|-------|----------|
| `test_geqoe_taylor.py` | 14 | Conversions, Kepler eq, propagation, STM, Cowell J2 |
| `test_geqoe_taylor_general.py` | 11 | General equations, third-body, composite, Cowell full |
| `test_geqoe_taylor_zonal.py` | 22 | Legendre, zonal construction, J2 match, gradient FD, higher-order, Cowell zonal |
| `test_geqoe_taylor_thrust.py` | 11 | Mass-augmented GEqOE, smooth spline law, endpoint Jacobians, STM, Cowell thrust, parameter sensitivities |
| `test_geqoe_taylor_shooting.py` | 8 | Multi-arc continuity closure, sparse Jacobians, smoothness objective, named bounds, terminal/objective assembly, and SciPy solve adapters |

---

## 17. Known Pitfalls and Lessons Learned

### 17.1 K̇ Coefficient Bug

The coefficient of U in K̇ is `(1/c)·(1 + α·(1 - r/a))`, NOT `(1/c)·(ℓ/α + α·(1 - r/a))`. The latter comes from naively substituting into the L̇ formula (Eq. 49) and confusing the L→K transformation. The correct form comes from Eq. 75 directly. Getting this wrong gives K̇ approximately 7x too large.

### 17.2 q-dot Equations (General Form)

The plan originally had wrong q-dot equations. The correct forms from Eqs. 50-51 are:
```
q̇₁ = (γ/2)·w_Y
q̇₂ = (γ/2)·w_X
```
NOT the simplified J2-only forms (`-I·sinL`, `-I·cosL`) applied to general perturbations.

### 17.3 IC Consistency

Always use the same perturbation model for `cart2geqoe` and the subsequent propagation. Mixing models (e.g., converting with J2-only but propagating with J2+J3+J4) introduces systematic errors because h and c differ between models. See Section 11.3.

### 17.4 heyoka Parameter Count Mismatch

When building the heyoka expression tree, the number of `hy.par[i]` references detected by heyoka must match the length of the `pars` array passed to `taylor_adaptive`. If a parameter is declared but never referenced in the expressions (e.g., `A_J2` when using a model that bakes coefficients as literals), heyoka raises "Invalid number of parameter values".

**Solution**: Only include parameters in `par_map` that are actually used in the expression tree. The general and zonal paths use `par_map = {"mu": 0}` (only μ as a parameter); A and other coefficients are baked in as literal floats.

### 17.5 heyoka LLVM JIT Caching

First build of an expression graph takes ~1.2s (LLVM compilation). Subsequent builds with the same structure reuse the cache (~9ms). The cache does not persist across Python sessions. Use `compact_mode=True` for large expression trees (42-DOF STM, complex force models) to reduce compilation time.

### 17.6 Third-Body Ephemeris Compile Time

heyoka's VSOP2013 and ELP2000 theories generate large expression trees. At default precision (Sun thresh=1e-9, Moon thresh=1e-6), JIT compilation takes minutes. Use coarser thresholds (1e-4, 1e-2) for interactive use or testing.

### 17.7 Thread Safety

Each `taylor_adaptive` instance is independent. Do not share instances across threads. Create separate integrators per thread.

---

## File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 31 | Package exports |
| `constants.py` | 37 | Physical constants (μ, J2–J6, GM_Sun/Moon, AU, obliquity, time conversions) |
| `conversions.py` | 176 | Cartesian ↔ GEqOE in NumPy |
| `rhs.py` | 399 | Three RHS code paths (J2-only, zonal, general) + shared intermediates |
| `integrator.py` | 167 | heyoka integrator wrappers (state, STM, propagation helpers) |
| `utils.py` | 48 | Kepler equation solver, K↔L conversion |
| `cowell.py` | 244 | Cartesian ground truth propagators (scipy, heyoka, full) |
| `perturbations/base.py` | 57 | PerturbationModel and GeneralPerturbationModel protocols |
| `perturbations/j2.py` | 75 | J2 perturbation model |
| `perturbations/zonal.py` | 178 | Arbitrary zonal harmonics (fast path + auto-gradient fallback) |
| `perturbations/third_body.py` | 167 | Sun/Moon via VSOP2013/ELP2000 |
| `perturbations/composite.py` | 90 | Multi-model aggregation |
