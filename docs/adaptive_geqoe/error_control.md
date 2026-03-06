# Adaptive Error Control for the GEqOE Taylor Propagator

## Overview

The adaptive GEqOE Taylor propagator chains multiple short Taylor
expansions end-to-end so that the accumulated truncation error stays
below a user-specified position tolerance.  Each expansion is anchored
at a **checkpoint** — a Cartesian state together with pre-computed Taylor
coefficients.  An **embedded-pair error estimator** determines the
maximum safe time step (`dt_max`) at each checkpoint, and checkpoints
are created lazily as query times demand.

The method gives orders-of-magnitude speed advantage over numerical
integration while providing controllable, sub-metre position accuracy
for J2-perturbed LEO arcs.

---

## 1. Mathematical Background

### 1.1 Taylor Expansion in GEqOE Space

The propagator works in **Generalised Equinoctial Orbital Elements**
(GEqOE), a set of six slowly-varying elements:

```
y = [nu, q1, q2, p1, p2, Lr]
```

where `nu` is a mean-motion-like parameter, `(q1, q2)` encode the
inclination frame rotation, `(p1, p2)` encode eccentricity, and `Lr` is
the generalised mean longitude.

Given the J2-perturbed equations of motion `dy/dt = f(y)`, we expand `y`
as a Taylor polynomial about a checkpoint epoch `t_0`:

```
y(t_0 + dt) = y_0 + c_1 * u + c_2 * u^2 / 2! + c_3 * u^3 / 3! + c_4 * u^4 / 4!
```

where:

- `u = dt / T` is the normalised time offset
- `T = sqrt(R_e^3 / mu)` is the normalisation time scale (~806 s for Earth)
- `c_k` is the k-th raw derivative of `y` evaluated at the epoch (a
  6-vector stored as column `k-1` of the `map_components` matrix)

The `map_components` matrix has shape `(6, order)` and stores **raw**
(unfactored) derivatives.  The `1/k!` factorial is applied at evaluation
time:

| Order | Contribution to `y` |
|-------|---------------------|
| 1     | `c_1 * u`           |
| 2     | `c_2 * u^2 / 2`     |
| 3     | `c_3 * u^3 / 6`     |
| 4     | `c_4 * u^4 / 24`    |

### 1.2 Mapping to Cartesian Space

GEqOE states are not directly interpretable as position/velocity.  The
mapping to Cartesian is performed via the Jacobian:

```
J = pYpEq = d(r, v) / d(nu, q1, q2, p1, p2, Lr)
```

This 6x6 matrix is computed analytically at each evaluation point.  Its
upper 3x6 block `J_pos = J[:3, :]` maps GEqOE perturbations to position
perturbations in metres.  Similarly, `J_vel = J[3:, :]` maps to velocity
perturbations in m/s.

The full Cartesian STM (state transition matrix) is assembled via the
chain rule:

```
dY/dY_0 = (dY/dEq) @ (dEq/dEq_0) @ (dEq_0/dY_0)
         = pYpEq   @ eq_eq0       @ peq_py_0
```

where `peq_py_0 = d(Eq_0)/d(Y_0)` is the epoch Jacobian from Cartesian
to GEqOE.

---

## 2. Embedded-Pair Error Estimator

### 2.1 Error Model

The error estimator uses the **embedded-pair** approach: the difference
between the order-k and order-(k-1) Taylor solutions.  The highest-order
term that is present in `y_k` but absent in `y_{k-1}` is:

```
y_k - y_{k-1} = c_k * u^k / k!
```

Projected into Cartesian position space via the Jacobian, the position
error estimate is:

```
err(dt) = || J_pos @ c_k || * (dt / T)^k / k!
```

This is the **position-space norm** of the embedded-pair difference — a
scalar in metres that approximates the position error introduced by
truncating the Taylor series at order k rather than including higher
terms.

### 2.2 Why the Factorial Matters

The `map_components` matrix stores raw derivatives `c_k`, not the scaled
Taylor coefficients `c_k / k!`.  The factorial is applied during
evaluation.  Omitting it from the error estimate overestimates the error
by a factor of `k!`:

| Order | `k!` | Error overestimate | Step penalty factor `(k!)^(1/k)` |
|-------|------|--------------------|----------------------------------|
| 2     | 2    | 2x                 | 1.41x shorter steps              |
| 3     | 6    | 6x                 | 1.82x shorter steps              |
| 4     | 24   | 24x                | 2.21x shorter steps              |

Without the factorial, order 4 gets disproportionately shorter steps,
generating many more checkpoints.  The accumulated chaining error from
those extra checkpoints dominates, making order 4 *worse* than order 2
at tight tolerances — the opposite of the expected convergence behaviour.

### 2.3 Implementation

```python
def estimate_position_error(map_components, pYpEq_epoch, dt_seconds, time_scale, order):
    dt_norm = dt_seconds / time_scale
    J_pos = pYpEq_epoch[:3, :]
    delta_eq = map_components[:, order - 1] * dt_norm**order / math.factorial(order)
    return float(np.linalg.norm(J_pos @ delta_eq))
```

Key points:

- `map_components[:, order - 1]` selects the k-th raw derivative column
  (0-indexed).
- `dt_norm**order / factorial(order)` applies the correct Taylor scaling.
- `J_pos @ delta_eq` maps the GEqOE-space perturbation to a 3-vector of
  position error (metres).
- The 2-norm gives a scalar error bound.

---

## 3. Step-Size Controller

### 3.1 Direct Inversion Formula

Given the error model, the maximum step size that keeps position error at
exactly `pos_tol` is obtained by inverting:

```
|| J_pos @ c_k || * (dt / T)^k / k! = pos_tol
```

Solving for `dt`:

```
dt_max = T * (pos_tol * k! / || J_pos @ c_k ||)^(1/k) * safety_factor
```

This is a **closed-form, non-iterative** formula.  The safety factor
(default 0.8) provides a conservative margin — at `dt_max`, the actual
error is `pos_tol * safety_factor^order`, which is always below `pos_tol`.

| Safety factor | Order 2 | Order 3 | Order 4 |
|---------------|---------|---------|---------|
| 0.8           | 0.64x   | 0.51x   | 0.41x   |
| 0.9           | 0.81x   | 0.73x   | 0.66x   |

### 3.2 Implementation

```python
def compute_max_dt(map_components, pYpEq_epoch, time_scale, order, pos_tol,
                   safety_factor=0.8):
    J_pos = pYpEq_epoch[:3, :]
    pc_k = np.linalg.norm(J_pos @ map_components[:, order - 1])
    if pc_k == 0.0:
        return np.inf
    return float(
        time_scale
        * (pos_tol * math.factorial(order) / pc_k) ** (1.0 / order)
        * safety_factor
    )
```

The `pc_k == 0` guard handles the degenerate case where the highest
derivative vanishes (e.g., circular equatorial orbits with zero J2
contribution at certain orders), returning an unbounded step.

### 3.3 Properties of the Direct Formula

- **Exact relationship**: `dt_safe / dt_full = safety_factor` exactly
  (no iterative approximation error).
- **Error at dt_max**: `err(dt_max) = pos_tol * safety_factor^order`
  exactly.
- **Monotonic in tolerance**: halving `pos_tol` reduces `dt_max` by a
  factor of `2^(1/k)`.
- **Monotonic in order**: for the same tolerance, higher order yields
  longer steps (fewer checkpoints).

---

## 4. Checkpoint Architecture

### 4.1 Checkpoint Data Structure

Each checkpoint stores everything needed to evaluate the Taylor
polynomial and compose the STM:

```python
@dataclass
class Checkpoint:
    epoch_seconds: float       # Absolute time from t=0 (seconds)
    y0_cart: np.ndarray        # Cartesian state [rx,ry,rz,vx,vy,vz] (6,)
    coeffs: Any                # Taylor coefficients (Python or C++ capsule)
    peq_py_0: np.ndarray       # (6,6) d(GEqOE)/d(Cart) at this epoch
    dt_max: float              # Maximum |dt| from this checkpoint (seconds)
    cumulative_stm: np.ndarray # (6,6) STM from t=0 to this epoch
```

### 4.2 Checkpoint Chain

Checkpoints are stored in a sorted list by `epoch_seconds`.  The initial
checkpoint is created at `t = 0` with the user-supplied initial orbit
and an identity STM.

When a query time `t` is requested:

1. **Search**: Binary search the checkpoint list for the nearest
   checkpoint(s).  Check if `|t - epoch| <= dt_max` for the two
   candidates straddling `t`.

2. **Hit**: If a covering checkpoint exists, evaluate the Taylor
   polynomial at the local offset `dt = t - epoch`.

3. **Miss**: If no checkpoint covers `t`, extend the chain:
   - Determine direction (forward or backward from the boundary).
   - Evaluate the Taylor polynomial at `+/- dt_max` to get the
     Cartesian state at the next checkpoint epoch.
   - Compose the cumulative STM: `new_stm = local_stm @ prev_stm`.
   - Build a new checkpoint (prepare coefficients, compute new `dt_max`).
   - Insert into the sorted list.
   - Repeat until the new checkpoint covers `t`.

### 4.3 Lazy Evaluation

Checkpoints are created **on demand**.  A 10-orbit propagation that only
queries `t = 0` and `t = T_final` will only create the intermediate
checkpoints needed to reach `T_final`.  Subsequent queries at
intermediate times reuse existing checkpoints without creating new ones.

### 4.4 Batch Optimisation

`propagate_array(dt_seconds)` groups query times by their covering
checkpoint and evaluates each group as a vectorised batch, avoiding
redundant coefficient preparation.

---

## 5. Configuration Reference

| Parameter        | Type             | Default | Description                                                       |
|------------------|------------------|---------|-------------------------------------------------------------------|
| `taylor_order`   | `int`            | `4`     | Taylor expansion order (1-4). Higher = fewer checkpoints.         |
| `pos_tol`        | `float`          | `1.0`   | Position tolerance in metres per checkpoint step.                 |
| `safety_factor`  | `float`          | `0.8`   | Multiplicative margin on `dt_max`.                                |
| `max_step`       | `float` or `None`| `None`  | If set, caps `dt_max` at this value (seconds). Overrides `pos_tol`. |
| `backend`        | `str`            | `"cpp"` | `"cpp"` (fast, compiled) or `"python"` (reference).               |

### Usage Example

```python
from astrodyn_core import AstrodynClient, BuildContext, PropagatorSpec

app = AstrodynClient()
ctx = BuildContext(initial_orbit=orbit)

# Error-controlled mode (recommended)
prop = app.propagation.build_propagator(
    PropagatorSpec(
        kind="geqoe-adaptive",
        orekit_options={
            "taylor_order": 4,
            "backend": "cpp",
            "pos_tol": 1.0,          # 1 metre per step
            "safety_factor": 0.8,
        },
    ),
    ctx,
)

# Fixed-step diagnostic mode
prop_fixed = app.propagation.build_propagator(
    PropagatorSpec(
        kind="geqoe-adaptive",
        orekit_options={
            "taylor_order": 4,
            "backend": "cpp",
            "max_step": 300.0,       # Fixed 300 s steps
        },
    ),
    ctx,
)

# Propagate
y, stm = prop.propagate_array(dt_grid)
```

---

## 6. Convergence Behaviour

The following measurements are from the reference LEO orbit used by the
plot script (`examples/geqoe_native/geqoe_adaptive_error_plot.py`): 6878 km semi-major
axis, e = 0.0012, i = 51.6 deg, period ~5677 s.  The reference solution
is a numerical GEqOE integration at 1e-13 tolerance.

### 6.1 Average Step Sizes

For a given `pos_tol`, higher orders permit longer steps:

| Order | `pos_tol = 10 m` | `pos_tol = 1 m` | `pos_tol = 0.1 m` | `pos_tol = 0.01 m` |
|-------|------------------|------------------|--------------------|---------------------|
| 2     | ~37 s            | ~12 s            | ~3.7 s             | ~1.2 s              |
| 3     | ~115 s           | ~53 s            | ~25 s              | ~12 s               |
| 4     | ~205 s           | ~115 s           | ~65 s              | ~36 s               |

### 6.2 Checkpoint Counts (10-orbit arc, ~56,770 s)

| Order | `pos_tol = 10 m` | `pos_tol = 1 m` | `pos_tol = 0.1 m` | `pos_tol = 0.01 m` |
|-------|------------------|------------------|--------------------|---------------------|
| 2     | 1,530            | 4,836            | 15,293             | 48,358              |
| 3     | 496              | 1,068            | 2,299              | 4,954               |
| 4     | 278              | 493              | 877                | 1,559               |

### 6.3 Max Position Error over 10 Orbits

| Order | `pos_tol = 10 m` | `pos_tol = 1 m` | `pos_tol = 0.1 m` | `pos_tol = 0.01 m` |
|-------|------------------|------------------|--------------------|---------------------|
| 2     | 189.7 m          | 18.9 m           | 1.88 m             | 0.19 m              |
| 3     | 61.3 m           | 8.8 m            | 2.1 m              | 0.78 m              |
| 4     | 21.3 m           | 5.8 m            | 3.2 m              | 1.77 m              |

### 6.4 Convergence Hierarchy and the Chaining Trade-off

At **loose tolerances** (`pos_tol >= 1 m`), the expected hierarchy holds:

```
error(order 4) < error(order 3) < error(order 2)
```

Higher order means fewer checkpoints and less accumulated chaining
error.  Order 4 at 1 m tolerance produces 5.8 m max error vs 18.9 m for
order 2.

At **tight tolerances** (`pos_tol <= 0.1 m`), the hierarchy **inverts**:

```
error(order 2) < error(order 3) < error(order 4)
```

This happens because `pos_tol` controls only the **single-step**
truncation error.  The total arc error is dominated by **chaining error**
— the accumulation of seed-state errors across checkpoints.  Higher
orders take longer steps, so each seed carries a larger absolute error
even though it is within the per-step tolerance.  Order 2 compensates
with many more (but individually shorter and more accurate) steps, and
its chaining error grows more slowly.

The crossover occurs around `pos_tol ~ 0.5 m` for this orbit.  The
practical recommendation is:

- **`pos_tol >= 1 m`**: Use order 4 for fewest checkpoints and best
  accuracy.
- **`pos_tol < 0.1 m`**: Use order 2 for lowest accumulated error, or
  accept that total error will exceed `pos_tol` by the chaining factor.

---

## 7. Error Sources and Limitations

### 7.1 Truncation Error (Controlled)

The embedded-pair estimator controls the **single-step** truncation
error — the difference between the order-k and order-(k-1) solutions
within one checkpoint.  This is the dominant error source and is bounded
by `pos_tol * safety_factor^order` per step.

### 7.2 Chaining Error (Accumulated)

Each checkpoint evaluates the Taylor polynomial at `dt_max` to seed the
next checkpoint.  Small errors in the seed state propagate forward.  The
total accumulated error is the sum of all per-step seed errors along the
chain.

At loose tolerances, fewer checkpoints (higher order) means fewer error
contributions and better long-arc accuracy.  At tight tolerances, the
relationship inverts: higher orders take longer steps with larger
per-step seed errors, and even though they create fewer checkpoints, the
total chaining error exceeds that of lower orders which use many shorter,
more accurate steps (see section 6.4).

### 7.3 Jacobian Approximation

The error estimator uses the **epoch** Jacobian `pYpEq_epoch` rather
than the Jacobian at the evaluation point.  For steps small enough that
the Jacobian varies little, this is an excellent approximation.  The
safety factor provides additional margin for Jacobian variation.

### 7.4 J2-Only Dynamics

The underlying equations of motion include only the J2 zonal harmonic.
Higher-order geopotential terms (J3, J4, ...), atmospheric drag, solar
radiation pressure, third-body perturbations, and other forces are not
modelled.  The propagator is best suited for short-to-medium duration
arcs where J2 dominates the perturbation budget.
