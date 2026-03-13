# GEqOE Averaged Propagator — Performance Optimization Plan

**Goal**: Reduce wall-clock time of the analytical propagator from 23–118 s to
competitive with DSST (0.4–4 s), through three staged refactors (A → B → C)
with intermediate benchmarks after each stage.

---

## Stage 0 — Baseline Profile

### Current Architecture

The evaluation pipeline has three cost centres:

| Phase | What it does | Call pattern | Timing |
|-------|-------------|-------------|--------|
| **One-time load** | `sp.sympify` + `sp.lambdify` on 130 cached string expressions | Once, LRU-cached | ~5 s |
| **Mean propagation** | RK4 integration of 4-variable slow flow | Sequential (RK4 is serial) | 40–78 s |
| **SP reconstruction** | `mean_to_osculating_state` + `geqoe2cart` per sample | Independent per sample | 20–40 s |

### Mean propagation breakdown

Each `evaluate_truncated_mean_rhs_pqm()` call:
- Loops 4 zonal degrees (n=2..5)
- For each degree: 5 variables (g, Q, Ψ, Ω, M)
- For each (degree, variable): iterates over Laurent harmonics m → calls `func(q_val, Q_val)` (lambdified scalar)

Non-zero term count in `MEAN_DATA`:
| Degree | Non-zero terms (across 5 variables) |
|--------|-------------------------------------|
| n=2 | 2 |
| n=3 | 10 |
| n=4 | 10 |
| n=5 | 20 |
| **Total** | **42** |

Call volume per case (LEO, 50 orbits, `samples_per_orbit=64`, `rk4_substeps=8`):
- N_intervals ≈ 3200
- Per RK4 substep: 4 stages × 42 lambdified calls = 168
- Per interval: 168 × 8 = 1344
- **Total mean-prop calls ≈ 3200 × 1344 ≈ 4.3 M scalar Python calls**

### SP reconstruction breakdown

Each `mean_to_osculating_state()` call:
- 1 Kepler solve (`solve_kepler_gen`) — already vectorizable
- `evaluate_truncated_short_period()`: 4 degrees × 5 variables × m harmonics

Non-zero term count in `SHORT_DATA`:
| Degree | Non-zero terms |
|--------|---------------|
| n=2 | 14 |
| n=3 | 20 |
| n=4 | 24 |
| n=5 | 30 |
| **Total** | **88** |

Per sample: 88 lambdified calls (each `func(q, Q, F)` scalar) + 1 Kepler + 1 `geqoe2cart`.
- N_samples ≈ 3200 → **~282K SP calls + 3200 geqoe2cart calls**

### SP-to-Cartesian (`geqoe2cart`) bottleneck

`geqoe2cart()` in `src/astrodyn_core/geqoe_taylor/conversions.py:111-179` is scalar:
- Takes `PerturbationModel` for `U_numeric(r_vec, t)`
- For zonal models, U depends only on `|r|` and `ẑ` — fully vectorizable in principle
- Currently called in a Python list comprehension: `[geqoe2cart(s, ...) for s in osc_rec]`

### Reference timings (from `dsst_timing_table.tex`)

| Case | Cowell | GEqOE m+SP | GEqOE mean | DSST osc | Brouwer |
|------|--------|-----------|------------|----------|---------|
| LEO circ. | 0.37 | 65.62 | 45.77 | 4.13 | 0.19 |
| Crit. low-e | 0.04 | 117.86 | 78.54 | 1.44 | 0.08 |
| MEO/GPS | 0.03 | 58.30 | 39.18 | 0.88 | 0.03 |
| Molniya | 0.02 | 34.99 | 23.59 | 0.68 | 0.02 |

**Target**: bring GEqOE m+SP below 5 s for all cases.

---

## Stage A — Vectorize SP Reconstruction

### Rationale

SP reconstruction is embarrassingly parallel: all N_samples states are
independent. Converting the per-sample loop into batched numpy calls eliminates
~285K Python function call overhead and replaces it with ~88 array calls.

### Implementation

#### A.1 — Vectorized `evaluate_truncated_short_period_batch()`

Create a new function in `short_period.py`:

```python
def evaluate_truncated_short_period_batch(
    nu_arr: np.ndarray,       # (N,)
    g_arr: np.ndarray,        # (N,)
    Q_arr: np.ndarray,        # (N,)
    G_arr: np.ndarray,        # (N,)
    omega_arr: np.ndarray,    # (N,)
    j_coeffs: dict[int, float],
    re_val: float = RE,
    mu_val: float = MU,
) -> dict[str, np.ndarray]:   # each (N,)
```

Inner loop structure:
```python
for n, jn_val in sorted(j_coeffs.items()):
    q_arr = q_from_g(g_arr)               # vectorized
    a_arr = (mu_val / nu_arr**2)**(1/3)
    scale_arr = jn_val * (re_val / a_arr)**n
    F_arr = _complex_f_from_g(g_arr, G_arr)  # vectorized
    w_arr = np.exp(1j * omega_arr)

    for variable in ("g", "Q", "Psi", "Omega", "M"):
        for m_val, func in coeffs[variable].items():
            # func is lambdified with "numpy" module → already array-safe
            total += func(q_arr, Q_arr, F_arr) * w_arr**m_val
```

The lambdified functions (`sp.lambdify((q, Q, F), expr, "numpy")`) already
accept numpy arrays — no changes needed to the lambdification step.

Key changes:
- `q_from_g()` — verify it handles arrays (it's `sqrt(1 - sqrt(1-g²)) / ...`, all numpy-safe)
- `_complex_f_from_g()` — already numpy-safe (`np.exp`, division)
- Complex accumulation over m harmonics → vectorized dot product

#### A.2 — Vectorized `mean_to_osculating_state_batch()`

```python
def mean_to_osculating_state_batch(
    mean_states: np.ndarray,   # (N, 6)
    j_coeffs: dict[int, float],
    ...
) -> np.ndarray:               # (N, 6) osculating GEqOE
```

Steps:
1. Extract arrays: `nu_arr, p1_arr, ...` from columns
2. Compute `g_arr = hypot(p1, p2)`, `Psi_arr = arctan2(p1, p2)`, etc.
3. Solve vectorized Kepler: `K_mean = solve_kepler_gen(L_mean, p1_arr, p2_arr)` — already array-safe
4. Call `evaluate_truncated_short_period_batch()`
5. Apply corrections (all elementwise)
6. Solve Kepler again for osculating K

#### A.3 — Vectorized `geqoe2cart_batch()`

For **zonal perturbations only**, U depends on `(|r|, ẑ·r̂)`. Create:

```python
def geqoe2cart_zonal_batch(
    states: np.ndarray,     # (N, 6)
    mu: float,
    zonal_model,            # ZonalPerturbation — provides Jn, Re
) -> tuple[np.ndarray, np.ndarray]:   # (N,3), (N,3)
```

The body of `geqoe2cart` (L128-179) is pure numpy arithmetic except for
`perturbation.U_numeric(r_vec, t)`. For zonal models, replace with:

```python
r_mag = np.linalg.norm(r_vecs, axis=1)           # (N,)
z_component = r_vecs[:, 2]                         # (N,)
sin_lat = z_component / r_mag
U_arr = sum(Jn * (Re/r_mag)**n * legendre_P(n, sin_lat)
            * (-mu / r_mag) for n, Jn in ...)
```

This avoids N calls to `U_numeric` and replaces them with vectorized Legendre
evaluation.

#### A.4 — Wire into `extended_validation.py`

Replace lines 426-428:
```python
# Before (scalar loop):
osc_rec = np.array([mean_to_osculating_state(s, ...) for s in mean_hist])
meansp_cart = np.array([geqoe2cart(s, ...) for s in osc_rec])

# After (batch):
osc_rec = mean_to_osculating_state_batch(mean_hist, J_COEFFS)
meansp_cart = geqoe2cart_zonal_batch(osc_rec, MU, pert)
```

### Expected gains

| Component | Before | After (est.) | Speedup |
|-----------|--------|-------------|---------|
| SP lambdified calls | 282K scalar | 88 array | ~100–500× |
| Kepler solve | 3200 scalar | 1 array | ~50× |
| geqoe2cart | 3200 scalar | 1 batch | ~50× |
| **SP total** | 20–40 s | **0.1–0.5 s** | ~50–200× |
| Mean prop | 40–78 s | unchanged | 1× |
| **Overall** | 60–118 s | **40–79 s** | ~1.5× |

SP reconstruction becomes negligible. Mean propagation now dominates entirely.

### Benchmark checkpoint A

Run `extended_validation.py` on all 12 cases. Record:
- `mean_prop_time` (unchanged — sanity check)
- `sp_time` (should drop to < 1 s)
- `meansp_time` total
- Verify all position RMS values match baseline to < 1e-10 km

---

## Stage B — CSE-Optimized Hardcoded Python

### Rationale

Mean propagation can't be vectorized across time (RK4 is sequential). The
bottleneck is 42 Python function calls per RHS evaluation, each with
Python↔C overhead and redundant subexpression computation. Replacing 42
separate lambdified functions with a **single compiled function** that uses
Common Subexpression Elimination (CSE) will:

1. Eliminate per-call Python overhead (42 → 1 call per RHS)
2. Eliminate redundant computation across terms (shared factors of q, Q, powers)
3. Produce a single `.py` file with raw arithmetic — no SymPy dependency at runtime

### Implementation

#### B.1 — CSE code generator (`generate_hardcoded.py`)

New script in `docs/geqoe_averaged/geqoe_mean/`:

```python
def generate_mean_rates_function(j_degrees=(2,3,4,5)):
    """Generate a single Python function computing all mean rates."""
    # 1. Load all 42 symbolic expressions from generated_coefficients
    # 2. Build combined expressions for each rate, with w^m Fourier terms
    # 3. Run sp.cse() on the combined set
    # 4. Emit Python function with CSE intermediates
```

The generated function signature:
```python
def mean_rates_cse(q: float, Q: float, omega: float,
                   scale_2: float, scale_3: float,
                   scale_4: float, scale_5: float) -> tuple[float, ...]:
    """Returns (g_dot, Q_dot, Psi_dot, Omega_dot, M_dot) / nu."""
```

Key design choices:
- Input `omega` (not `w`): compute `cos(m*omega)`, `sin(m*omega)` internally for
  real arithmetic (avoids complex numbers entirely)
- Input `scale_n = Jn * (Re/a)^n` precomputed by caller
- Output 5 dimensionless rates; caller multiplies by `nu`
- All intermediate variables named `x0, x1, ...` by CSE

#### B.2 — CSE code generator for SP expressions

Same approach for the 88 short-period terms:

```python
def short_period_cse(q: float, Q: float, f: float, omega: float,
                     scale_2: float, ..., scale_5: float) -> tuple[float, ...]:
    """Returns (dg, dQ, dPsi, dOmega, dM) short-period corrections."""
```

Note: even though SP is now vectorized (Stage A), a CSE-optimized scalar
version is still useful for the mean RHS (which calls SP-like coefficient
evaluations) and as a stepping stone to Stage C.

For the SP batch path, generate an array-compatible version:
```python
def short_period_cse_batch(q_arr, Q_arr, f_arr, omega_arr,
                           scale_2_arr, ...) -> tuple[np.ndarray, ...]:
```

This replaces 88 lambdified calls with 1 function that internally uses CSE.

#### B.3 — Conversion to real arithmetic

The current code uses complex exponentials (`w^m`, `F^k`). For hardcoded
generation, convert to real trig:

- `w^m = cos(m*omega) + i*sin(m*omega)` → extract `.real` at generation time
- `F` in SP expressions: `F = exp(i*f)` → rewrite as `cos(f) + i*sin(f)`,
  then expand and take `.real` of the full sum

This avoids all complex arithmetic at runtime.

#### B.4 — Wire into `short_period.py`

Add a fast-path in `evaluate_truncated_mean_rhs_pqm()`:

```python
try:
    from .hardcoded_rates import mean_rates_cse
    _USE_HARDCODED = True
except ImportError:
    _USE_HARDCODED = False

def evaluate_truncated_mean_rhs_pqm(state_pqm, j_coeffs, ...):
    if _USE_HARDCODED and set(j_coeffs.keys()) == {2, 3, 4, 5}:
        return _fast_mean_rhs(state_pqm, j_coeffs, ...)
    # fallback to existing loop
```

Similarly, add batch fast-path for SP reconstruction.

### Expected gains

| Component | Stage A | Stage B (est.) | Additional speedup |
|-----------|---------|----------------|--------------------|
| Mean RHS single call | 42 lambdified | 1 CSE function | ~10–30× per call |
| Mean prop total | 40–78 s | **2–8 s** | ~5–10× |
| SP batch | 0.1–0.5 s | **0.05–0.2 s** | ~2–3× |
| **Overall (m+SP)** | 40–79 s | **2–8 s** | ~5–10× |

The CSE win comes from:
- Eliminating ~42 Python function call overheads per RHS (→ 1)
- Sharing subexpressions across terms (powers of q, Q appear repeatedly)
- Pure float arithmetic (no complex, no numpy overhead on scalars)

### Benchmark checkpoint B

Run `extended_validation.py` on all 12 cases. Record:
- `mean_prop_time` (should drop by ~5–10×)
- `sp_time` (minor further improvement from A)
- `meansp_time` total
- Verify all position RMS values match Stage A to < 1e-12 km (exact arithmetic parity)

---

## Stage C — heyoka Compiled Functions (`cfunc`)

### Rationale

heyoka's `cfunc` compiles expression trees to native SIMD machine code via
LLVM. This gives:

1. **Zero Python overhead**: each call is a direct C function pointer invocation
2. **Automatic SIMD vectorization**: batch evaluation over arrays at CPU vector width
3. **Optimal register allocation**: LLVM optimizes the full expression tree

The mean propagation RHS and SP map can each become a single `cfunc` call.

### Implementation

#### C.1 — Build heyoka expression tree for mean rates

In a new module `docs/geqoe_averaged/geqoe_mean/heyoka_compiled.py`:

```python
import heyoka as hy

def build_mean_rates_cfunc(j_coeffs):
    """Build a cfunc computing all 5 mean rates from (q, Q, omega, scale_2..5)."""
    # Declare heyoka variables
    q_hy = hy.expression("q")
    Q_hy = hy.expression("Q")
    omega_hy = hy.expression("omega")
    s2, s3, s4, s5 = [hy.expression(f"s{n}") for n in (2,3,4,5)]

    # Build the same rational expressions from generated_coefficients
    # but as heyoka expression trees
    # ...

    # Compile
    cf = hy.cfunc(
        [g_dot_expr, Q_dot_expr, Psi_dot_expr, Omega_dot_expr, M_dot_expr],
        vars=[q_hy, Q_hy, omega_hy, s2, s3, s4, s5],
    )
    return cf
```

**Key challenge**: Translating 42 SymPy string expressions → heyoka expression
trees. Two approaches:

**(a) Symbolic translation**: Parse each `generated_coefficients.py` string back
to SymPy, then walk the expression tree emitting heyoka operations.
`sp.sympify(expr_str) → recursive emit_hy(expr)`.

**(b) Direct algebraic construction**: Build the Laurent series algebra directly
in heyoka expressions (Legendre polynomials, D^{-n}, ẑ, etc.), mirroring the
symbolic pipeline but in the heyoka expression DSL.

Approach **(a)** is safer and reuses the validated generated expressions.
Write a `sympy_to_heyoka(expr, var_map)` converter.

#### C.2 — Build heyoka cfunc for SP evaluation

Similar structure for the 88 short-period expressions:

```python
def build_sp_cfunc(j_coeffs):
    """Build a cfunc computing 5 SP corrections from (q, Q, f, omega, s2..s5)."""
```

The batch version:
```python
cf_sp = hy.cfunc([dg_expr, dQ_expr, ...],
                 vars=[q, Q, f, omega, s2, s3, s4, s5])
# Call: cf_sp(input_array)  → evaluates over entire array at SIMD speed
```

#### C.3 — Compile-time management

`cfunc` compilation takes time (seconds to minutes depending on expression
complexity). Strategies:
- **One-time build at import**: Cache the compiled object (heyoka may support
  serialization via `cfunc.save()` / `cfunc.load()`)
- **Lazy build**: First call triggers compilation; subsequent calls are instant
- **Startup flag**: `_USE_HEYOKA_CFUNC = True/False` controlled by availability

Check heyoka docs for cfunc caching/serialization support.

#### C.4 — Integration with existing pipeline

Replace the RK4 inner loop in `validation.py`:

```python
def rk4_integrate_mean_compiled(state0, t_eval, cf_mean, j_coeffs, substeps=8):
    """RK4 with cfunc-accelerated RHS."""
    def rhs(y):
        q_val = q_from_g(hypot(y[1], y[2]))
        # ... extract inputs ...
        out = cf_mean(np.array([q_val, Q_val, omega_val, s2, s3, s4, s5]))
        # ... assemble p1_dot, p2_dot from out ...
    # Standard RK4 loop using rhs()
```

For SP batch: pass full arrays to `cf_sp` for SIMD evaluation.

#### C.5 — Optional: heyoka-native RK4/RK8

Instead of Python RK4, use heyoka's `taylor_adaptive` or RK integrator on the
mean-element ODE directly. This would eliminate ALL Python overhead in the mean
propagation loop. The ODE is just 4 autonomous equations in (ḡ, Q̄, Ψ̄, Ω̄)
with Fourier harmonics in ω̄ = Ψ̄ − Ω̄.

This is the ultimate performance path but requires building the full mean ODE
as a heyoka system, not just a cfunc.

### Expected gains

| Component | Stage B | Stage C (est.) | Additional speedup |
|-----------|---------|----------------|--------------------|
| Mean RHS single call | 1 CSE Python | 1 cfunc C call | ~5–20× |
| Mean prop total | 2–8 s | **0.2–1.0 s** | ~5–10× |
| SP batch | 0.05–0.2 s | **0.01–0.05 s** | ~3–5× |
| **Overall (m+SP)** | 2–8 s | **0.2–1.0 s** | ~5–10× |

With C.5 (heyoka-native integration): potentially **< 0.1 s** for mean
propagation, as the entire time-stepping loop runs in compiled code.

### Benchmark checkpoint C

Run `extended_validation.py` on all 12 cases. Record:
- All timings
- Compilation time (one-time cost)
- Verify all position RMS values match Stage B to < 1e-12 km
- Compare against DSST/Brouwer/Cowell timings

---

## Execution Order and Dependencies

```
Stage 0 (baseline) ─── record reference timings
    │
    ▼
Stage A (vectorize SP) ─── SP drops from 20-40s to <1s
    │                       Mean prop unchanged
    │                       Checkpoint A: verify accuracy
    │
    ▼
Stage B (CSE hardcoded) ── Mean prop drops from 40-78s to 2-8s
    │                       SP gets minor further boost
    │                       Checkpoint B: verify accuracy
    │
    ▼
Stage C (heyoka cfunc) ─── Both drop to <1s each
    │                       Checkpoint C: full comparison table
    │
    ▼
Report ──────────────────── Performance summary document
```

### File creation plan

| Stage | New files | Modified files |
|-------|-----------|---------------|
| A | — | `geqoe_mean/short_period.py` (add batch functions), `scripts/extended_validation.py` |
| A | `geqoe_mean/batch_conversions.py` | — |
| B | `geqoe_mean/generate_hardcoded.py` | `geqoe_mean/short_period.py` (add fast-path) |
| B | `geqoe_mean/hardcoded_rates.py` (generated) | — |
| C | `geqoe_mean/heyoka_compiled.py` | `geqoe_mean/validation.py`, `scripts/extended_validation.py` |

### Invariants across all stages

1. **Numerical parity**: All stages must reproduce the baseline position RMS
   to within floating-point noise (< 1e-10 km). The expressions are the same;
   only the evaluation mechanism changes.

2. **Fallback path**: Each stage preserves the previous implementation as a
   fallback (try/except import or feature flag). The scalar loop path is never
   deleted.

3. **Test coverage**: Each new function gets a standalone unit test comparing
   its output against the scalar baseline for at least 3 representative orbits.

---

## Final Report Template

After Stage C, produce a short report (`docs/geqoe_averaged/process/performance_report.md`):

```markdown
# GEqOE Averaged Propagator — Performance Report

## Summary
[1-2 sentence summary of final speedup achieved]

## Timing Comparison (all 12 cases)

| Case | Baseline m+SP | Stage A | Stage B | Stage C | DSST osc | Brouwer |
|------|-------------|---------|---------|---------|----------|---------|
| ...  | ...         | ...     | ...     | ...     | ...      | ...     |

## Accuracy Verification
[Table showing position RMS parity across stages]

## Implementation Notes
- Stage A: vectorized batch SP (N lambdified → N array calls)
- Stage B: CSE-optimized Python (42 calls → 1 call, ~X shared subexpressions)
- Stage C: heyoka cfunc (compiled LLVM, SIMD)

## Compilation Costs (Stage C)
- cfunc build time: X s
- Serialization/caching: [yes/no]

## Conclusions
[Which stages to ship, where the remaining time is spent, path to <0.1s]
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Lambdified functions don't vectorize cleanly (e.g., Python `int` powers) | Test with small array first; fix `sp.lambdify` modules if needed (use `"numpy"` module) |
| CSE output too large to be readable | Generated code doesn't need to be readable; validate numerically |
| SymPy→heyoka expression translation fails on edge cases | Start with n=2 (simplest), validate each degree independently |
| heyoka cfunc compilation takes minutes | Cache compiled objects; lazy-build on first call |
| Complex arithmetic in vectorized path gives different rounding | Convert to real trig in Stage B; carry through to C |
