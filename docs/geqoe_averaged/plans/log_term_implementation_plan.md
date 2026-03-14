# Plan: Include Log Terms in the Short-Period Correction

## Goal

Add the log(1+qF) correction to the short-period map for secular-rate
harmonics, making u₁ the **complete** first-order antiderivative.  This
gives "true orbit-average" mean elements and is a prerequisite for correct
second-order secular rates.

## Mathematical Summary

For secular-rate harmonics, the complete antiderivative on |F|=1 is:

    u₁_full(F) = u₁_rational(F) + C(q,Q) · φ(K)

where:
- C(q,Q) = 2(1-q²)³/(1+q²) · E₁(q,Q) is a rational function (real for m=0,
  complex for |m|>0)
- φ = arctan2(q·Im(F), 1 + q·Re(F)) is a smooth periodic function of K
- E₁ is the simple-pole residue at F=0 (already computed in the investigation)

The φ function has the Fourier expansion:
    φ(f) = Σ_{n≥1} (-1)^{n+1} qⁿ/n · sin(nf)

## Architecture

The log correction is additive and uses the SAME evaluation infrastructure
(w-harmonic sum, Re extraction).  For each secular-rate harmonic:

    total_m += C_log(q_val, Q_val) · phi_val · w_val^m

This adds exactly **one atan2 + one complex multiply** per secular-rate
harmonic per evaluation — negligible cost.

---

## Phase 1: Symbolic Layer (`direct_residue.py`)

### Step 1a: Compute log residues in `integrate_harmonic_residue()`

Currently returns: `u1_rational` (SymPy expression).
New return: `(u1_rational, log_coeff)` where `log_coeff` is either `None`
(zero-mean harmonic) or `C_log` = `scale · E₁ · 2I` (SymPy expression in q, Q).

Changes:
1. After the Taylor recurrence (line ~270), extract `E₁ = chi_taylor[shift-1]`
   for j=1 (currently skipped with comment "j=1 gives log(F), which cancels")
2. Compute `C_log = scale * E1 * 2 * sp.I` where scale = `-I*(1-q²)³/(1+q²)`
3. Simplify with `_fast_cancel()`
4. Return `(u1_rational, C_log)` if E₁ ≠ 0, else `(u1_rational, None)`

**Backward compatibility**: Callers that expect a single expression will break.
All call sites must be updated (Phase 2).

### Step 1b: Add `compute_log_coefficient()` standalone function

For regeneration of `generated_coefficients.py`, add a standalone helper:

```python
def compute_log_coefficient(raw_by_k, mean_coeff):
    """Compute the log-term coefficient C_log for one harmonic.

    Returns C_log such that:
        u1_full = u1_rational + C_log * arctan2(q*Im(F), 1+q*Re(F))
    Returns None if the harmonic has zero mean (log terms vanish).
    """
```

---

## Phase 2: Storage Layer (`generated_coefficients.py`)

### Step 2a: Add LOG_DATA dictionary

Add a new top-level dict alongside `MEAN_DATA` and `SHORT_DATA`:

```python
LOG_DATA = {
    2: {
        "Psi": {"0": "<C_log SymPy string>"},
        "Omega": {"0": "<C_log SymPy string>"},
    },
    3: {
        "g": {"-1": "...", "1": "..."},
        "Q": {"-1": "...", "1": "..."},
        "Psi": {"-1": "...", "1": "..."},
        "Omega": {"-1": "...", "1": "..."},
        "M": {"-1": "...", "1": "..."},
    },
    # 4, 5 similarly
}
```

Only secular-rate harmonics have entries.  Zero-mean harmonics are absent
(C_log = 0 structurally).

### Step 2b: Regeneration script

Add a generation pass in the existing `_regenerate_coefficients.py` (or
similar) that computes C_log for each (n, variable, m) and serializes
alongside SHORT_DATA.

---

## Phase 3: Evaluation Layer (`short_period.py`)

### Step 3a: Parse log coefficients

In `_lambdified_short_period_expressions()` (or equivalent loader):
- Parse `LOG_DATA[n][variable][m_str]` into SymPy, then lambdify as
  `C_log_func(q, Q) -> complex`
- Store alongside the rational lambdified functions

### Step 3b: Add log term to `evaluate_isolated_degree_short_period()`

After the w-harmonic sum loop (line ~516):

```python
# Compute φ once per evaluation (shared by all variables/harmonics)
phi = np.arctan2(q_val * F_val.imag, 1.0 + q_val * F_val.real)

# For each variable, add log contributions
for m_val, C_log_func in log_lambdified[variable].items():
    C_log_val = complex(C_log_func(q_val, Q_val))
    total += C_log_val * phi * (w_val ** m_val)
```

### Step 3c: Batch evaluation fallback

Same pattern in `evaluate_truncated_short_period_batch()` fallback path:
vectorized `np.arctan2(q_arr * F_arr.imag, 1 + q_arr * F_arr.real)`.

### Step 3d: No changes needed to higher-level functions

`osculating_to_mean_state()`, `mean_to_osculating_state()`, and their batch
variants all call `evaluate_truncated_short_period()` → automatically inherit
log support.

---

## Phase 4: Compiled Path (`heyoka_compiled.py`)

### Step 4a: Add φ computation to cfunc

In `build_sp_cfunc()`, after the Chebyshev recurrence setup:

```python
# φ = arctan2(q * sin(f), 1 + q * cos(f))
# sin(f) and cos(f) are the first-order Chebyshev terms
phi_hy = hy.atan2(q_hy * sin_f_tab[1], 1.0 + q_hy * cos_f_tab[1])
```

### Step 4b: Add log contributions to rate expressions

For each (n, variable, m) with a LOG_DATA entry:

```python
C_log_sp = sp.sympify(log_data_str, locals={"q": q_sym, "Q": Q_sym})
C_log_re, C_log_im = _sympy_to_heyoka_complex(C_log_sp, var_map, ...)
# log contribution = C_log * phi * w^m
log_contrib_re = (C_log_re * cos_m - C_log_im * sin_m) * phi_hy
log_contrib_im = (C_log_re * sin_m + C_log_im * cos_m) * phi_hy
rates[key] += s_hy[n] * log_contrib_re  # (for real extraction)
```

---

## Phase 5: Equinoctial Route (`equinoctial_sp_feasibility.py`)

### Step 5a: Adapt `compute_equinoctial_sp()` to new return format

The function calls `integrate_harmonic_residue()` at line 104.  Update to
handle `(rational, C_log)` tuple.  Store log coefficients for equinoctial
harmonics.

### Step 5b: Add log terms to `evaluate_equinoctial_sp_numerical()`

Same φ computation, same additive pattern.  The equinoctial route uses the
same φ = arctan2(q*Im(F), 1+q*Re(F)).

---

## Phase 6: Validation

### Step 6a: Derivative check

Run `_verify_log_correction.py` — should now show 0% error for ALL
harmonics (not just zero-mean), since the log terms are included.

### Step 6b: Round-trip self-consistency

Run `_verify_self_consistency.py` — round-trip error should remain O(ε²)
(or improve slightly, since the transformation is now the complete
first-order map).

### Step 6c: Scaling diagnostic

Re-run `scaling_diagnostic.py` — slopes should remain 2.0 (the self-
consistency is preserved; including the log terms doesn't change the order).

### Step 6d: Full validation suite

Run `extended_validation.py` — expect results to be very close to current
(differences are O(ε²) which is below the first-order error).  Any
significant changes would indicate a bug.

### Step 6e: Equinoctial comparison

Re-run `equinoctial_sp_feasibility.py` — the equinoctial and polar routes
should still agree to machine precision.

---

## Phase 7: Documentation Updates (after validation passes)

### Main paper (`geqoe_averaged_zonal_theory.tex`)
- Update Step 4 assembly equation to include the log term
- Strengthen the structural vanishing proof
- Remove "rational-only" caveats from the self-consistency paragraph
- Update abstract if needed

### Second-order note (`geqoe_averaged_second_order.tex`)
- Update S3 to state u₁ is now complete (not rational-only)
- Simplify the log-term item (no longer needs "advisable to include")

### Equinoctial alternatives (`equinoctial_sp_alternatives.tex`)
- Add subsection on log terms in the equinoctial formulation
- Note that the arctan φ function is shared between polar and equinoctial
- Confirm the g→0 regularity extends to log coefficients

### Investigation plan
- Mark as fully resolved (implementation complete)

---

## Execution Order

1. **Phase 1** (1h): Modify `direct_residue.py` — new return format
2. **Phase 2** (1h): Regenerate `generated_coefficients.py` with LOG_DATA
3. **Phase 3** (1h): Update `short_period.py` evaluation chain
4. **Phase 4** (1h): Update `heyoka_compiled.py` cfunc
5. **Phase 5** (30m): Update equinoctial feasibility script
6. **Phase 6** (1-2h): Run full validation suite
7. **Phase 7** (1h): Update documentation

Total estimated effort: ~1 day

## Key Risks

1. **Regenerating coefficients**: The symbolic computation for n=4,5 is slow
   (~80 min).  LOG_DATA computation is much cheaper (just extracting E₁
   from the existing Taylor recurrence), but must be run alongside.

2. **Heyoka cfunc cache invalidation**: The cfunc is cached; adding log
   terms requires rebuilding.  Ensure the cache key changes.

3. **Backward compatibility**: Any code that calls `integrate_harmonic_residue()`
   directly must be updated for the new return format.  All known call sites
   are within the geqoe_mean package.
