# Implementation Plan: Lara (2024) Self-Consistent J2–J4 Propagator

## Goal

Upgrade the existing Lara-Brouwer propagator from J2-only SP corrections
(with incomplete radial-only J3-J5) to the full self-consistent J2–J4 theory
from Lara et al. (2024), "Higher-Order Composition of Short- and Long-Period
Effects for Improving Analytical Ephemeris Computation" (CNSNS, arXiv:2307.06864).

This provides a **fair comparison target** for GEqOE: both methods will have
complete first-order J2–J4 short-period corrections, isolating the difference
in the mathematical frameworks (Delaunay/Lie-Deprit vs. GEqOE/residue).

## Background

### Current Lara Implementation (`lara_theory/`)

| Component | File | Status |
|-----------|------|--------|
| W₁ generating function (J2-only) | `short_period.py:373-438` | Done |
| Poisson brackets via heyoka AD | `short_period.py:807-931` | Done |
| Polar-nodal forward map (J2-only) | `short_period.py:1132-1341` | Done |
| Lyddane-space iterative inverse | `short_period.py:670-796` | Done |
| BV energy calibration | `propagator.py:81-130` | Done |
| J2 + J2² secular rates | `mean_elements.py:38-181` | Done |
| J3/J5 frozen secular rates | `mean_elements.py:271-351` | Done |
| J3-J5 radial-only SP (incomplete) | `short_period.py:179-249` | To be replaced |
| Linear mean propagation | `mean_elements.py:354-396` | Done |

### What Lara (2024) Adds

The paper decomposes the zonal problem into three sequential Lie transforms:

1. **W^P (Parallax elimination)** — removes dominant r-dependent oscillations
2. **W^D (Delaunay normalization)** — removes remaining ℓ-dependent oscillations
3. **W^L (Long-period elimination)** — removes ω-dependent slow oscillations

Each has first-order (W₁) and second-order (W₂) terms. The second-order terms
carry J₃ = J₃/J₂ and J₄ = J₄/J₂ contributions.

**Key innovation**: compose all three into a single generating function
W = W₁ + J₂·W₂, evaluate Poisson brackets once (not 3×). Same accuracy,
30%+ faster.

### Scope

- **Harmonics**: J₂, J₃, J₄ (paper excludes J₅ as negligibly small ~10⁻⁷)
- **Periodic corrections**: through second order
- **Secular rates**: through third order
- **Variables**: Delaunay (ℓ, g, h, L, G, H) — canonical
- **Critical inclination**: (5s²-4)⁻¹ denominators remain (inherent to theory)

---

## Phase 1: Transcribe W₂ Generating Function (~2 days)

### 1.1 Second-Order Parallax W₂^P

**Source**: Lara (2024) Eq. 12-13, Tables A.2-A.5

The second-order parallax function contains J₃ and J₄ terms:

```
W₂^P = (G·Re³/p³) · J̃₃ · Σ q_{i,j,k}(s) · e^i · [sin/cos](j·f + k·ω)
      + (G·Re⁴/p⁴) · J̃₄ · Σ Q_{i,j,k}(s) · e^i · [sin/cos](j·f + k·ω)
```

Where:
- J̃₃ = J₃/J₂, J̃₄ = J₄/J₂ (normalized)
- s = sin(i), c = cos(i)
- q_{i,j,k} and Q_{i,j,k} are inclination polynomials from appendix tables
- f = true anomaly, ω = argument of perigee

**Implementation**:
- Create `_evaluate_W2_parallax(ell, g, h, L, G, H, mu, Re, J2, J3, J4)`
- Use same Delaunay input as `evaluate_W1`
- Return scalar W₂^P value
- ~20 polynomial coefficients from Tables A.2-A.5

**Validation**: Compare finite-difference ∂W₂^P/∂ξ against known Keplerian SP
corrections for J₃, J₄ from Brouwer (1959).

### 1.2 Second-Order Delaunay W₂^D

**Source**: Lara (2024) Eq. 18

The second-order Delaunay normalization:

```
W₂^D = (G·Re²/p²)² · [...terms in (3s²-2)², sin f, e·sin 2f, φ²...]
```

Contains the "equation of center" φ = f - ℓ and its square φ². About 10 lines
of trig expressions.

**Implementation**:
- Create `_evaluate_W2_delaunay(ell, g, h, L, G, H, mu, Re)`
- No J₃/J₄ dependence at this stage (pure J₂² effect)
- Return scalar W₂^D value

### 1.3 Long-Period Generating Functions W^L

**Source**: Lara (2024) Eqs. 21-22, 28-29

First-order long-period:
```
W₁^L = (G·Re²/p) · e · s · sin(2ω)
```

Second-order long-period:
```
W₂^L = (G·Re²/p)² · [...terms with (5s²-4)⁻¹ denominators...]
      + J̃₃ terms + J̃₄ terms
```

**WARNING**: Contains `(5s² - 4)⁻¹` denominators — singular at critical
inclination i ≈ 63.435°. This is inherent to the theory. We must:
- Add a guard: if |5s²-4| < ε, skip long-period W₂^L or use a regularized form
- Document this limitation in the propagator docstring

**Implementation**:
- Create `_evaluate_W1_long_period(ell, g, h, L, G, H, mu, Re)`
- Create `_evaluate_W2_long_period(ell, g, h, L, G, H, mu, Re, J2, J3, J4)`
- Tables A.10-A.11 for polynomial coefficients

### 1.4 Compose into Single W

**Source**: Lara (2024) Eqs. 26-29

```python
def evaluate_W_composed(ell, g, h, L, G, H, mu, Re, J2, J3, J4):
    """Full composed generating function W = W₁ + J₂·W₂."""
    # First order (existing + long-period)
    W1_SP = evaluate_W1(ell, g, h, L, G, H, mu, Re)  # existing
    W1_LP = _evaluate_W1_long_period(ell, g, h, L, G, H, mu, Re)
    W1 = W1_SP + W1_LP

    # Second order
    W2_P = _evaluate_W2_parallax(ell, g, h, L, G, H, mu, Re, J2, J3, J4)
    W2_D = _evaluate_W2_delaunay(ell, g, h, L, G, H, mu, Re)
    W2_L = _evaluate_W2_long_period(ell, g, h, L, G, H, mu, Re, J2, J3, J4)
    W2 = W2_P + W2_D + W2_L

    return W1, J2 * W2
```

**File**: Add all new functions to `lara_theory/short_period.py` (or a new
`lara_theory/generating_functions.py` to keep the file manageable).

### 1.5 Inclination Polynomial Tables

**Source**: Lara (2024) Tables A.2-A.11

These are ~10 tables, each with 5-15 polynomial coefficients in sin²(i).
Transcribe as Python dictionaries:

```python
# Example from Table A.2 (J₃ parallax, short-period)
_Q_J3_SP = {
    # (e_power, f_harmonic, omega_harmonic): coefficient_function(s²)
    (0, 1, 1): lambda s2: (3/8) * s2 * (5*s2 - 4),
    (1, 0, 1): lambda s2: -(1/4) * s2 * (5*s2 - 4),
    # ...
}
```

**Risk**: This is the most error-prone step. Mitigation:
- Double-check each coefficient against the paper
- Cross-validate W₂ against the paper's own figures (TOPEX: ±20 cm)
- Unit-test individual polynomial entries

---

## Phase 2: Second-Order Poisson Brackets (~1 day)

### 2.1 First-Order Brackets (existing)

Already implemented: `{ξ, W₁}` for ξ ∈ {r, ṙ, u, rf̊, Ω, I}.

Uses the formula (from `_build_sp_polar_heyoka_cfunc`, line 1132):
```
{ξ, W₁} = (ξ_E·W₁_L - ξ_L·W₁_E)/Δ + ξ_g·W₁_G - ξ_G·W₁_g
         + σ_G·(ξ_g·W₁_E - ξ_E·W₁_g)
```

### 2.2 Second-Order Brackets (new)

The second-order mean-to-osculating correction:
```
δ²ξ = {ξ, W₂} + ½{{ξ, W₁}, W₁}
```

**{ξ, W₂}**: Same Poisson bracket formula as {ξ, W₁} but with W₂ instead
of W₁. Since W₂ is just another scalar function of (ℓ, g, h, L, G, H),
the heyoka AD machinery handles it identically.

**{{ξ, W₁}, W₁}**: The double bracket. Compute y₁ = {ξ, W₁} first (already
done), then compute {y₁, W₁}. With heyoka this is automatic — y₁ is itself
an expression in the DAG, and heyoka can differentiate it again.

**Implementation**:
- Extend `_build_sp_polar_heyoka_cfunc` to accept W₂ as an additional expression
- Build a new cfunc that outputs: `[δ¹r + J₂·δ²r, δ¹ṙ + J₂·δ²ṙ, ...]`
- The double bracket `{{ξ, W₁}, W₁}` requires heyoka to differentiate the
  first bracket expression w.r.t. the Delaunay variables again — this is
  where heyoka's AD capability pays off (no manual derivation needed)

**New function**: `_build_sp_polar_heyoka_cfunc_2nd(mu, Re, J2, J3, J4)`
- Input: `[E, g, L, G, H]` (same as existing)
- Output: `[δr, δṙ, δu, δ(rf̊), δΩ, δI]` (now through second order)

### 2.3 Lyddane-Space Second-Order Brackets

Similarly extend `_build_sp_heyoka_cfunc` to include W₂:
- Output: `[da, d(ecosω), d(esinω), dI, dΩ, d(M+ω)]` through second order
- Used by the iterative inverse

---

## Phase 3: Third-Order Secular Rates (~0.5 day)

### 3.1 Current Secular Rates

The existing code (`mean_elements.py`) computes:
```
K = H₀,₀ + J₂·H₀,₁ + (J₂²/2)·H₀,₂
```

and takes ∂K/∂(L, G, H) for secular rates. This is J₂ through second order.

### 3.2 Add J₃ and J₄ to the Averaged Hamiltonian

From Lara (2024), the averaged Hamiltonian includes:

```
K = H₀,₀ + J₂·H₀,₁ + (J₂²/2)·H₀,₂
  + J₃·H₁,₃ + J₄·H₁,₄          ← first-order J₃, J₄
  + J₂·J₃·H₂,₃ + J₂·J₄·H₂,₄    ← second-order cross terms
  + (J₂³/6)·H₀,₃                  ← third-order J₂
```

**H₁,₃ (first-order J₃)**: Contains sin(ω) dependence — NOT averaged to zero
for odd harmonics. This gives the "frozen eccentricity" secular rate in ω.

**H₁,₄ (first-order J₄)**: Averages like J₂ — contributes to dl/dt, dg/dt, dh/dt.

**Implementation**:
- Extend `total_averaged_hamiltonian()` in `mean_elements.py`
- Add H₁,₃, H₁,₄ and cross-terms
- Recompute ∂K/∂(L, G, H) via heyoka AD (existing infrastructure)
- Replace the current "frozen J₃/J₅ numerical averaging" with exact symbolic terms

### 3.3 Validate Secular Rates

Compare against numerical one-revolution averaging (already have this
infrastructure in `_orbit_averaged_Rn`). The new analytical rates should
match the numerical average to ~1e-12.

---

## Phase 4: Integration into Propagator (~0.5 day)

### 4.1 New Propagator Mode

Add a new mode to `LaraBrouwerPropagator`:

```python
class LaraBrouwerPropagator:
    def __init__(self, mu, Re, j_coeffs, use_w1_sp=False, use_lara2024=False):
        self.use_lara2024 = use_lara2024
        # ...
```

When `use_lara2024=True`:
1. **Initialization**: Use the composed W = W₁ + J₂·W₂ for osc→mean inverse
2. **Propagation**: Use third-order secular rates (including J₃, J₄ terms)
3. **Reconstruction**: Use second-order polar-nodal SP from composed W

### 4.2 BV Calibration Update

The BV correction currently uses H₀,₁ + (J₂²/2)·H₀,₂. With the extended
Hamiltonian, update to include H₁,₃ and H₁,₄ terms:

```python
sum_Hm = J2 * H01 + 0.5 * J2**2 * H02 + J3 * H13 + J4 * H14
```

### 4.3 Remove Legacy J3-J5 Radial SP

When `use_lara2024=True`, disable the old `brouwer_sp_polar_batch` path
(which only corrects the radial component). The new second-order SP from
the composed W handles all components consistently.

---

## Phase 5: Testing and Validation (~1.5 days)

### 5.1 Unit Tests

| Test | Description | Expected |
|------|-------------|----------|
| W₂ round-trip | osc → mean → osc via 2nd-order W | < 1 m everywhere |
| Polynomial parity | q_{i,j,k} match paper tables | exact |
| Secular rate parity | Analytical vs numerical averaging | < 1e-12 |
| BV calibration | Energy conservation check | < 1e-10 |
| Critical inclination guard | i = 63.4° doesn't crash | no NaN |

### 5.2 Topex Validation (Primary)

Run the Topex orbit (a=7707 km, e=0.0001, i=66.04°) for 30 days:
- **Target**: RSS < 30 m (paper reports ~20 m for the composed theory)
- **Comparison**: Current Lara W₁ J2-only gives 11 m (but J2-only truth)
- The 2024 theory should give comparable accuracy against J2-J4 truth

### 5.3 Grid Heatmap Re-run

Re-run `scripts/grid_comparison.py` with the new Lara 2024 propagator:
- Expect Lara to improve significantly at moderate-to-high eccentricities
- GEqOE should still win overall (equinoctial regularity advantage)
- The gap should narrow, especially in the a-vs-e and e-vs-i planes

### 5.4 12-Orbit Comparison Re-run

Re-run `scripts/lara_comparison.py` with the new propagator:
- Update Table 3 in the paper with Lara 2024 results
- Document the improvement per regime

---

## File Changes Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `lara_theory/short_period.py` | Add W₂ functions, extend heyoka cffuncs | +400 |
| `lara_theory/mean_elements.py` | Add H₁,₃, H₁,₄, cross-terms, 3rd-order | +100 |
| `lara_theory/propagator.py` | Add `use_lara2024` mode | +30 |
| `lara_theory/polynomial_tables.py` | NEW: inclination polynomial data | +200 |
| `tests/test_lara_theory.py` | Add 2nd-order validation tests | +100 |
| `scripts/grid_comparison.py` | Add Lara 2024 column | +20 |

**Total**: ~850 new lines

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Transcription error in polynomial tables | High | High | Cross-validate against paper figures |
| Critical inclination blow-up | Certain | Medium | Guard with |5s²-4| threshold |
| heyoka double-bracket compilation time | Medium | Low | Cache aggressively; ~30s one-time cost |
| J₅ exclusion unfairness | Low | Low | Note in paper: J₅ ~ 10⁻⁷ negligible |
| W₂ + W₁ composition error | Medium | High | Validate W₁-only path still matches existing results |

---

## Verification Checklist

Before declaring complete:

- [ ] `evaluate_W1` still gives identical results (regression)
- [ ] W₂^P returns zero when J₃ = J₄ = 0 (J₂-only should reduce to W₁)
- [ ] Topex 30-day RSS < 30 m (against J₂-J₄ Cowell truth)
- [ ] PRISMA orbit (a=6878 km, e=0.001, i=97.4°) < 50 m
- [ ] GTO (a=24500 km, e=0.73, i=7°) < 100 m
- [ ] Critical inclination cases don't crash (with guard)
- [ ] Grid heatmaps regenerated with Lara 2024 mode
- [ ] All 437 existing tests still pass
- [ ] Paper updated with new comparison results

---

## Reference

- Paper PDF: `docs/geqoe_averaged/references/lara2024.pdf`
- Text extract: `docs/geqoe_averaged/references_txt/lara2024.txt`
- Existing Lara code: `docs/geqoe_averaged/lara_theory/`
- Existing comparison: `docs/geqoe_averaged/scripts/lara_comparison.py`
- Grid comparison: `docs/geqoe_averaged/scripts/grid_comparison.py`
