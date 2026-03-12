# Lessons Learned — Averaged GEqOE Development

Consolidated knowledge from the development of the first-order mixed-zonal
averaged GEqOE theory. Intended to save future developers (human or AI) from
repeating hard-won discoveries.

---

## 1. Mathematical Bugs Found in Plans/Derivations

### K_dot coefficient
The original plan had `ell/alpha = c²/(mu·alpha)` for the K_dot equation —
**wrong**. The correct coefficient is `(1 + alpha·(1 - r/a))` from Eq. 75 of
Baù et al. Compare: dL/dt uses `1/alpha + alpha·(1 - r/a)`, which is different.

### q_dot equations
The plan's formulas for q₁_dot and q₂_dot were **wrong**. Correct from paper
Eqs. 50–51: q̇₁ = (γ/2)·wY, q̇₂ = (γ/2)·wX, where wX = (X/h)·Fh,
wY = (Y/h)·Fh, Fh = F·eZ, γ = 1 + q₁² + q₂².

### K_dot simpler form
K̇ = (a/r)·(L̇ − ṗ₁·cos K + ṗ₂·sin K) — much simpler than the full Eq. 75
expansion. Derive by differentiating the K–L relationship directly.

### Lesson
Always cross-check plan-derived equations against the original paper before
implementing. Transcription errors in multi-step derivations are common.

---

## 2. Symbolic Computation Pitfalls (SymPy)

### `ratint` is unreliable for combined expressions
SymPy's `ratint` fails fatally on combined rational expressions with log terms.
The direct residue-based approach (`short_period_direct.py`) bypasses this.

### `sp.cancel` on scaled polynomials
`sp.cancel(scale * polynomial)` triggers catastrophic multivariate GCD
computation. **Workaround**: distribute the coefficient term-by-term before
calling cancel.

### Deserialized expression cleanup
Running `cancel` + `together` on expressions loaded from serialized form is
redundant and catastrophically slow. The expressions are already simplified
at generation time; re-simplifying wastes hours.

### J5 generation time
Generating J5 short-period coefficients takes ~80 minutes. The bottleneck is
the Ψ and Ω elements, which each have 6 ω-harmonics. Budget computation time
accordingly and cache generated results.

### Expression size scaling
Expression sizes scale ~28× from J3 → J5 (18 KB → 510 KB) due to polynomial
GCD on the Q(q, Q) rational expressions. This is fundamental to the algebraic
structure, not a code issue.

---

## 3. GEqOE-Specific Pitfalls

### ν is nearly constant
GEqOE ν is nearly constant even for osculating elements — it's a poor choice
for visualization. Use classical Keplerian elements (a, e, i) for plots aimed
at physical intuition.

### Altitude oscillates for mean elements too
Mean orbital altitude still oscillates perigee → apogee within each orbit.
This is geometric, not a theory error. Don't use altitude for mean vs.
osculating comparisons.

### Classical a shows the J2 effect clearly
Classical Keplerian `a = −μ/(2E_kep)` from Cartesian state shows dramatic
±6 km short-period oscillation under J2 — this is the right visualization to
demonstrate what averaging removes.

### IC consistency
**Must** use `cart2geqoe(r0, v0, mu, zonal_model)` with the same perturbation
model used for propagation. Using a mismatched model (e.g., `PERT_J2` for a
J2–J5 propagation) introduces large initial errors.

### Near-circular conditioning
Extracting Ψ = atan2(p₁, p₂) amplifies errors by ~1/g (where g = √(p₁²+p₂²)
is the GEqOE eccentricity analog). For near-circular orbits, compare p₁ and p₂
directly rather than converting to angle/magnitude.

### K vs L
Baù's published GEqOE uses L (true longitude analog), but heyoka requires K
(eccentric longitude analog) because L → K involves an implicit Kepler-like
equation that cannot be expressed in heyoka's expression DAG. The averaged
theory uses M (mean longitude) as the fast variable.

---

## 4. Heyoka Integration Patterns

### Automatic STM via `var_ode_sys`
`var_ode_sys(sys, hy.var_args.vars, order=1)` gives 42 DOF (6 state + 36 STM).
Matches finite-difference STM to ~1e-6 relative error.

### Ephemeris JIT thresholds
Use polynomial truncation thresholds ≥1e-4 for Sun (VSOP2013) and ≥1e-2 for
Moon (ELP2000) to keep JIT compilation fast. Default thresholds (1e-9/1e-6)
cause multi-minute compilation times.

### Time conversion
`hy.time - t0` gives relative seconds. Convert to Julian millennia (Sun) or
centuries (Moon) via the epoch Julian date. The `time_origin` parameter in
`build_geqoe_system` ensures the epoch stays tied to the IC epoch even when
the integrator starts at nonzero t₀.

---

## 5. Publication / Cross-Check Pitfalls

### "Singularity-free" is overstated
GEqOE is still singular for retrograde equatorial (i = 180°) and rectilinear
(e = 1) orbits. Claim "singularity-free for prograde non-degenerate orbits."

### "Closed-form propagation" vs "closed-form RHS"
J2-only secular rates are truly closed-form (no ODE needed). General J2–J5
averaging gives a closed-form RHS but requires numerical ODE integration of
the slow flow. Be precise about which is which.

### Bibliography errors
Cross-checking revealed wrong authors, page ranges, and title pluralization
in the bibliography. Always verify references against the original source.

---

## 6. Visualization for Uninitiated Readers

### "Stranger in the hallway" test
Every figure should be comprehensible within 10 seconds by someone who hasn't
read the paper. If it fails this test, simplify.

### Axis labels and titles
Use words on axes, short declarative titles, minimal notation. One plot, one
message — don't overload panels with multiple narratives.

### Build-up strategy
Start with a words-only version of the figure, then add minimal notation back
only where needed. This prevents notation overload.

### TikZ + `\resizebox` incompatibility
TikZ `\matrix` nodes fail inside `\resizebox` due to catcode issues. Use plain
`\node` elements with manual positioning instead.
