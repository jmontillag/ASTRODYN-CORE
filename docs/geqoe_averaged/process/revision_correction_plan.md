# Revision & Correction Plan for *geqoe_averaged_zonal_theory.tex*

**Date**: 2026-03-13
**Document**: `docs/geqoe_averaged/main_docs/geqoe_averaged_zonal_theory.tex` (2645 lines)
**Branch**: `averaged_geqoe`
**Triggered by**: Independent review (two-part, treated as single referee report)

---

## Overview

Two independent LLM-based reviews were conducted on the manuscript. The reviews
agree that the core mathematical framework is sound and the numerical results
are convincing, but they identify:

- **1 derivation-level bug** (w_h formula missing factor γ at L653)
- **1 incorrect proof** (log-cancellation winding-number argument at L1115)
- **1 genuine theory gap** (general-zonal M̄_dot absent from paper, though present in code)
- **Several expository/logical inconsistencies** (error-order language, harmonic bound
  presentation, fast-phase story, numerical ranges)
- **Multiple editorial issues** (notation reuse, internal-note language, missing citations,
  overclaiming)

None of these invalidate the results (the code is correct throughout), but they
collectively make the manuscript not submission-ready.

---

## Task List

### Task 0: Prepare Reviewer Response Letter

**Priority**: Execute alongside or immediately after the corrections.

Prepare a formal point-by-point response to the combined referee report in the
style of a journal revision letter. The letter must:

1. **Format**: One section per reviewer finding, numbered to match the combined
   finding list below (R-1 through R-22). Each section has:
   - **Reviewer comment** (quoted or summarized)
   - **Response** (what was done: agree/disagree, what changed, where)
   - **Manuscript changes** (line numbers or equation numbers of the revised text)

2. **Tone**: Respectful, specific, technical. Acknowledge valid points directly.
   Where we disagree, explain why with equations or references. Never dismiss a
   concern without justification.

3. **Output file**: `docs/geqoe_averaged/process/reviewer_response_letter.tex`
   (LaTeX, compilable standalone).

4. **Execution**: This task is done LAST, after all corrections are implemented,
   so that the response can reference final line numbers and equation numbers.

---

### Tier 1 — Must Fix (Correctness / Logical Coherence)

#### Task 1: Fix w_h formula and show the reduction chain

**Finding references**: R1-High-4 (w_h missing γ), R2-3 (sign/definition chain not shown)

**Problem**:
At L653 the paper writes:
```
w_h = (3A / (c r³)) · c_i · ẑ²  + O(J₂²)
```
where `c_i = δ/γ`. The correct first-order expression is:
```
w_h = (3A δ ẑ²) / (c r³) = (3A γ c_i ẑ²) / (c r³)
```
The paper's formula is missing a factor of γ. Derivation:

```
w_h = (F_h / h)(X q₁ − Y q₂)
    = (F_h / h)(−γ r ẑ / 2)           [from ẑ = 2(Yq₂−Xq₁)/(γr)]
    = (δ ẑ / (2h)) · ∂U/∂ẑ           [substituting F_h = −(δ/(γr)) ∂U/∂ẑ]
```
For J₂: `∂U₂/∂ẑ = 6Aẑ/r³`, so `w_h = 3Aδẑ²/(cr³)` at first order (h≈c).

The code (`fourier_model.py`) computes w_h from the exact definitions and is
correct. The symbolic computations (`symbolic.py`) also work from exact
equations. The bug is isolated to the paper's display formula.

**Impact assessment**:
- At i = 63.4° (critical): γ ≈ 1.38, so the display is off by 28%.
- The reduced equations at L672–688 MUST be checked. If they were derived
  directly from the exact GEqOE equations (as claimed at L671), they are
  correct regardless of L653. If any were derived by substituting L653, they
  inherit the error.
- The J₂ secular rates at L338–357 are classical results quoted independently
  of L653, so they are correct.
- The short-period corrections for Ψ̇ go through `dΨ/dG` at L728–734, which
  contains `d − w_h`. If derived from exact equations, correct; if from L653, wrong.

**Actions**:
1. Fix L653: replace `c_i ẑ²` with `δ ẑ²` (or equivalently `γ c_i ẑ²`).
2. Re-derive the full chain from exact GEqOE equations (L187–208) to the
   reduced first-order equations (L672–688) by hand. Document the intermediate
   algebra.
3. Add a short derivation sketch (inline or appendix) showing:
   - ε_Z = (1−n)U_n = −U for J₂
   - d = −U/c + O(J₂²)
   - w_h = 3Aδẑ²/(cr³) + O(J₂²) [with explicit steps]
   - How these combine into ġ, Q̇, Ω̇, Ψ̇ (L672–688)
4. Verify that the reduced equations L672–688 are numerically correct by
   evaluating both the exact GEqOE RHS and the reduced forms at a test point.
5. Cross-check: confirm that dΨ/dG at L728–734, when evaluated with the
   correct w_h, matches the code's output.

**Verification**: Run the existing validation suite to confirm no regressions.
The code is not changed — only the paper.

---

#### Task 2: Fix the log-cancellation proof

**Finding references**: R1-High-5

**Problem**:
At L1115–1118, the paper states: "the poles F = −q and F = −1/q lie strictly
inside and outside the unit circle respectively, so neither contributes a net
winding number."

This is mathematically false. The unit circle |F| = 1 has winding number **1**
around F = −q (since |−q| = q < 1, the point is inside the contour). The
conclusion (no logarithmic terms) is correct, but the stated proof is wrong.

**Correct argument**:
1. The antiderivative u₁(F) must be single-valued (periodic) on |F| = 1.
2. The integrand `f_v^(n) − ⟨f_v^(n)⟩_M` has zero M-average by construction,
   so its contour integral over |F| = 1 vanishes:
   `∮ [integrand] dF = 0`.
3. By the residue theorem, the sum of residues at poles inside the contour
   (F = 0 and F = −q) must equal zero.
4. The residue at F = 0 is zero because the M-average has been subtracted
   (by construction, the F⁰ term in the Laurent expansion vanishes after
   mean-subtraction).
5. Therefore the residue at F = −q must also vanish.
6. The pole at F = −1/q is outside the contour, so its residue does not
   contribute to the contour integral; however, its coefficient in the
   antiderivative must also vanish for periodicity (since log(1+qF) is
   multi-valued on the unit circle).
7. With all simple-pole residues vanishing, no logarithmic terms arise.

**Actions**:
1. Replace L1115–1118 with the correct residue-theorem argument above.
2. Keep the structural observation that the denominator `(F+q)²(1+qF)²` has
   double poles (not simple poles), so the partial fractions yield `1/(F+q)²`
   and `1/(F+q)` terms; the log-producing `1/(F+q)` residue is the one that
   vanishes.
3. Consider adding a one-paragraph remark: "The vanishing of the simple-pole
   residues is not a coincidence but a structural consequence of the
   zero-mean normalization. It is this structural guarantee that makes the
   residue decomposition method well-posed for all zonal degrees."

---

#### Task 3: Fix the error-order language

**Finding references**: R1-High-2

**Problem**:
The paper uses "O(ε²)" and "O(J²)" inconsistently across three separate
validation contexts, and mixes up what each test actually measures.

Three distinct error sources exist in a first-order averaged theory:

| Error type | Scaling | Where tested |
|---|---|---|
| **Initialization/round-trip** | O(ε²) | L1811–1836 |
| **SP reconstruction per orbit** | O(ε) amplitude | L1840–1851 (slope test) |
| **Secular drift accumulation** | O(ε²) per orbit × t | L1728–1733 (slow-flow) |

Specific inconsistencies:
- **L1731**: Says RHS parity gap "decreases approximately linearly" and calls
  it "the expected signature of an O(J²) remainder." The existing test
  (`zonal_mean_validation.py:110–144`) computes a **relative** RMS gap:
  `RMS(exact − numeric) / max(|numeric|)`. With λ = 0.1 scaling, the
  reported gaps decrease ~10× (slope 1 in log-log). This is consistent with
  an O(J²) **absolute** remainder divided by an O(J) **signal**, yielding
  an O(J) **relative** gap. So the paper's claim is substantively correct
  but the language conflates absolute and relative metrics.
- **L1840**: Says the test checks for "expected O(ε²) residual of a
  first-order theory." But the measured slope is 0.998 (O(ε¹)), and L1848
  correctly explains this is the SP amplitude, not the truncation error.
  The opening sentence is misleading.
- **L1997** (conclusions): Says "scaling linearly with the zonal perturbation
  parameter as expected for a first-order theory" — this is correctly phrased
  but needs to be consistent with earlier sections.

**Diagnostic protocol** (run before writing the fix):

The existing tests are too thin (only 2 scaling factors for the slow-flow
parity gap, only 3 for the SP reconstruction). A proper diagnosis requires
more data points and both absolute and relative metrics.

Step 3a — Augmented slow-flow parity test:
  1. Modify `zonal_mean_validation.py` (or write a small wrapper script) to
     run `pointwise_rhs_validation()` at multiple scaling factors:
     `λ ∈ {1.0, 0.5, 0.25, 0.125, 0.0625}`.
  2. For each λ and each variable (g_dot, Q_dot, Psi_dot, Omega_dot),
     record BOTH:
     - **Absolute gap**: `RMS(exact − numeric)` (no normalization)
     - **Relative gap**: `RMS(exact − numeric) / max(|numeric|)` (existing metric)
  3. Compute log-log slopes for both metrics via `np.polyfit`.
  4. Expected results:
     - Absolute gap slope ≈ 2 → confirms O(J²) remainder
     - Relative gap slope ≈ 1 → confirms O(J²)/O(J) = O(J) relative scaling
  5. If absolute slope ≠ 2: investigate. Possible causes:
     - Numerical quadrature error in `avg_slow_drift` (4097 samples may be
       insufficient at small λ where the gap is tiny)
     - A genuine missing first-order term (would indicate a code bug)
  6. Record the exact slopes and a table of (λ, absolute_gap, relative_gap)
     for each variable. This table goes into the paper.

Step 3b — Augmented SP reconstruction scaling test:
  1. Modify `zonal_short_period_validation.py` (or wrapper) to use more
     scaling factors: `λ ∈ {1.0, 0.5, 0.25, 0.125}` (add 0.125).
  2. Confirm the slope ≈ 1 (SP amplitude dominates) or ≈ 2 (truncation
     dominates). The existing 3-point fit gave 0.998; adding a 4th point
     strengthens the evidence.
  3. Record the updated slope.

Step 3c — Interpret and write:
  Based on the diagnostic results, apply the following paper changes:

**Actions** (after diagnostics):
1. Introduce the three-part error taxonomy explicitly (paragraph or small
   table) at the beginning of Section 11 (Validation), before any results.
   Define clearly:
   - **Initialization error**: O(ε²), from evaluating u₁ at osculating
     instead of mean state. Tested by round-trip (L1811–1836).
   - **Short-period reconstruction error**: O(ε) per orbit, the dominant
     observable. Tested by multi-orbit propagation (Table j2j5_validation).
   - **Secular drift error**: O(ε²) per orbit, accumulates linearly in time.
     Tested by slow-flow parity (L1728–1733) and visible as the growing
     envelope in multi-orbit figures.
2. At L1731: Fix the language based on diagnostic results.
   - If absolute slope ≈ 2 (expected): rewrite to "the **absolute** parity
     gap decreases quadratically with the scaling factor (slope ≈ X.XX in
     log-log), consistent with an O(J²) remainder. The **relative** gap
     decreases linearly (slope ≈ Y.YY), reflecting the O(J) normalization
     of the first-order signal."
   - If absolute slope ≈ 1 (unexpected): investigate root cause and report
     honestly. This would suggest a missing first-order contribution.
3. At L1840: Replace "expected O(ε²) residual" with neutral language:
   "To identify the dominant error source, the validation is repeated with
   the zonal coefficients scaled by factors λ ∈ {1.00, 0.50, 0.25, 0.125}.
   A log-log fit gives slope = X.XX. A slope near 1 indicates that the
   reconstruction error is dominated by the O(ε) short-period amplitude
   (the inherent first-order approximation), not by an O(ε²) truncation
   artifact (which would produce slope ≈ 2)."
4. Ensure the conclusions (L1997) match the corrected taxonomy.
5. Add the diagnostic data (scaling table with slopes) to the paper, either
   inline or in a small supplementary table, so the claim is fully auditable.

**Estimated runtime**: The slow-flow parity test at 5 scaling factors with
16 omega samples each takes ~5 min per factor (dominated by numerical
averaging with 4097 K-samples per orbit). Total: ~25 min. The SP scaling
test at 4 factors takes ~10 min per factor (osculating Taylor integration).
Total: ~40 min. Both can run in parallel.

**Output artifacts**:
- Console output with slopes and gap tables (for pasting into the paper)
- Updated figures if the existing ones only show 2–3 data points

---

#### Task 4: Add general-zonal M̄_dot to the paper

**Finding references**: R2-6

**Problem**:
The mean-state ODE at L1580–1616 gives (ḡ_dot, Q̄_dot, Ψ̄_dot, Ω̄_dot) as
finite-harmonic functions of ω̄, but the mean anomaly rate M̄_dot is absent.
For J₂ alone it is given at L473–477. For the general J₂–J₅ case, M̄_dot is
needed to propagate M̄(t), which is required to solve the Kepler equation for
K̄ at each output epoch (Step 3 of the runtime procedure).

The code DOES compute M̄_dot: `short_period.py:223–268` derives it symbolically
via Laurent series, and precomputed coefficients exist in
`generated_coefficients.py` (keyed by harmonic index for degrees 2–6). So the
gap is in the paper, not in the implementation.

**Actions**:
1. Add a new equation block after L1616, giving M̄_dot in the same
   finite-harmonic form:
   ```
   M̄_dot = ν + M₀(ν̄,ḡ,Q̄) + Σ_m [M terms in sin/cos(m ω̄)]
   ```
   where M₀ contains the secular J₂ piece and the higher-zonal constant
   contributions.
2. Note that M̄_dot has the same parity structure as Ψ̄_dot and Ω̄_dot (even
   harmonics from even zonals, odd from odd), plus the Keplerian ν term.
3. In the runtime procedure (Section 9.2, Step 2, L1289–1302), explicitly
   state that M̄ is integrated alongside the slow state, or (for J₂-only)
   that M̄ advances linearly.
4. In Step 3 (L1310–1324), clarify that K̄ is obtained from M̄ by solving
   the generalized Kepler equation, and then the SP map is evaluated at K̄.

**Verification**: Confirm that the M̄_dot formula written in the paper matches
the code's `generated_coefficients.py` values when evaluated at test points.

---

#### Task 5: Fix numerical performance ranges

**Finding references**: R1-Medium-1

**Problem**:
The stated improvement ranges are materially wrong relative to the paper's own
tables.

**GEqOE vs Brouwer–Lyddane** (Table `extended_validation`, L2125–2136):
| Case | GEqOE pos [km] | Brouwer pos [km] | Ratio |
|------|----------------|------------------|-------|
| Molniya | 3.16 | 12.64 | 4.0× |
| HEO | 1.43 | 6.24 | 4.4× |
| Crit. mod-e | 0.28 | 29.17 | **104×** |
| MEO/GPS | 0.009 | 2.88 | **320×** |

Paper says "4–80×" (L2002) but actual range is **4–320×**. Then L2168 says
"best case is MEO/GPS (335×)" which contradicts the 80× upper bound.

**GEqOE vs DSST** (Table `dsst_osculating`, L13–24 of the input file):
Paper says "1.5–8×" (L2009) but actual range includes Crit. mod-e at 29× and
MEO/GPS at 93×. True range is **1.5–93×**.

**Timing claim** (L2009–2010):
"despite evaluating 10–100× faster" — but measured wall-clock times (Table
`dsst_timing`) show GEqOE Python/SymPy at 23–118 s vs DSST at 0.4–4 s,
i.e. GEqOE is 10–100× **slower** in the current implementation. The claim
refers to theoretical per-epoch cost in a compiled implementation, but this
is not stated.

**Actions**:
1. Recompute all ratios from the published tables. Create a small script or
   spreadsheet to avoid manual errors.
2. At L2002: Change "4–80×" to the correct range (e.g., "4–320×, with a
   median improvement of ~10×" or similar summary statistic).
3. At L2009: Change "1.5–8×" to the correct range (e.g., "1.5–93×").
4. At L2168: Ensure the "best case" number matches the table (320× from the
   table, vs 335× in the text — resolve the discrepancy, likely a rounding
   difference).
5. At L2009–2010: Either (a) remove the "10–100× faster" claim, or (b)
   qualify it explicitly: "The theoretical per-epoch cost of the GEqOE
   evaluation is O(μs) vs O(ms) for DSST, a 10–100× advantage that would
   be realized in a compiled implementation; the current Python prototype
   is slower due to interpreted evaluation (Table X)."
6. Audit the abstract (L58–64) for consistency with the corrected numbers.

---

#### Task 6: Fix harmonic-bound presentation

**Finding references**: R1-High-1

**Problem**:
Three statements about the maximum harmonic order tell different stories:
- L996: `|m| ≤ n−2` (sharp bound, claimed)
- L1427: `M_n = {m ≤ n, m ≡ n mod 2}` (structural overbound)
- L1583: Total model sums `m ≤ N` (overbound for total model)

The code confirms n−2 is correct (`symbolic.py` iterates `range(1, n, 2)` for
odd, `range(2, n, 2)` for even — both stop at n−2). The formal set M_n at
L1427 is a preliminary bound refined later, and the total-model N is an
overbound (extra terms are zero). But the presentation is confusing.

**Actions**:
1. At L996: Keep the claim, but add "(proved in Section X)" forward reference.
2. At L1417–1428: Replace M_n definition with the sharp bound directly:
   `M_n = {m ∈ ℕ₀ : m ≤ n−2, m ≡ n mod 2}`, or introduce M_n with `m ≤ n`
   and immediately add a remark: "The sharper bound |m| ≤ n−2, established
   in Section X.Y, applies; the difference is that the m = n and m = n−1
   harmonics vanish identically upon averaging."
3. At L1580–1616: Change summation bounds from `m ≤ N` to `m ≤ N−2` (the
   sharp bound for the total model), or add a remark that terms with
   m > N−2 have zero coefficients.

---

#### Task 7: Make the paper self-contained

**Finding references**: R1-High-6

**Problem**:
- L1131: `⟨F^k⟩_M = (known closed-form expressions)` — placeholder.
- L1668: Defers J₃/J₄ drift coefficients to companion note.
- L1691: Defers general degree-n generator to companion note.

**Actions**:
1. **L1131**: Replace placeholder with explicit formulas. The ⟨F^k⟩_M are
   obtained by substituting F = (z−q)/(1−qz) and evaluating the contour
   integral at the interior pole z = 0. For k ≥ 1:
   ```
   ⟨F^k⟩_M = (coefficient from binomial expansion of ((z-q)/(1-qz))^k
               evaluated at z=0 with the dM/dz Jacobian)
   ```
   The first few are straightforward; give k = 1, 2, 3 explicitly and
   state the general pattern.
2. **L1668**: Inline the J₃ and J₄ secular drift formulas in the paper (they
   are not excessively long — the J₃ drift is a single cos(ω) harmonic per
   variable, J₄ is constant + sin(2ω)/cos(2ω)). Relegate J₅ to supplementary
   material if space is tight.
3. **L1691**: The general degree-n generator can remain in a companion note,
   but the paper should state the result in closed combinatorial form (even
   if the proof is deferred). Currently the paper says nothing about the
   structure of the general formula.
4. Reframe all companion-note references as "supplementary material" or
   "electronic appendix" rather than by repo path.

---

### Tier 2 — Strong Revision Targets (Clarity / Completeness)

#### Task 8: Clarify fast-phase reconstruction

**Finding references**: R1-High-3

**Problem**:
- L524–530 define u₁ (for slow state z) and v₁ (for fast phase K) as
  separate functions. Steps 1 and 3 (L1268–1317) write u₁ applied to the
  full state y = (z, K), conflating the two.
- The paper claims "K is already available" (L1323) for the GEqOE→Cart
  conversion, but doesn't clarify that Kepler inversion IS needed to get
  K̄ from M̄ during reconstruction.
- The validation at L1774 says "solve the generalized Kepler equation for K"
  without specifying which K (mean or osculating) and why.

**Actions**:
1. In Steps 1 and 3, use both u₁ and v₁ explicitly:
   ```
   z̄(t₀) = z_osc(t₀) − ε u₁(z_osc, K_osc)
   K̄(t₀) = K_osc(t₀) − ε v₁(z_osc, K_osc)
   ```
2. In Step 2 (mean propagation), state explicitly: "The mean fast phase M̄(t)
   is propagated alongside the slow state. At each output epoch, the mean
   eccentric longitude K̄ is recovered from M̄ by solving the generalized
   Kepler equation K̄ − ḡ sin K̄ = M̄."
3. In Step 3, clarify: "The osculating K is obtained directly as
   K_osc = K̄ + ε v₁(z̄, K̄), without an additional Kepler inversion. This
   is the computational advantage of the K-based formulation over the
   L-based formulation of Baù et al."
4. At L1774, change to: "solve the generalized Kepler equation for mean K̄
   from the propagated M̄, then apply the short-period map to obtain the
   osculating state, and convert to Cartesian."

---

#### Task 9: Flag osculating-vs-mean ω transition

**Finding references**: R2-2

**Problem**:
ω = Ψ − Ω is used interchangeably for osculating and mean values. The
critical transition is at L714–734 where the SP integrands use u = ω + f
with ω implicitly frozen at its mean value.

**Actions**:
1. At the beginning of Section 5.2 (L622 or nearby), add: "In the integrands
   that follow, all slow variables — including ω, g, Q — are evaluated at
   their mean values, consistent with the first-order averaging prescription.
   For notational brevity we omit the overbars on these quantities in the
   anomaly-form equations."
2. In Section 7.1 (complex true anomaly), note that w = e^{iω} inherits this
   convention: ω is the mean argument of pericenter when evaluating the
   short-period map.

---

#### Task 10: Support the J₅ indistinguishability claim

**Finding references**: R2-7

**Problem**:
L1807–1809 claims results are "indistinguishable from J₂–J₄" but Table
`j2j5_validation` only shows J₂–J₅. No J₂–J₄ data is provided.

**Actions**:
Either:
(a) Re-run the validation with J₂–J₄ only and add a column to the table, or
(b) Quote the maximum difference numerically: "The maximum position RMS
    difference between J₂–J₄ and J₂–J₅ across all test cases is X meters,
    confirming that J₅ contributes negligibly."
Option (b) is sufficient and avoids table clutter.

---

#### Task 11: Sharpen novelty framing

**Finding references**: R1-Medium-2

**Problem**:
The introduction (L91–117) identifies differences from prior art but doesn't
isolate the real novelties sharply enough.

**Actions**:
Add a paragraph after L117 (or replace L112–117) structured as:

> "The present work differs from all of the above in three specific respects.
> First, it operates directly in the non-canonical GEqOE variables, which are
> free of the circular-orbit and equatorial singularities that plague the
> classical Delaunay and Poincaré formulations. Second, the averaging is
> performed in the generalized eccentric longitude K — not in mean anomaly
> or true longitude — which produces a finite-harmonic mean-element system
> in the single slow angle ω = Ψ − Ω for any zonal truncation order. Third,
> the short-period kernels are constructed by a direct residue decomposition
> in the complex true anomaly F, exploiting the fixed pole structure of the
> dM/dF Jacobian; this replaces both the Hamiltonian Lie series of
> Deprit/Hori and the numerical quadrature of DSST with exact closed-form
> rational expressions."

Contrast explicitly with:
- de Saedeleer (2005): generic zonal, canonical variables, different element set
- Lara (2021): Brouwer redux in Poincaré style, canonical, Lie series
- DSST / San-Juan (2022): equinoctial, numerical quadrature / Gauss-Kronrod
- Baù (2021): defined GEqOE but did not develop averaged theory

---

#### Task 12: Decompose Brouwer comparison into SP vs drift

**Finding references**: R2-10

**Problem**:
The DSST appendix has a clean mean-drift decomposition (Table
`dsst_mean_drift`), but the Brouwer appendix does not. Since Brouwer is
second-order in J₂, a reader might wonder whether the 4–320× advantage is
from better SP reconstruction or just from different arc lengths.

**Actions**:
Add a "Brouwer mean-only" comparison column or paragraph. Run Brouwer with
mean-only output (if Orekit supports it) for a few representative cases, or
note that "Brouwer is a purely analytical theory with no mean-only output
mode; the reported errors combine both secular drift and short-period
reconstruction. The DSST decomposition (Table X) shows that SP reconstruction
dominates the total error budget, and the same is expected for Brouwer."

---

#### Task 13: Qualify or remove the timing claim

**Finding references**: R1-Medium-1 (timing sub-issue)

**Problem**:
L2009–2010: "despite evaluating 10–100× faster" — measured times show the
opposite. The caveat at L2486–2497 explains this, but the unqualified claim
at L2009 is misleading.

**Actions**:
1. At L2009, change to: "despite a theoretical per-epoch cost that is
   10–100× smaller (Table~\ref{tab:dsst_timing}, implementation caveat)."
2. At L2486–2497, strengthen the caveat: state the measured wall-clock
   ratio explicitly ("The current Python/SymPy prototype is X–Y× slower
   than the compiled Java DSST implementation").
3. In the abstract (L48–49), soften "no numerical quadrature, no
   interpolation, no iterative solution" by adding "(except the standard
   generalized Kepler equation)" — which is already there. But check that
   the abstract doesn't contain any timing claims that conflict.

---

#### Task 14: Handle thrust sketch section

**Finding references**: R2-12

**Problem**:
Section 12 (L1926–1965) is a brief sketch with no validation, which weakens
the paper's otherwise rigorous presentation.

**Actions**:
Compress Section 12 into a paragraph at the end of Conclusions (Section 13),
framed as future work:

> "The formulation extends naturally to a Fourier-in-K thrust averaging layer.
> Because r/a, X/a, and Y/a are affine in sin K and cos K, the thrust-induced
> averaged kernels are low-order trigonometric polynomials, and the resulting
> control-influence matrix B(z̄) depends on a finite set of Fourier
> coefficients. Development of this extension is deferred to future work."

Remove the standalone section heading to avoid raising expectations.

---

#### Task 15: Explain averaging weight T = 2πa

**Finding references**: R2-5

**Problem**:
L229–234 uses ⟨f⟩_K = (1/2πa)∫r·f dK without explaining why T = 2πa.

**Actions**:
Add after L234: "Here T = ∫₀²π (r/w) dK = 2πa/w · w = 2πa follows from
⟨r⟩_K = a for the unperturbed ellipse (since the sin K and cos K terms in
r = a(1 − p₁ sin K − p₂ cos K) average to zero). The weight r/a in the
K-average replaces the unit weight of the classical M-average and arises
from the non-uniform parametrization dt/dK = r/w."

---

### Tier 3 — Editorial / Polish

#### Task 16: Replace \date{\today}

**Finding reference**: R1-Medium-4

**Action**: Replace L21 `\date{\today}` with a fixed date or journal-neutral
title block (e.g., `\date{March 2026}` or remove the date line entirely).

---

#### Task 17: Remove internal-note language

**Finding reference**: R1-Medium-4, R2 (implicit)

**Instances**:
- L167: "and the current implementation" → remove or replace with "and is
  given by"
- L186: "the current branch uses" → "the autonomous zonal dynamics are
  described by" or simply "the exact conservative GEqOE system is"
- L272: "the variables used by the present branch" → "the present variables"
- L1531: repo path `docs/geqoe_averaged/zonal_harmonic_probe.py` → remove or
  replace with "a dedicated numerical probe script (see supplementary
  material)"
- L1659: "In the current probe" → "Numerically"
- L1668, L1691, L1699: repo paths → "supplementary note" / "electronic
  appendix"

Grep for "current branch", "current implementation", "current probe",
"docs/geqoe_averaged" and fix all instances.

---

#### Task 18: Fix w notation reuse

**Finding references**: R1-Medium-3, R2-1 (implicit)

**Problem**:
- L200: `w = √(μ/a)` (mean motion-like quantity, used in GEqOE equations)
- L925: `w = e^{iω}` (complex pericenter exponential, used in §7–8)

**Action**: Rename the complex pericenter exponential. Options:
- `ω̂ = e^{iω}` (hat notation)
- `\varpi = e^{iω}` (varpi, visually distinct)
- `W = e^{iω}` (capital)

Preferred: `W = e^{iω}` — simple, unambiguous, easy grep-and-replace.
Propagate through all of Sections 7–8 and the residue decomposition.

---

#### Task 19: Add q-overloading remark

**Finding reference**: R2-1

**Action**: At L932 (after defining q = g/(1+β)), add a parenthetical:
"(The eccentricity parameter q should not be confused with the equinoctial
inclination components q₁, q₂; in the short-period expressions, q always
denotes the eccentricity parameter.)"

---

#### Task 20: Tighten Ψ̇ narrative

**Finding reference**: R2-4

**Action**: At L705–706, after "and therefore (eq) for J₂ with ε_Z = −U",
add: "Equations (86) and (88) are consistent, confirming that the direct
derivation from the ṗ equations reproduces the expected structure."

---

#### Task 21: Add ℓ₁ derivation note

**Finding reference**: R2-8

**Action**: At L847, after eq (ell_v_relation), add a one-line remark:
"Equation (ell_v_relation) follows from expanding L_osc at first order using
∂L/∂K = r/a (eq dLdK); the cross-term v₁(p₁ sin K + p₂ cos K) reduces to
(r̄/a)v₁."

---

#### Task 22: Add n₀ well-definedness note

**Finding reference**: R2-9

**Action**: At L617, after "v₁ is also obtained by quadrature", add:
"(The division by n₀ = w/r is well-defined for all non-rectilinear orbits,
where r > 0 throughout.)"

---

#### Task 23: Add h domain remark

**Finding reference**: R2-11

**Action**: At L201, after defining h = √(c² − 2r²U), add: "The radicand is
positive for perturbations satisfying 2r²|U|/c² ≪ 1, which holds for all
practical zonal truncations."

---

#### Task 24: Cite Phipps

**Finding reference**: R1-Medium-5

**Action**: At L2099, either:
(a) Add a proper bibliography entry for Phipps 1992 (if the reference is
    publicly available), or
(b) Replace "Warren Phipps' 1992 fix" with "Orekit's internal
    critical-inclination correction (documented in the Orekit v13.1 source
    and API documentation~\cite{orekit})."

---

#### Task 25: Audit overclaiming language

**Finding reference**: R1-Medium (overclaiming)

**Action**: Grep for "complete", "exact", "orders of magnitude" and audit
each instance:
- "complete" — acceptable if qualified (e.g., "complete at first order for
  the autonomous zonal problem")
- "exact" — acceptable for the short-period expressions (they ARE exact in
  eccentricity), but not for the theory as a whole (it's first-order)
- "orders of magnitude" — must be backed by specific numbers

---

#### Task 26: Add limitations paragraph

**Finding reference**: R1 (strong revision target)

**Action**: Add a paragraph before or within Conclusions covering:
1. First-order truncation: O(ε²) errors neglected, including J₂² secular
   correction and all cross-zonal products O(J_n J_m).
2. Singular cases: retrograde equatorial (Q → ∞) and rectilinear orbits
   (g → 1, β → 0) are excluded.
3. High-eccentricity degradation: first-order SP errors grow with e due to
   pole proximity (already discussed at L1870–1884, but summarize here).
4. Zonal-only: no tesseral, no third-body, no drag. The autonomous
   restriction is fundamental to the averaging closure.

---

#### Task 27: Cross-reference and compile audit

**Finding reference**: R1 (editorial)

**Action**: After all changes, do one clean compile pass:
1. `latexmk -pdf geqoe_averaged_zonal_theory.tex`
2. Check for undefined references, multiply-defined labels, orphaned figures.
3. Verify all equation cross-references point to the right targets.
4. Check bibliography completeness (every \cite has a \bibitem and vice versa).

---

#### Task 28: Reduce repetitive phrasing

**Finding reference**: R1 (editorial)

**Action**: The phrase "no numerical quadrature, no interpolation, no
iterative solution" (or variants) appears in the abstract, introduction,
Section 9, and conclusions. State it once cleanly in the abstract and once
in Section 9 (operational structure). Remove or soften other occurrences.

---

## Execution Order

| Phase | Tasks | Description |
|-------|-------|-------------|
| **A** | 1, 10 (part of 1) | Fix w_h, show derivation chain. Only paper change. |
| **B** | 2, 3 | Fix proof and error-order language. |
| **C** | 4, 8 | Complete operational theory (M̄_dot, fast-phase). |
| **D** | 5, 13 | Fix all numerical claims and timing. |
| **E** | 6, 7, 10 | Consistency/completeness (harmonics, self-contained, J₅). |
| **F** | 9, 11, 12, 14, 15 | Strong revision targets. |
| **G** | 16–28 | Editorial pass. |
| **H** | 27 | Final compile audit. |
| **I** | 0 | Write reviewer response letter (references final line numbers). |

---

## Mapping: Reviewer Findings → Tasks

For the reviewer response letter, each finding is addressed by one or more
tasks. This table provides the cross-reference.

| Finding ID | Finding (short) | Task(s) | Classification |
|------------|----------------|---------|----------------|
| **R-1** | Harmonic-order contradiction (m≤n vs m≤n−2) | 6 | Presentation fix |
| **R-2** | Error-order inconsistency (O(ε) vs O(ε²)) | 3 | Expository fix |
| **R-3** | Fast-phase reconstruction ambiguity | 8 | Clarification |
| **R-4** | Log-cancellation proof wrong | 2 | Proof rewrite |
| **R-5** | w_h formula missing γ | 1 | Derivation-level fix |
| **R-6** | Paper not self-contained (placeholders, deferrals) | 7 | Content addition |
| **R-7** | Performance ranges numerically inconsistent | 5 | Data correction |
| **R-8** | Novelty positioning too loose | 11 | Framing improvement |
| **R-9** | w notation reuse (√(μ/a) vs e^{iω}) | 18 | Notation fix |
| **R-10** | Internal-note language and repo paths | 17 | Editorial cleanup |
| **R-11** | Reproducibility gaps (Phipps citation) | 24 | Citation addition |
| **R-12** | q notation overloading (q scalar vs q₁,q₂) | 19 | Remark addition |
| **R-13** | ω osculating vs mean not flagged | 9 | Clarification |
| **R-14** | Sign/definition chain A→ε_Z→ṗ not shown | 1 (subtask 3) | Derivation sketch |
| **R-15** | Ψ̇ equations (86)/(88) narrative confusion | 20 | Narrative tightening |
| **R-16** | Averaging weight T=2πa unexplained | 15 | One-line derivation |
| **R-17** | Missing general-zonal M̄_dot | 4 | Theory completion |
| **R-18** | J₅ claim unsupported by table data | 10 | Evidence addition |
| **R-19** | ℓ₁ derivation gap (eq 113) | 21 | One-line remark |
| **R-20** | n₀² division well-definedness | 22 | Parenthetical note |
| **R-21** | Brouwer comparison fairness | 12 | Decomposition note |
| **R-22** | h domain of validity | 23 | Parenthetical note |
| **R-23** | Thrust section too thin for journal | 14 | Restructure to future work |
| **R-24** | Overclaiming ("complete", "exact", "orders of magnitude") | 25 | Language audit |
| **R-25** | \date{\today} on title page | 16 | One-line fix |
| **R-26** | Repetitive phrasing | 28 | Editorial trim |
| **R-27** | Missing limitations paragraph | 26 | Content addition |
| **R-28** | Timing claim misleading | 13 | Qualification |

---

## Notes for the Reviewer Response Letter (Task 0)

The response letter should be organized in the following sections:

### Preamble
Thank the referee for the thorough and constructive review. Acknowledge that
the manuscript was submitted in internal-note form and that the revision
addresses all concerns raised.

### Point-by-point responses

For each R-N finding, the response should follow this template:

> **R-N: [Short title]**
>
> *Reviewer*: [1–2 sentence summary of the concern]
>
> *Response*: [Our assessment: agree / partially agree / disagree with reason.
> What was changed. If we disagree, the mathematical or empirical justification.]
>
> *Changes*: [Specific equation numbers, line ranges, or section references
> in the revised manuscript. E.g., "Eq. (XX) corrected; new Appendix C added;
> Section 11.1 rewritten."]

### Key classification for each response:

- **R-5 (w_h)**: "The referee correctly identified a missing factor of γ in
  the display formula. We have corrected Eq. (XX) and added a derivation
  sketch in Section X.Y. The reduced equations (XX)–(XX) and all numerical
  results are unaffected because the symbolic computation operates on the
  exact GEqOE equations, not on the hand-reduced display formula."

- **R-4 (log-cancellation)**: "The referee is correct that the winding-number
  statement was incorrect. We have replaced it with a residue-theorem
  argument (Section X.Y). The conclusion — absence of logarithmic terms —
  is unchanged."

- **R-2 (error-order)**: "We agree that the error-order discussion was
  internally inconsistent. We have introduced a three-part error taxonomy
  (Table X) and revised the language in Sections X.Y and X.Z to clearly
  distinguish initialization error O(ε²), short-period amplitude O(ε),
  and secular drift accumulation O(ε²)·t."

- **R-7 (numerical ranges)**: "The referee is correct. We have recomputed
  all improvement ratios from the published tables and corrected the stated
  ranges throughout."

- **R-17 (M̄_dot)**: "The referee correctly identified a gap in the paper.
  The general-zonal M̄_dot is implemented in the code but was not written in
  the manuscript. We have added Eq. (XX) and updated the runtime procedure
  description."

- **R-15 (Ψ̇ narrative)**: "The two expressions are algebraically identical
  for J₂ (since ε_Z = −U). The cautionary remark was intended to warn
  against an incorrect derivation path; we have clarified that the two
  forms agree and explained why the chain-rule derivation is preferred."

- **R-21 (Brouwer fairness)**: "The referee raises a valid point. We have
  added a paragraph noting that Brouwer–Lyddane is second-order in J₂ and
  that its errors are dominated by short-period reconstruction rather than
  secular drift, consistent with the DSST mean-drift decomposition shown
  in Appendix B."

- **R-23 (thrust section)**: "We agree that the sketch was too thin for a
  journal paper. We have compressed it into a paragraph in the Conclusions
  section, framed as future work."

---

## Success Criteria

The revision is complete when:
1. All 28 tasks are executed and verified.
2. The manuscript compiles cleanly with no undefined references.
3. The reviewer response letter addresses all 28 findings with specific
   manuscript references.
4. The existing validation suite still passes (code is unchanged).
5. Any new numerical claims (ratios, slopes, M̄_dot values) are verified
   against the code or re-run scripts.
