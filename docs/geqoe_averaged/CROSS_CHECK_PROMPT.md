# Cross-Check and Validation Prompt for `geqoe_averaged_j2_zonal_note.tex`

## Objective

Systematically verify every claim, citation, attribution, and number in the
document against the actual reference PDFs (in `docs/geqoe_averaged/references/`)
and the codebase. The goal is to catch hallucinations, wrong attributions,
misquoted numbers, incorrect paper metadata, and unsupported claims.

All references are available as PDFs in `docs/geqoe_averaged/references/`,
named by their bibkey (e.g., `brouwer1959.pdf`).

---

## Part 1: Bibliography Metadata Verification

For each entry in `\begin{thebibliography}`, open the corresponding PDF and
verify **all** of the following fields against the paper's actual title page /
header:

| Bibkey | Check | What to verify |
|--------|-------|----------------|
| `bau2021` | Authors | "Baù, G., Hernando-Ayuso, J., Bombardelli, C." — correct names, order, spelling? |
| `bau2021` | Title | "A generalization of the equinoctial orbital elements" — exact? |
| `bau2021` | Journal | *Celestial Mechanics and Dynamical Astronomy* — correct? |
| `bau2021` | Year/Vol/Pages | 2021, 133, 50 — correct? (note: article number, not page range) |
| `brouwer1959` | Authors | "Brouwer, D." — correct? |
| `brouwer1959` | Title | "Solution of the problem of artificial satellite theory without drag" — exact? |
| `brouwer1959` | Journal | *The Astronomical Journal* — correct? |
| `brouwer1959` | Year/Vol/Pages | 1959, 64, 378–397 — correct? |
| `kozai1959` | Authors | "Kozai, Y." — correct? |
| `kozai1959` | Title | "The motion of a close earth satellite" — exact? |
| `kozai1959` | Journal | *The Astronomical Journal* — correct? |
| `kozai1959` | Year/Vol/Pages | 1959, 64, 367–377 — correct? |
| `deprit1969` | Authors | "Deprit, A." — correct? |
| `deprit1969` | Title | "Canonical transformations depending on a small parameter" — exact? |
| `deprit1969` | Journal | *Celestial Mechanics* — correct? |
| `deprit1969` | Year/Vol/Pages | 1969, 1, 12–30 — correct? |
| `hori1966` | Authors | "Hori, G." — correct? First name initial? |
| `hori1966` | Title | "Theory of general perturbation with unspecified canonical variable" — exact? |
| `hori1966` | Journal | *Publications of the Astronomical Society of Japan* — correct? |
| `hori1966` | Year/Vol/Pages | 1966, 18, 287–296 — correct? |
| `broucke1972` | Authors | "Broucke, R.A., Cefola, P.J." — correct names, order? |
| `broucke1972` | Title | "On the equinoctial orbit elements" — exact? |
| `broucke1972` | Journal | *Celestial Mechanics* — correct? |
| `broucke1972` | Year/Vol/Pages | 1972, 5, 303–310 — correct? |
| `lyddane1963` | Authors | "Lyddane, R.H." — correct? |
| `lyddane1963` | Title | "Small eccentricities or inclinations in the Brouwer theory of the artificial satellite" — exact? |
| `lyddane1963` | Journal | *The Astronomical Journal* — correct? |
| `lyddane1963` | Year/Vol/Pages | 1963, 68, 555–558 — correct? |
| `depritrom1970` | Authors | "Deprit, A., Rom, A." — correct names, order? |
| `depritrom1970` | Title | "The main problem of artificial satellite theory for small and moderate eccentricities" — exact? |
| `depritrom1970` | Journal | *Celestial Mechanics* — correct? |
| `depritrom1970` | Year/Vol/Pages | 1970, 2, 166–206 — correct? |
| `coffey1982` | Authors | "Coffey, S.L., Deprit, A." — correct names, order? |
| `coffey1982` | Title | "Third-order solution to the main problem in satellite theory" — exact? |
| `coffey1982` | Journal | *Journal of Guidance, Control and Dynamics* — correct? |
| `coffey1982` | Year/Vol/Pages | 1982, 5(4), 366–371 — correct? |
| `desaedeleer2005` | Authors | "de Saedeleer, B." — correct? Any co-authors? |
| `desaedeleer2005` | Title | "Complete zonal problem of the artificial satellite: generic compact analytic first order in closed form" — exact? |
| `desaedeleer2005` | Journal | *Celestial Mechanics and Dynamical Astronomy* — correct? |
| `desaedeleer2005` | Year/Vol/Pages | 2005, 91, 239–268 — correct? |
| `lara2021` | Authors | "Lara, M." — correct? Any co-authors? |
| `lara2021` | Title | "Brouwer's satellite solution redux" — exact? |
| `lara2021` | Journal | *Celestial Mechanics and Dynamical Astronomy* — correct? |
| `lara2021` | Year/Vol/Pages | 2021, 133, 47 — correct? |
| `cefola1972` | Authors | "Cefola, P.J." — correct? Any co-authors? |
| `cefola1972` | Title | "Equinoctial orbit elements—application to artificial satellite orbits" — exact? |
| `cefola1972` | Venue | AIAA Paper 72-937 — correct paper number? |
| `sanjuan2022` | Authors | "San-Juan, J.F., San-Martín, M., Pérez, I." — correct names, order, accents? |
| `sanjuan2022` | Title | "A second-order closed-form J₂ model for the Draper Semi-Analytical Satellite Theory" — exact? |
| `sanjuan2022` | Journal | *The Journal of the Astronautical Sciences* — correct? |
| `sanjuan2022` | Year/Vol/Pages | 2022, 69, 1292–1332 — correct? |
| `mahajan2019` | Authors | "Mahajan, B., Alfriend, K.T." — correct names, order? |
| `mahajan2019` | Title | "Analytic orbit theory with any arbitrary spherical harmonic as the dominant perturbation" — exact? |
| `mahajan2019` | Journal | *Celestial Mechanics and Dynamical Astronomy* — correct? |
| `mahajan2019` | Year/Vol/Pages | 2019, 131, 45 — correct? |

---

## Part 2: Citation-Context Claims

For each `\cite{}` usage in the document, verify the specific claim made about
the cited work. Open the referenced PDF and confirm.

### Line 57: `\cite{bau2021}`
> "Generalized Equinoctial Orbital Elements (GEqOE) of Baù et al."

**Check**: Does Baù et al. (2021) actually introduce/define GEqOE? Is "GEqOE"
their terminology or adopted later? What do they call their element set?

### Line 75: `\cite{deprit1969,hori1966}`
> "the Lie–Deprit–Hori perturbation framework"

**Check**: Is it standard to call this the "Lie–Deprit–Hori framework"? Deprit
(1969) develops canonical transformations via Lie series. Hori (1966) develops
perturbation theory with unspecified canonical variables. Are these correctly
attributed as the same framework, or are they distinct approaches that are often
conflated? Read both papers' abstracts and introductions to confirm.

### Lines 83–84: `\cite{brouwer1959}` and `\cite{kozai1959}`
> "The classical analytical treatment of the zonal satellite problem originates
> with Brouwer and Kozai, who derived first-order secular and short-period
> corrections in Delaunay variables."

**Check**:
1. Does Brouwer (1959) work in Delaunay variables specifically?
2. Does Kozai (1959) work in Delaunay variables?
3. Is "first-order" accurate for Brouwer? (Brouwer's theory is often described
   as going to second order in J2.)
4. Did both derive short-period corrections, or only Brouwer?

### Lines 85–86: `\cite{coffey1982}`
> "Subsequent developments by Coffey and Deprit extended the theory to third order"

**Check**: Does Coffey & Deprit (1982) actually present a third-order solution?
Verify from the paper's abstract/title. (Title says "Third-order solution" so
this is likely correct, but confirm what "third order" means — third order in
J2? Third order in the small parameter?)

### Lines 86–88: `\cite{desaedeleer2005}`
> "de Saedeleer provided a generic compact first-order closed-form solution
> for any individual zonal harmonic"

**Check**: Does de Saedeleer (2005) actually provide closed-form solutions for
arbitrary zonal degree n? Is it truly "generic" (any n) or limited to specific
degrees? Read the abstract and main results. The paper title says "generic
compact analytic first order in closed form" which supports this, but verify.

### Lines 88–89: `\cite{lara2021}`
> "Lara recently revisited the complete Brouwer solution using a single
> canonical transformation in Poincaré style"

**Check**:
1. Does Lara (2021) use a "single canonical transformation"?
2. Is "Poincaré style" accurate? Does Lara use that terminology?
3. Does the paper "revisit the complete Brouwer solution" or does it do
   something different/more?

### Lines 91–92: `\cite{cefola1972,sanjuan2022}`
> "the Draper Semi-analytical Satellite Theory (DSST)"

**Check**:
1. Is Cefola (1972) actually about DSST, or is it about equinoctial elements
   more generally? The DSST was developed at Draper Labs by multiple authors.
   Is Cefola the right primary citation?
2. Does San-Juan et al. (2022) actually work within the DSST framework?
3. Is "Semi-analytical" hyphenated in the official name?

### Line 92: `\cite{broucke1972}`
> "developed the semi-analytical approach in non-singular equinoctial elements"

**Check**: Does Broucke & Cefola (1972) introduce equinoctial elements, or do
they build on earlier work? The phrase "non-singular equinoctial elements" — is
this the terminology used in the paper?

### Line 145: `\cite{bau2021}`
> "The generalized angular momentum proxy used throughout [Baù et al.]"

**Check**: Does Baù et al. (2021) define c = (μ²/ν)^{1/3} β as a "generalized
angular momentum proxy"? What do they call it? Verify the formula matches.

### Lines 313–315: `\cite{brouwer1959,kozai1959}`
> "the secular drift of the mean GEqOE angular variables coincides with the
> classical J₂ secular drift written in the GEqOE angles Ψ and Ω"

**Check**: Verify the specific secular rate formulas (Eqs. 18–19 in the doc)
against Brouwer (1959) and Kozai (1959). The formulas should be equivalent to:
- ω̇_Ω = -(3/2) n J₂ (R_e/p)² cos i
- ω̇_ω = (3/4) n J₂ (R_e/p)² (5cos²i - 1)

Note the document uses ω̇_Ψ = ω̇_ω + ω̇_Ω (longitude of pericenter rate), not
ω̇_ω (argument of pericenter rate). Verify the formula
`(5c_i² - 2c_i - 1)` is consistent with the classical `(5cos²i - 1) + 2(-cos i)`.
Actually: ω̇_Ψ = ω̇_ω + ω̇_Ω, so check whether the formula in Eq. 19 correctly
combines these.

### Lines 477–478: `\cite{deprit1969,hori1966}`
> "a near-identity transformation"

**Check**: The near-identity transformation concept is standard in perturbation
theory. Confirm that both Deprit and Hori describe this type of transformation,
or if the document is using the concept more broadly.

### Line 891: `\cite{depritrom1970}`
> "following the general philosophy of complex exponential methods in satellite
> theory"

**Check**: Does Deprit & Rom (1970) actually use complex exponential methods
(F = e^{if} type variables)? Read their paper to confirm this is a correct
attribution. This is a key claim about the provenance of the F-variable approach.

### Lines 1181–1182: `\cite{brouwer1959}`
> "exactly analogous to classical analytical theories such as Brouwer's: the
> derivation of the short-period and secular expressions is a one-time algebraic
> effort"

**Check**: Is this characterization of Brouwer's theory accurate? Was Brouwer's
original work done entirely by hand algebra, with the resulting formulas then
evaluated? This seems like a reasonable claim but verify.

### Line 1251: `\cite{bau2021}`
> "the standard GEqOE-to-Cartesian map, which involves solving the generalized
> Kepler equation for K"

**Check**: Does Baù et al. (2021) present a "generalized Kepler equation"?
What do they call it? Verify the relation L = K + p₁cosK - p₂sinK matches
their paper.

### Lines 1256: `\cite{cefola1972,sanjuan2022}`
> "In the semi-analytical approach (e.g., DSST), the mean-element equations
> of motion are formulated but the short-period corrections are evaluated by
> numerical quadrature at each output step"

**Check**: Is this an accurate characterization of DSST? Does DSST use numerical
quadrature for short-period corrections, or are the short-period corrections
also in closed form in DSST? The key distinction claimed is that DSST uses
quadrature while this work uses closed-form expressions. Verify this is correct
— it's a central positioning claim.

### Lines 1279–1280: `\cite{brouwer1959}`, `\cite{desaedeleer2005,mahajan2019}`
> "The J₂ closure is unusually simple because the averaged disturbing function
> depends only on a, e, and i. Higher zonals are not guaranteed to share that
> property."

**Check**:
1. Is it true that the J₂ averaged disturbing function depends only on a, e, i?
   (Yes, this is standard, but verify against Brouwer.)
2. Do de Saedeleer (2005) and/or Mahajan (2019) discuss higher zonals having
   argument-of-pericenter dependence in the singly averaged problem? This is
   the key claim.

### Line 1436: `\cite{depritrom1970}`
> "in the complex true anomaly variable F, the coefficient functions arise from
> contour integrals of rational functions of F on the unit circle, with interior
> poles only at F = 0 and F = q"

**Check**: Does Deprit & Rom (1970) specifically identify these pole locations
for their complex variable formulation? Or is this the document's own analysis
attributed to their framework?

---

## Part 3: Potentially Unsupported Claims

These claims in the document do NOT have citations and may need one, or may
need verification that they are original contributions:

### Abstract and throughout: "GEqOE"
The abbreviation "GEqOE" is used throughout. Verify whether Baù et al. (2021)
use this abbreviation or if it was coined by the author.

### Line ~39: dM/dF Jacobian formula
> dM/dF = -i(1-q²)³/(1+q²) · F/[(F+q)²(1+qF)²]

This is presented without citation. It should be derivable from the chain rule
through the Kepler equation and the Möbius transform. Verify the formula is
correct by:
1. Checking the derivation chain: M → G → z → F
2. dM/dG = 1 - g cos G = r/a
3. dG/dz = -i/z (since z = e^{iG})
4. dz/dF from the Möbius inverse: z = (F+q)/(1+qF)

### Eq. 13 (line ~270): J₂ potential form
> U_{J₂} = -(A/r³)(1 - 3ẑ²), with A = μJ₂R_e²/2

Verify this is the correct standard form. The zonal J₂ potential is usually
written as U₂ = -(μJ₂R_e²)/(2r³) · (3sin²φ - 1) where φ is geocentric
latitude. Check sign conventions.

### Eq. 27 (line ~452): Mean anomaly secular rate
> Ṁ_bar = ν + (3/4)νJ₂(R_e/p)²β(3c_i² - 1)

This is the Brouwer/Kozai mean motion correction. Verify:
1. The factor β (= √(1-e²)) is correct here. Some formulations use η = √(1-e²)
   and the coefficient is n₀(1 + 3J₂R_e²/(2p²)·√(1-e²)·(1 - 3/2 sin²i)).
2. The sign and coefficient match the standard result.

### Lines 1093–1096: Log-term cancellation argument
> "the poles F = -q and F = -1/q lie strictly inside and outside the unit circle
> respectively, so neither contributes a net winding number, and the residue at
> F = 0 is zero because the average has been subtracted"

This is presented as a proof sketch. Verify the mathematical logic:
1. Is -q inside the unit disk for 0 ≤ q < 1? Yes: |-q| = q < 1.
2. Is -1/q outside? Yes: |-1/q| = 1/q > 1.
3. Does zero average imply zero residue at F = 0? This needs more careful
   argument — the residue at F=0 of the integrand, not of the function itself.

### Structural mean formulas (Eqs. 56–58)
> ⟨1/(F+q)⟩_M = -q(2+q²)/((1-q²)(1+q²))
> ⟨1/(1+qF)⟩_M = (1+2q²)/((1-q²)(1+q²))

These are presented without citation. They should be verifiable by direct
computation: substitute F = (z-q)/(1-qz), multiply by dM/dz, and evaluate
the contour integral by residues at z=0. Consider spot-checking numerically.

---

## Part 4: Numerical Claims Cross-Check

### Validation table (Table 2, lines 1697–1718)
Cross-check a sample of the numerical values against the validation script
output. The relevant scripts are:
- `docs/geqoe_averaged/scripts/run_full_validation.py`
- `docs/geqoe_averaged/scripts/zonal_short_period_validation.py`

Run the validation and compare:
1. Low-e, M0=20°: K_rms = 9.51e-6 rad, pos_rms = 0.155 km, pos_max = 0.297 km
2. High-e, M0=35°: K_rms = 2.41e-4 rad, pos_rms = 2.909 km, pos_max = 4.568 km
3. Any other rows for spot-check

### Round-trip test (lines 1729–1738)
> low-e: 7.2 m, high-e: 1.6 m

Verify these numbers are reproducible from the code.

### Zonal scaling slope (line 1749)
> slope = 0.998

Verify by running the scaling test with λ ∈ {1.00, 0.50, 0.25, 0.125}.

### Eccentricity parameter q values (lines 1669–1671)
> "the eccentricity parameter q ≈ 0.025" for low-e (e=0.05)
> "q ≈ 0.44" for high-e (e=0.65)

Verify: q = g/(1+β) = e/(1+√(1-e²))
- e=0.05: β = √(1-0.0025) ≈ 0.99875, q = 0.05/1.99875 ≈ 0.02502 ✓
- e=0.65: β = √(1-0.4225) ≈ 0.7599, q = 0.65/1.7599 ≈ 0.3693

**ALERT**: q ≈ 0.44 for e=0.65 looks WRONG. Compute carefully:
β = √(0.5775) = 0.75993..., so q = 0.65/(1+0.75993) = 0.65/1.75993 ≈ 0.3693.
The document claims q ≈ 0.44. This may be a hallucination. Double-check.

### 80-minute computation time (lines 1148, 1170, 1895)
> "approximately 80 minutes of CPU time"

This appears 3 times. Verify consistency and whether this is documented
in `short_period_direct_method.md` (which says "~80min" for J5).

### Test orbit parameters
> low-e: a=9000 km, e=0.05, i=40°, Ω₀=25°, Ω₀=60°
> high-e: a=18000 km, e=0.65, i=63°, Ω₀=40°, ω₀=250°

Cross-check against `run_full_validation.py` lines 59, 79–81.

### Mean drift validation reference orbit (lines 1622–1630)
> a=16000 km, e=0.35, i=50°, Ω₀=25°, ω₀=40°, M₀=30°

This is a different orbit from the short-period validation orbits. Verify it
matches what's used in the mean drift validation scripts.

---

## Part 5: Mathematical Consistency Checks

### Eq. 19 (Ψ secular rate)
> ω_{Ψ,J₂} = (3/4)νJ₂(R_e/p)²(5c_i² - 2c_i - 1)

This should equal ω̇_ω + ω̇_Ω where:
- ω̇_ω = (3/4)n J₂(R_e/p)²(5cos²i - 1) [argument of pericenter rate]
- ω̇_Ω = -(3/2)n J₂(R_e/p)²cos i [nodal rate]

Sum: (3/4)n J₂(R_e/p)²[(5cos²i - 1) - 2cos i] = (3/4)n J₂(R_e/p)²(5c_i² - 2c_i - 1) ✓

This checks out algebraically.

### GEqOE definitions consistency
Verify that the GEqOE definitions used in the document match Baù et al. (2021):
1. State vector ordering: (ν, p₁, p₂, K, q₁, q₂)
2. p₁ = g sin Ψ, p₂ = g cos Ψ (not the other way around)
3. q₁ = Q sin Ω, q₂ = Q cos Ω (not the other way around)
4. L = K + p₁ cos K - p₂ sin K

### Eq. 43: ẑ = 2(Yq₂ - Xq₁)/(γr)
Verify this formula against Baù et al. (2021). The normalized z-coordinate
should involve the equinoctial frame vectors.

### Eq. 6: r = a(1 - p₁ sin K - p₂ cos K)
Verify against Baù et al. (2021). Standard equinoctial form would be
r = a(1 - e cos E), and with p₁ = e sin ϖ, p₂ = e cos ϖ, we get
r = a(1 - p₁ sin K - p₂ cos K). Check if this matches.

---

## Part 6: Claims About Novelty and Positioning

### "non-canonical but singularity-free" (line 93)
Verify that GEqOE are indeed:
1. Non-canonical: confirm Baù et al. (2021) states they are not canonical
   (i.e., the equations of motion are not in Hamiltonian form with these
   variables as canonical coordinates)
2. Singularity-free: confirm Baù et al. (2021) claims no singularities
   (no division by e or sin i)

### "replaces the Hamiltonian Lie series machinery with a direct residue decomposition" (lines 95–96)
This is the core novelty claim. Verify that:
1. The standard approach (Brouwer, Deprit, Lara) uses Hamiltonian Lie series
2. This paper genuinely does NOT use Lie series
3. The "direct residue decomposition" is indeed a distinct algorithmic approach

### "no numerical quadrature, no interpolation, no iterative solution (except the standard Kepler equation)" (lines 43–44)
Verify this is accurate for the runtime phase. Specifically:
1. The mean drift for J₃–J₅ requires ODE integration — is this disclosed?
   (Yes, in Section 9.2, but the abstract might be misleading.)
2. The Kepler equation exception is noted.

---

## Execution Instructions

1. Read each reference PDF using the Read tool (they support PDF reading).
2. For each check item, quote the relevant passage from the reference PDF.
3. Mark each item as: ✅ CONFIRMED, ❌ ERROR (with correction), or ⚠️ UNCLEAR
   (needs more investigation).
4. Pay special attention to the **ALERT** items flagged above.
5. For numerical checks, run the validation scripts if possible, or at minimum
   verify the formulas by hand computation.
6. Produce a summary table at the end listing all findings.

Priority order (highest risk of error first):
1. Part 4 ALERT: q ≈ 0.44 for e=0.65 (likely wrong)
2. Part 2: DSST characterization (lines 1256) — central positioning claim
3. Part 2: Brouwer "first-order" claim (line 83) — Brouwer goes beyond first order
4. Part 1: All bibliography metadata
5. Part 2: All citation-context claims
6. Part 3: Unsupported claims
7. Part 5: Mathematical consistency
