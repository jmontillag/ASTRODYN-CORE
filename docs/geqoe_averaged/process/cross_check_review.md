# Cross-Check Review: `geqoe_averaged_j2_zonal_note.tex`

**Date:** 2026-03-11  
**Scope:** `docs/geqoe_averaged/CROSS_CHECK_PROMPT.md` executed against:

- `docs/geqoe_averaged/geqoe_averaged_j2_zonal_note.tex`
- `docs/geqoe_averaged/references/*.pdf`
- `docs/s10569-021-10049-1.pdf` for the final published Baù et al. article
- `src/astrodyn_core/geqoe_taylor/rhs.py`
- `src/astrodyn_core/geqoe_taylor/conversions.py`
- `docs/geqoe_averaged/scripts/run_full_validation.py`
- `docs/geqoe_averaged/scripts/zonal_mean_validation.py`

## Executive Findings

The note is technically strong overall, but it still contains several source-level
errors or overstatements that should be fixed before treating the literature
positioning and bibliography as reliable.

Highest-priority corrections:

1. `sanjuan2022` is wrong in the bibliography. The cited paper is by **Juan F.
   San-Juan, Rosario López, and Paul J. Cefola**, not San-Martín/Pérez, and the
   page range is **1292-1318**, not 1292-1332.
2. `hori1966` is wrong in the bibliography. The title page reads **"Theory of
   General Perturbations with Unspecified Canonical Variables"**. The note uses
   singular forms for both nouns.
3. `brouwer1959` is likely miscited as **378-397**. The local PDF runs from
   **378 to 396**.
4. The note overstates Baù et al. by calling GEqOE "singularity-free". Baù et
   al. state they remain singular for **retrograde equatorial** and
   **rectilinear** cases.
5. The note conflates the original Baù GEqOE with the branch's **K-based**
   propagated state. Baù's published GEqOE use **`L`** as the fourth element,
   not `K`.
6. The DSST discussion is too broad and partly mis-cited. `cefola1972` is an
   equinoctial-elements precursor, not a DSST paper. `sanjuan2022` supports a
   numerical-quadrature-based GTDS-DSST path for the `J2^2` problem, but not the
   blanket claim currently made in the note.
7. The high-eccentricity pole parameter is wrong. For `e = 0.65`, the note says
   `q ≈ 0.44`; the correct value is **`q ≈ 0.3693`**.
8. The statement that the contour-integral coefficient functions have interior
   poles at `F = 0` and `F = q` has a **sign error**. With the note's own
   Möbius map, the interior pole is at **`F = -q`**.
9. The abstract/runtime language overstates the general `J2-J5` model as
   "closed-form mean-element propagation" requiring only algebraic evaluation.
   Later sections correctly state that the mixed-zonal mean system still
   requires **numerical integration of a slow ODE**.
10. The operational statement that GEqOE-to-Cartesian conversion "involves
    solving the generalized Kepler equation for `K`" is true for Baù's
    **`L`-based** GEqOE, but not for the branch's current **`K`-based** state
    and code path.

## Part 1: Bibliography Metadata

| Bibkey | Status | Review |
|---|---|---|
| `bau2021` | ✅ | Authors, title, journal, year, volume, and article number are correct in the note. The copy in `references/bau2021.pdf` is an arXiv preprint; the final published version is `docs/s10569-021-10049-1.pdf`, which shows `Celestial Mechanics and Dynamical Astronomy (2021) 133:50`. |
| `brouwer1959` | ❌ | Author, title, journal, year, and volume are correct. The local PDF starts at page 378 and ends at page 396. The note's `378-397` is very likely wrong. |
| `kozai1959` | ✅ | Metadata matches the local PDF title page and page range `367-377`. |
| `deprit1969` | ✅ | Metadata matches the title page: `Celestial Mechanics 1 (1969) 12-30`. |
| `hori1966` | ❌ | The title page reads **"Theory of General Perturbations with Unspecified Canonical Variables"** by **Gen-ichiro Hori**. The note currently uses singular forms: "perturbation" and "variable". Volume/pages `18, 287-296` are consistent with the scan length. |
| `broucke1972` | ✅ | Metadata matches the first-page footer: `Celestial Mechanics 5 (1972) 303-310`. |
| `lyddane1963` | ✅ | Metadata matches the local PDF: `The Astronomical Journal 68, 555-558`. |
| `depritrom1970` | ✅ | Metadata matches the title page/footer: `Celestial Mechanics 2 (1970) 166-206`. |
| `coffey1982` | ✅ | Title, authors, volume, issue, and pages `5(4), 366-371` match the local PDF. The exact journal-title styling varies across secondary sources, but the note's modernized `Journal of Guidance, Control and Dynamics` form is acceptable. |
| `desaedeleer2005` | ✅ | Metadata matches the title page: `Celestial Mechanics and Dynamical Astronomy 91, 239-268`. |
| `lara2021` | ✅ | Metadata matches the published article: `Celestial Mechanics and Dynamical Astronomy 133:47`. |
| `cefola1972` | ✅ | Title and paper number `AIAA Paper 72-937` match the title page. |
| `sanjuan2022` | ❌ | The note's entry is wrong. The local PDF shows **Juan F. San-Juan, Rosario López, Paul J. Cefola**, title **"A Second-Order Closed-Form J2 Model for the Draper Semi-Analytical Satellite Theory"**, journal **The Journal of the Astronautical Sciences**, volume **69**, pages **1292-1318**. |
| `mahajan2019` | ✅ | Metadata matches the published article: `Celestial Mechanics and Dynamical Astronomy (2019) 131:45`. |

## Part 2: Citation-Context Claims

| Location | Status | Review |
|---|---|---|
| Line 57, `bau2021` | ⚠️ | Baù et al. do define and use **GEqOE**. However, the published set is `(ν, p1, p2, L, q1, q2)` or the `L0` variant, not `(ν, p1, p2, K, q1, q2)`. The note should say explicitly that it adopts the Baù geometry but propagates `K` instead of `L`, following the branch implementation. |
| Line 75, `deprit1969,hori1966` | ⚠️ | The pairing is conceptually reasonable, but "Lie-Deprit-Hori framework" is modern shorthand rather than a label used verbatim by those papers. Deprit formalizes Lie-series canonical transformations depending on a small parameter; Hori presents general perturbation theory via Lie canonical transformations with unspecified canonical variables. This should be phrased as a **related Lie-transform tradition**, not as a single named framework directly attested in the sources. |
| Lines 83-84, `brouwer1959,kozai1959` | ❌ | Both papers use Delaunay/canonical variables, and both derive first-order periodic terms. But the note understates Brouwer and Kozai by saying only "first-order secular and short-period corrections". Their abstracts explicitly state **first-order periodic terms and secular terms up to second order**. |
| Lines 85-86, `coffey1982` | ✅ | Confirmed. The title and abstract explicitly present a third-order analytic solution to the main problem. The abstract further says the transformed Hamiltonian is developed to order 4 in the small parameter and agrees through order 3 with Brouwer/Kozai. |
| Lines 86-88, `desaedeleer2005` | ✅ | Confirmed. The abstract explicitly says the paper treats **any zonal harmonic `Jn (n >= 2)`** and provides a first-order averaged Hamiltonian and generator in closed form. |
| Lines 88-89, `lara2021` | ✅ | Confirmed. The abstract says Brouwer's solution is revisited to show the complete Hamiltonian reduction is achieved in **plain Poincaré's style, through a single canonical transformation**. |
| Lines 91-92, `cefola1972,sanjuan2022` | ❌ | `sanjuan2022` is genuinely DSST-related. `cefola1972` is **not** a DSST paper; it is an equinoctial-elements / single-averaged variation-of-parameters paper. The sentence currently collapses distinct historical threads. |
| Line 92, `broucke1972` | ⚠️ | Broucke and Cefola investigate equinoctial orbit elements, show they are free from the zero-`e` and zero-`i` singularities, and derive partial derivatives / brackets. But they do **not** "develop the semi-analytical approach" in the later DSST sense, and they do not claim to have introduced the elements from scratch. Rephrase more narrowly. |
| Line 145, `bau2021` | ⚠️ | The formula `c = (μ²/ν)^{1/3} β` is correct and Baù et al. do use `c` throughout, but they call it the **generalized angular momentum**, not an "angular momentum proxy". |
| Lines 313-315, `brouwer1959,kozai1959` | ✅ | Confirmed. The nodal rate is the classical `Ω̇`, and the note's `Ψ̇ = ω̇ + Ω̇` gives `(3/4) n J2 (Re/p)^2 (5 cos^2 i - 2 cos i - 1)`, which is algebraically correct. |
| Lines 477-478, `deprit1969,hori1966` | ✅ | The near-identity transformation language is well aligned with both sources. Deprit's paper explicitly develops canonical transformations in formal powers of a small parameter, and Hori's paper treats general perturbations through Lie transformations of canonical variables. |
| Line 891, `depritrom1970` | ❌ | Not substantiated. The accessible title page/abstract describe Lie transforms, successive canonical transformations, and Delaunay-type construction. They do **not** provide evidence for the note's specific attribution to "complex exponential methods" or the `F = e^{if}` philosophy. This looks like the note's own analytical framing, not Deprit and Rom's explicit claim. |
| Lines 1181-1182, `brouwer1959` | ✅ | As a historical characterization, this is fair. Brouwer's paper is a large one-time symbolic derivation yielding closed formulas for computation. This is an inference from the structure of the paper, not a direct textual claim by Brouwer. |
| Line 1251, `bau2021` | ❌ | For Baù's published **`L`-based** GEqOE, conversion to Cartesian does require solving the generalized Kepler equation for `K`. But in this branch and in the note's own state definition, the propagated state already contains `K`. The current code in `src/astrodyn_core/geqoe_taylor/conversions.py` converts from `K` directly and does **not** solve Kepler here. The sentence is therefore wrong in the note's operational context. |
| Lines 1256+, `cefola1972,sanjuan2022` | ❌ | The broad DSST statement is too strong. `sanjuan2022` clearly discusses a numerical-based GTDS-DSST approach using Gauss-Kronrod quadrature for the relevant averaging/generator work that motivated their closed-form replacement. But `cefola1972` is not a DSST source, and the two citations together do not justify the universal claim that DSST short-period corrections are evaluated by numerical quadrature at each output step. |
| Lines 1279-1280, `brouwer1959,desaedeleer2005,mahajan2019` | ✅ | Confirmed. `J2` is special because its singly averaged disturbing function depends only on `a, e, i`. Mahajan explicitly states that when higher harmonics are treated as first-order perturbations, long-period terms appear in the first-order single-averaged Hamiltonian. |
| Line 1436, `depritrom1970` | ❌ | This line has both a **citation problem** and a **sign error**. With the note's own `z = (F+q)/(1+qF)` and `dM/dF`, the interior pole is at **`F = -q`**, not `F = q`. Also, the specific pole statement is not demonstrated as something Deprit and Rom themselves say. |

## Part 3: Unsupported / Overstated Claims

| Claim | Status | Review |
|---|---|---|
| "GEqOE" abbreviation used throughout | ✅ | Baù et al. explicitly use `GEqOE`. |
| `dM/dF = -i(1-q^2)^3/(1+q^2) * F/[(F+q)^2 (1+qF)^2]` | ✅ | Confirmed from the chain rule using the note's own definitions: `dM/dG = 1 - g cos G`, `dz/dG = i z`, `z = (F+q)/(1+qF)`, and `g = 2q/(1+q^2)`. |
| `U_J2 = -(A/r^3)(1 - 3 zhat^2)` with `A = μ J2 Re^2 / 2` | ✅ | Correct under Baù's sign convention, where `U` is the opposite of the disturbing potential. |
| `\dot{\bar M} = ν + (3/4) ν J2 (Re/p)^2 β (3 c_i^2 - 1)` | ✅ | Consistent with the standard Brouwer/Kozai first-order mean-anomaly rate written with `β = sqrt(1-e^2)` and `p = a β^2`. |
| Log-term cancellation proof sketch | ⚠️ | The geometric facts about `-q` being inside and `-1/q` being outside the unit circle are correct. But the note's sentence "the residue at `F=0` is zero because the average has been subtracted" is not a complete proof. Zero average constrains the **total contour integral**, not the residue at one interior pole individually. The conclusion may still be true for the specific integrand class, but the argument needs tightening. |
| Structural means `⟨1/(F+q)⟩_M` and `⟨1/(1+qF)⟩_M` | ✅ | Confirmed by direct residue computation after substituting `F = (z-q)/(1-qz)` and integrating over the unit circle in `z`. |

## Part 4: Numerical Cross-Checks

Commands run:

```bash
conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/run_full_validation.py
conda run -n astrodyn-core-env python docs/geqoe_averaged/scripts/zonal_mean_validation.py
```

Results:

| Check | Status | Review |
|---|---|---|
| Table 2, low-e, `M0 = 20°` | ✅ | Reproduced: `K_rms = 9.506e-06`, `pos_rms = 0.1553 km`, `pos_max = 0.2965 km`. |
| Table 2, high-e, `M0 = 35°` | ✅ | Reproduced: `K_rms = 2.414e-04`, `pos_rms = 2.909 km`, `pos_max = 4.568 km`. |
| Round-trip test | ✅ | Reproduced: `low-e ≈ 7.2 m`, `high-e ≈ 1.6 m`. |
| Zonal scaling slope | ✅ | Reproduced: `0.998`. |
| High-e `q ≈ 0.44` | ❌ | Wrong. For `e = 0.65`, `β = sqrt(1-e^2) = 0.759934...`, so `q = e/(1+β) = 0.369332...`. |
| 80-minute one-time generation claim | ✅ | Consistent with `short_period_direct_method.md`, which reports `~80 min` for `J5`. |
| Short-period validation orbit parameters | ✅ | Match `run_full_validation.py`: low-e `(9000 km, 0.05, 40°, 25°, 60°)`, high-e `(18000 km, 0.65, 63°, 40°, 250°)`. |
| Mean-drift validation reference orbit | ✅ | Matches `zonal_mean_validation.py`: `(16000 km, 0.35, 50°, 25°, 40°, 30°)`. |

## Part 5: Mathematical Consistency

| Item | Status | Review |
|---|---|---|
| `Ψ̇_J2 = ω̇ + Ω̇` formula | ✅ | Algebraically correct. |
| `p1 = g sin Ψ`, `p2 = g cos Ψ` | ✅ | Matches Baù et al. |
| `q1 = Q sin Ω`, `q2 = Q cos Ω` | ✅ | Matches Baù et al. |
| `L = K + p1 cos K - p2 sin K` | ✅ | Matches Baù et al. Eq. (25). |
| `zhat = 2(Y q2 - X q1)/(γ r)` | ✅ | Consistent with Baù's equinoctial frame vectors `eX`, `eY`. |
| `r = a(1 - p1 sin K - p2 cos K)` | ✅ | Matches Baù et al. Eq. (26). |
| Note's state ordering vs Baù | ⚠️ | The note's ordering `(ν, p1, p2, K, q1, q2)` matches the branch implementation, not the published GEqOE definition in Baù, which uses `L` as the fourth element. This is an adaptation and should be stated explicitly. |

## Part 6: Novelty / Positioning Claims

| Claim | Status | Review |
|---|---|---|
| "non-canonical but singularity-free" | ❌ | "Non-canonical" is plausible as an interpretive description, but Baù et al. do not present the set as canonical. "Singularity-free" is wrong as written: the paper says the elements are non-singular for circular and equatorial trajectories but still singular for retrograde equatorial and rectilinear cases. |
| "replaces the Hamiltonian Lie series machinery with a direct residue decomposition" | ✅ | As a description of the note's method, this is fair. The classical references use Lie/canonical-transform machinery; the present note derives the short-period kernels by direct rational/residue manipulation instead. |
| "no numerical quadrature, no interpolation, no iterative solution (except the standard Kepler equation)" | ❌ | Overstated for the general mixed-zonal case. The short-period kernels are precomputed algebraically, but the note later states that the general `J2-J5` mean dynamics require numerical integration of a slow ODE. The abstract should distinguish **closed-form RHS / kernel evaluation** from **closed-form propagation**. |

## Recommended Text Fixes

1. Fix the bibliography entries for `sanjuan2022`, `hori1966`, and likely
   `brouwer1959`.
2. Clarify at first mention that the note uses a **K-based branch adaptation**
   of Baù's GEqOE, whose published fourth element is `L`.
3. Replace "singularity-free" with a narrower source-faithful phrase such as
   "free of the circular and equatorial singularities, but still singular for
   retrograde equatorial and rectilinear motion".
4. Rephrase the Brouwer/Kozai sentence to acknowledge **first-order periodic**
   and **second-order secular** results.
5. Split the DSST discussion into two claims:
   - `cefola1972` for equinoctial-element / single-averaged precursor context.
   - later DSST sources for the actual semi-analytical theory and its numerical
     quadrature aspects.
6. Fix the high-eccentricity `q` value to `0.3693`.
7. Fix the sign in the contour-pole sentence: `F = -q`, not `F = q`.
8. Tighten the logarithm-cancellation explanation or present it as an
   implementation observation backed by the direct decomposition, not as a
   standalone proof.
9. Replace the abstract phrase "closed-form mean-element propagation" with
   something like "closed-form mean-element equations and short-period
   corrections", unless the claim is being restricted to pure `J2`.
10. Fix the operational statement about the GEqOE-to-Cartesian map so it does
    not claim a Kepler solve when the propagated state already contains `K`.

## Bottom Line

The literature grounding is mostly sound, but the note is not yet safe to treat
as publication-grade in its current citation/positioning form. The largest live
problems are:

- one definitely wrong bibliography entry (`sanjuan2022`);
- one definitely wrong title (`hori1966`);
- one likely wrong page range (`brouwer1959`);
- one clearly wrong numerical value (`q ≈ 0.44`);
- one clear sign error (`F = q` instead of `F = -q`);
- and two central positioning overstatements (DSST characterization, and
  "closed-form propagation" / "no iterative solution" for the mixed-zonal case).
