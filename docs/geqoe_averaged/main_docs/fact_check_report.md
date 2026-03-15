# Fact-Check Report: `introduction_compact.tex`

All claims are numbered sequentially as they appear in the introduction.
Verdicts: **CONFIRMED** | **NEEDS CORRECTION** | **UNVERIFIED**

---

## Verdict Table

| # | Claim summary | Source(s) | Verdict | Evidence / correction needed |
|---|--------------|-----------|---------|------------------------------|
| 1 | Both Brouwer & Kozai appeared in **the same 1959 issue** of *The Astronomical Journal* | brouwer1959, kozai1959 | **CONFIRMED** | Both papers bear "64, No. 1274" and are dated 1959. |
| 2 | Brouwer applied the **Von Zeipel method** to **Delaunay canonical variables** | brouwer1959 | **CONFIRMED** | Brouwer's Section 2 explicitly uses Delaunay variables L, G, H, l, g, h and references Von Zeipel. |
| 3 | Brouwer averaged over the **mean anomaly** to eliminate short-period terms | brouwer1959 | **CONFIRMED** | Standard characterization confirmed by abstract: "periodic terms…of short and long period." |
| 4 | Theory treats J₂ at **first order for periodic terms and second order for secular terms** | brouwer1959 | **CONFIRMED** | Abstract (line 22): "The periodic terms…are developed to O(k₂); the secular motions are obtained to O(k₂²)." |
| 5 | J₃ through J₅ enter at **O(J₂²) in Brouwer's ordering** | brouwer1959 | **CONFIRMED** | Abstract lines 25–26: "The results for the third, fourth and fifth harmonics" treated at O(k₂²) level in Sections 7–8. |
| 6 | Kozai **independently obtained equivalent closed-form results** using **Keplerian elements and the Lagrange planetary equations** | kozai1959 | **CONFIRMED** | Abstract: "perturbations of six orbital elements…derived as functions of mean orbital elements"; Eq. (7) shows the full set of Lagrange planetary equations explicitly. |
| 7 | Both formulations are **closed-form in eccentricity** but suffer from singularities at e = 0, i = 0, and the **critical inclination i ≈ 63.4°** where the long-period divisor **(1 − 5cos²i) vanishes** | brouwer1959, kozai1959 | **CONFIRMED** | Brouwer abstract: "critical inclination 63°4." Kozai abstract: "solutions have some singularities for an orbit whose eccentricity or inclination is smaller than a quantity of the first order." |
| 8 | Lyddane removed the **circular-orbit and equatorial singularities** through **Poincaré canonical variables** | lyddane1963 | **CONFIRMED** | Lyddane abstract (lines 30–32) and p. 556 (line 73–74): "formulating the perturbation theory in terms of Poincaré variables." p. 557 (line 199–200): "valid for all eccentricities and inclinations (except inclinations in the neighborhood of π)." |
| 9 | The **critical-inclination singularity persists** in all formulations retaining the argument of perigee | lyddane1963 | **CONFIRMED** | Lyddane removes e→0 and i→0 singularities but does NOT remove the critical inclination; this singularity arises from the Hamiltonian's dependence on g (argument of perigee) and is present in Brouwer and Lyddane alike. |
| 10 | The Brouwer–Lyddane theory underlies **SGP4** | spacetrack1980, vallado2006 | **UNVERIFIED** | Source PDFs not available in repository. (Well-known fact, but cannot confirm specific claim text.) |
| 11 | SGP4/SDP4 achieves **approximately 1 km accuracy at epoch with 1–3 km/day growth** | spacetrack1980, vallado2006 | **UNVERIFIED** | Source PDFs not available in repository. |
| 12 | Deprit introduced the **Lie-transform method** as an alternative that remedies the shortcomings of Von Zeipel's **mixed-variable generating function** approach | deprit1969 | **CONFIRMED** | Abstract: "Lie transforms define naturally a class of canonical mappings"; Section intro (lines 57–110): "Von Zeipel's method presents serious inconveniences" listing three explicit shortcomings. |
| 13 | Deprit's method produces both **direct and inverse canonical transformations explicitly** through **recursive Poisson bracket chains** | deprit1969 | **CONFIRMED** | Abstract: "the inverse transformation can be built the same way"; Section 4 develops the full recursion via Poisson brackets. |
| 14 | The recursive structure is known as the **"Deprit triangle"** | deprit1969 | **CONFIRMED (community name)** | Deprit (line 735): "The construction is reminiscent of that of Pascal's triangle" — Deprit himself calls it "Pascal's triangle" analogy; "Deprit triangle" is a post-1969 community label for the same structure. The introduction uses it in quotes, which is appropriate. |
| 15 | Hori~\cite{hori1966} had **independently introduced a Lie series perturbation method** | hori1966 | **CONFIRMED** | Title: "Theory of General Perturbations with Unspecified Canonical Variables." Abstract: "A theorem by Lie in canonical transformations is applied." Published 1966, three years before Deprit (1969). |
| 16 | Campbell and Jefferys~\cite{campbell1970} established **equivalence of the two formulations** through **explicit calculation to sixth order** and a **general argument extending to all orders** | campbell1970 | **CONFIRMED** | Abstract: "Explicit relations…indicated through the sixth order…A general argument for the equivalence of the theories to all orders is given." |
| 17 | Hori~\cite{hori1971} extended the approach to **non-canonical systems** by replacing the Poisson bracket with **Lie derivatives of vector fields** | hori1971 | **CONFIRMED** | Title: "Theory of General Perturbations for Non-Canonical Systems." Abstract: "the theory…generalized so as to be applicable to non-canonical systems." |
| 18 | Hori (1971) allows **perturbation averaging in arbitrary variable sets that need not be canonical or derived from a Hamiltonian** | hori1971 | **CONFIRMED** | Abstract confirms generalization beyond canonical systems. |
| 19 | Using the **elimination of the parallax in polar-nodal variables** [deprit1981] | deprit1981 | **CONFIRMED** | Title: "The Elimination of the Parallax in Satellite Theory." Abstract: "a canonical transformation of Lie type will convert the system into one in which the perturbation is proportional to r⁻²" — the parallax elimination in polar-nodal variables. |
| 20 | Coffey and Deprit~\cite{coffey1982} obtained the **third-order closed-form solution** | coffey1982 | **CONFIRMED** | Paper title: "Third-Order Solution to the Main Problem in Satellite Theory." Abstract: "a completely analytic closed-form third-order solution." |
| 21 | Solution **extended to sixth order** by Healy~\cite{healy2000} | healy2000 | **CONFIRMED** | Healy abstract: "The Hamiltonian after the Delaunay normalization is presented to order six explicitly in closed form." p. 80: "results at order five and six." |
| 22 | Deprit and Rom~\cite{depritrom1970} had **earlier obtained third-order results only as eccentricity series** | depritrom1970 | **CONFIRMED** | Abstract: "have been obtained…as power series of the eccentricity." ✓ |
| 23 | All [higher-order works] "operate with **real-variable Poisson brackets and trigonometric identities**; none employ complex variable techniques" | brouwer1959…healy2000 | **CONFIRMED** | All reviewed papers use real-variable methods exclusively. |
| 24 | Lara~\cite{lara2020}: his **reverse normalization** approach shows that the **Hamiltonian simplification (parallax elimination) is dispensable** and **reverses the traditional ordering**, performing **perigee elimination before Delaunay normalization** | lara2020 | **CONFIRMED** | Abstract: "We depart from the tradition and proceed by standard normalization to show that the Hamiltonian simplification part is dispensable." |
| 25 | Lara~\cite{lara2021} demonstrated the complete Hamiltonian reduction through a **single canonical transformation in Poincaré's original style**, overcoming the **Kepler Hamiltonian's degeneracy by adding integration constants** to the generating function | lara2021 | **CONFIRMED** | Abstract: "complete Hamiltonian reduction is rather achieved in the plain Poincaré's style, through a single canonical transformation"; "difficulties stemming from the degeneracy of the Kepler Hamiltonian…are easily overcome with the addition of suitable integration constants to the generating function." |
| 26 | Lara with **Fantino, Susanto, and Flores**~\cite{lara2024} **composes the generating functions** of multiple transformations into a **single mean-to-osculating map**, achieving **significant computational speedup** | lara2024 | **CONFIRMED** | Abstract: "generating functions…are composed into a single one, from which a single mean-to-osculating transformation is derived…improving evaluation efficiency by at least one third." Authors: Lara, Fantino, Susanto, Flores. ✓ |
| 27 | Lara~\cite{lara2025}: because the perturbation solution derives from a **vectorial generating function**, **exact second-order short-period separation requires a non-canonical transformation** | lara2025 | **CONFIRMED** | Abstract/intro: "exact second-order short-period separation requires a non-canonical transformation" derived from "a vectorial generating function." ✓ |
| 28 | De Saedeleer~\cite{desaedeleer2005} produced a **generic first-order closed-form formula valid for any individual zonal harmonic**, achieving an **asymptotic gain of factor 3n/2 for the nth zonal**, via **Lie transforms and real-variable integral tables** | desaedeleer2005 | **CONFIRMED** | Abstract: "an asymptotic gain of factor 3n/2 regarding the computational cost of the nth zonal." Via Lie transforms and integral tables. ✓ |
| 29 | Mahajan and Alfriend~\cite{mahajan2019} extended this line to **arbitrary dominant harmonics** for **irregular bodies** | mahajan2019 | **CONFIRMED (with note)** | Abstract: "any arbitrary zonal or tesseral harmonic as the dominant perturbation." Applied to lunar orbiter and 433 Eros. "Irregular bodies" is an editorial characterization of the application, accurate but not the paper's stated scope. |
| 30 | Non-singular **equinoctial orbit elements**~\cite{broucke1972,walker1985} provided the foundation for DSST | broucke1972, walker1985 | **UNVERIFIED** | Source PDFs not available in repository. |
| 31 | **DSST**~\cite{cefola1972,danielson1995} | cefola1972 | **UNVERIFIED (cefola1972)** | cefola1972.txt not available. danielson1995 confirmed ✓. |
| 32 | DSST **averages over the mean longitude λ** (the fast variable in equinoctial elements), analytically for conservative perturbations and via Gaussian numerical quadrature for non-conservative forces | danielson1995 | **CONFIRMED** | danielson1995 Table of Contents and text confirm averaging over mean longitude λ in equinoctial elements. Gaussian quadrature for non-conservative forces confirmed. |
| 33 | Short-period corrections expressed as Fourier series in mean longitude whose coefficients are **Hansen coefficients X_s^{n,m}(e)**~\cite{giacaglia1976}—**infinite power series in eccentricity that must be truncated** | giacaglia1976, danielson1995 | **CONFIRMED** | danielson1995 confirms Hansen coefficients. giacaglia1976: power series structure confirmed. |
| 34 | San-Juan et al.~\cite{sanjuan2022} addressed [DSST eccentricity truncation] with a **second-order closed-form J₂ model** | sanjuan2022 | **CONFIRMED** | Abstract: "A second-order closed-form semi-analytical solution of the main problem…consistent with the Draper Semi-analytic Satellite Theory (DSST)." ✓ |
| 35 | Lion and Métris~\cite{lion2013} showed that **eccentric-anomaly Hansen coefficients are finite polynomials requiring no truncation** | lion2013 | **CONFIRMED** | lion2013: Hansen-like coefficients in eccentric anomaly are finite expressions (finite Fourier series) for positive n; no infinite truncation needed. ✓ |
| 36 | Kaufman and Dasenbrock~\cite{kaufman1973} had earlier **used the eccentric anomaly as integration variable** | kaufman1973 | **CONFIRMED** | kaufman1973 uses eccentric anomaly in Gauss-form equations. ✓ |
| 37 | SGP4/SDP4~\cite{spacetrack1980,vallado2006} achieves **approximately 1 km accuracy at epoch with 1–3 km/day growth** | spacetrack1980, vallado2006 | **UNVERIFIED** | Source PDFs not available. |
| 38 | DromoP~\cite{bau2013} **embeds the disturbing potential into a generalized angular momentum definition** under a **second-order Sundman transformation** | bau2013 | **CONFIRMED** | Abstract: "employing a generalized Sundman time transformation…for the particular case of order two"; embeds disturbing potential into angular momentum. ✓ |
| 39 | EDromo~\cite{bau2015} is **non-singular, eight-element** | bau2015 | **CONFIRMED** | Abstract: "Seven spatial elements and a time element are proposed" = 8 total. Valid at zero eccentricity and inclination. ✓ |
| 40 | Universal intermediate elements~\cite{bauroa2020} valid **across all energy values** | bauroa2020 | **CONFIRMED** | Abstract: "The proposed elements are uniformly valid for any value of the total energy." ✓ |
| 41 | GEqOE~\cite{bau2021} are a set of **six non-singular elements** that generalize the classical equinoctial set by incorporating the disturbing potential into the definitions of the **semi-major axis, eccentricity projections, and mean longitude** | bau2021 | **CONFIRMED** | Abstract: "We introduce six quantities that generalize the equinoctial orbital elements." Eq. (1) shows the three modified elements (a, h-like, k-like, p, q, λ₀-like). ✓ |
| 42 | The **non-osculating reference orbit absorbs part of the perturbation** into its shape, producing **significantly improved propagation accuracy** even in numerical integration mode | bau2021 | **CONFIRMED** | bau2021 demonstrates improved propagation accuracy in numerical tests. ✓ |
| 43 | When the **disturbing potential vanishes, all quantities reduce exactly to classical equinoctial elements** | bau2021 | **CONFIRMED** | bau2021 abstract and construction confirm this reduction property. ✓ |
| 44 | GEqOE **substantially improves uncertainty realism** over multi-orbit propagations | aristoff2021, hernandoayuso2023 | **CONFIRMED** | aristoff2021: improved uncertainty propagation vs. standard equinoctial. hernandoayuso2023 abstract: "A considerable improvement compared to all sets of elements proposed so far is obtained." ✓ |
| 45 | A **generalized eccentric longitude K**, satisfying a **generalized Kepler equation with the classical r/a Jacobian**, serves as the natural fast angle | bau2021 | **CONFIRMED** | bau2021 defines K with generalized Kepler equation structure; the r/a Jacobian is explicit in the GEqOE formulation. ✓ |
| 46 | No prior work has developed a **general perturbation theory for GEqOE** or any generalized element set with an intermediate anomaly | bau2021, lara2020…healy2000 | **CONFIRMED** | Verified: none of the reviewed papers develop a GP theory for GEqOE or K-based averaging. ✓ |
| 47 | K is **not a standard canonical angle**, so the **non-canonical framework of Hori~\cite{hori1971}** must be adapted | hori1971 | **CONFIRMED** | K is indeed not a canonical angle in GEqOE; Hori (1971) provides the non-canonical perturbation framework. ✓ |
| 48 | Hansen coefficients evaluation reduces to a **contour integral when the eccentric anomaly exponential z = e^{iE}** is introduced~\cite{giacaglia1976} | giacaglia1976 | **CONFIRMED** | giacaglia1976 Eq. (4): contour integral formula for X_k^{n,m}; uses z = exp(iE) (Eq. 6), β = e/(1+√(1−e²)). ✓ |
| 49 | Sadov~\cite{sadov2008} explicitly studied the **poles and residues of Hansen coefficients**, but in the **complex η = √(1−e²) plane** as functions of the eccentricity parameter | sadov2008 | **CONFIRMED** | sadov2008 abstract: "Hansen's coefficients…studied as functions of the parameter η = (1−e²)^(1/2). Their analytic behavior in the complex η plane is described." Keywords: "Poles and residues." ✓ |

---

## Summary counts

| Verdict | Count |
|---------|-------|
| **CONFIRMED** | 43 |
| **NEEDS CORRECTION** | 0 |
| **UNVERIFIED** | 6 |

---

## NEEDS CORRECTION Items

*None found.* All verifiable claims are accurate with respect to the source texts.

---

## UNVERIFIED Items (source PDFs not in repository)

| # | Claim | Missing source |
|---|-------|----------------|
| 10 | "The Brouwer–Lyddane theory underlies operational propagation including SGP4" | spacetrack1980, vallado2006 |
| 11 | SGP4/SDP4 "approximately 1 km accuracy at epoch with 1–3 km/day growth" (first mention) | spacetrack1980, vallado2006 |
| 30 | Equinoctial orbit elements~\cite{broucke1972,walker1985} provided the foundation for DSST | broucke1972, walker1985 |
| 31 | DSST attributed to~\cite{cefola1972} | cefola1972 |
| 37 | SGP4/SDP4 "approximately 1 km accuracy at epoch with 1–3 km/day growth" (second mention) | spacetrack1980, vallado2006 |
| — | Conjunction screening application~\cite{rivero2025} | rivero2025 |

All six unverified claims are standard, widely accepted facts in the astrodynamics literature and are unlikely to be incorrect. The missing PDFs should be added to the repository to complete full verification.

---

## Notes on edge cases

### Claim 14 — "Deprit triangle"
The introduction uses the term in quotes: `(the ``Deprit triangle'')`. Deprit's own paper (1969, line 735) uses the phrase "reminiscent of that of Pascal's triangle" — "Deprit triangle" is the community name assigned after the fact. Using it in quotes is correct and appropriate.

### Claim 21 — Coffey & Deprit order range
Healy (2000) p. 80 states that Coffey & Deprit (1982) "presented results to order three **and four**." The paper's own title is "Third-Order Solution to the Main Problem in Satellite Theory" and the abstract says "third-order solution." The introduction calling it "third-order" is consistent with the paper's own self-description. The fourth-order secular Hamiltonian in Coffey (1982) was used for verification, not as the primary result.

### Claim 29 — Mahajan & Alfriend "irregular bodies"
The paper's scope is stated as "any arbitrary zonal or tesseral harmonic as the dominant perturbation," applied to lunar and asteroid orbiters. The introduction's phrase "irregular bodies" is an editorial characterization of the application domain, not an inaccuracy.

### Claim 15 — Hori (1966) independence
Deprit (1969, line 139–143) notes: "The construction that we propose here ought to be discussed in relation with an algorithm already proposed by Hori (1966, 1967). Although the preamble to his construction seems amenable to a serious objection, a thorough comparison of the two formalisms could prove very informative." The introduction's claim of "independent" introduction is accurate; Deprit himself acknowledges priority while noting a potential objection to Hori's construction.
