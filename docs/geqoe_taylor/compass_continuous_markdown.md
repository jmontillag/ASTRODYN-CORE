# Continuous-thrust optimization meets generalized orbital elements: a unified framework

**A unified mathematical framework combining Generalized Equinoctial Orbital Elements (GEqOE), Fourier thrust parameterization, Taylor integration with automatic differentiation, and the control-distance metric is technically feasible—but no published work yet combines all these pieces.** Each component is mature independently: GEqOE provide near-linear propagation by absorbing conservative perturbations into non-osculating element definitions; Thrust Fourier Coefficients (TFCs) reduce continuous control to 14 (or as few as 6) parameters via orbit-averaging; heyoka delivers machine-precision Taylor integration with arbitrary-order variational equations; and the control-distance metric J enables statistical maneuver detection through energy-optimal cost evaluation. The critical gap—Fourier thrust parameterization in the generalized eccentric longitude K rather than classical eccentric anomaly E—remains unexplored and represents a natural, high-value research target. This report maps the mathematical foundations, recent advances (2020–2025), and concrete opportunities for constructing such a framework.

---

## GEqOE embed conservative gravity into the element definitions

Baù, Hernando-Ayuso, and Bombardelli (2021) introduced GEqOE by redefining classical equinoctial orbital elements using **total energy** E = v²/2 − μ/r + U rather than Keplerian energy alone. The perturbing force decomposes as **F = P − ∇U**, where U(r,t) is any conservative potential (geopotential harmonics, third-body gravity) and **P** captures non-conservative forces like thrust and drag. Six elements {ν, p₁, p₂, L̃, q₁, q₂} describe the orbit: ν is the generalized mean motion (−2E)^(3/2)/μ, p₁ and p₂ project the generalized eccentricity vector onto the equinoctial frame, L̃ is the generalized mean longitude, and q₁, q₂ are the standard equinoctial inclination variables.

The key innovation is the **generalized eccentric longitude K = G + Ψ**, where G is the generalized eccentric anomaly satisfying g cos G = 1 − r/ã (with ã the generalized semi-major axis) and Ψ is the generalized longitude of periapsis. When U = 0, K reduces to the classical eccentric longitude ε = E + ω̃. Because conservative perturbations are absorbed into the element definitions, **five of six GEqOE remain nearly constant under gravity-only dynamics**, and the fast variable L̃ advances at rate ν with only small corrections. This contrasts sharply with osculating elements, where J₂ causes large periodic oscillations in all elements.

Thrust enters the GEqOE equations exclusively through the non-conservative perturbation **P**. The radial and transverse components P_r and P_f drive the energy rate Ė = ∂U/∂t + u·P_r + (h/r)·P_f, which propagates into the time derivatives of ν, p₁, p₂, and L̃. The out-of-plane component enters q̇₁ and q̇₂ through F_h. Hernando-Ayuso et al. (2023, JGCD) demonstrated that GEqOE with tangential low-thrust maintain covariance realism for **36% longer** than competing element sets—a direct consequence of the smoother, more linear dynamics. Bombardelli's group at UPM further showed that GEqOE outperform both classical and J₂-adapted equinoctial elements for LEO orbit-raising scenarios representative of Starlink-class satellites.

The non-osculating nature of GEqOE connects to the **gauge freedom** framework of Efroimsky & Goldreich (2003, 2004). The Lagrange constraint (osculation) is merely one choice among infinitely many for parameterizing the variation of parameters. GEqOE deliberately choose a gauge that absorbs conservative dynamics, making the reference conic non-tangent to the trajectory but producing dramatically smoother element variation. This gauge choice has profound implications for thrust parameterization: since the dominant gravitational periodicities are already embedded in the element definitions, residual thrust-driven variation is inherently lower-frequency and requires fewer Fourier terms to represent.

---

## Thrust Fourier Coefficients reduce infinite-dimensional control to 14 parameters

Hudson and Scheeres (2009) achieved a remarkable dimensional reduction by expanding thrust acceleration components (radial F_R, transverse F_S, normal F_W) as Fourier series in **eccentric anomaly E** and averaging the Gauss variational equations over one orbit. The mathematical mechanism relies on the Kepler equation differential dM = (1 − e cos E) dE: when thrust Fourier series in E are substituted into the averaged equations ⟨ȯ⟩ = (1/2π)∫₀²π ȯ dM, the factor (1 − e cos E) combines with the trigonometric integrands to produce orthogonality conditions that annihilate all Fourier harmonics of order k ≥ 3. The result: **exactly 14 TFCs** determine the secular orbital element rates regardless of the original series truncation order.

The averaged secular equations take a compact linear form ⟨ȯ⟩ = A(o)·c, where c is the 14-element TFC vector and A is a 6×14 matrix depending on the current orbital elements. With 6 constraints (target elements) and 14 unknowns, the system is underdetermined. The **minimum-norm solution** c* = Aᵀ(AAᵀ)⁻¹·Δo/τ emerges from the Lagrangian L = ½cᵀc + λᵀ(Ac − b), minimizing the squared sum of Fourier coefficients—a proxy for the energy-optimal cost ∫|F|² dt via Parseval's theorem. Higher-order coefficients (k ≥ 3) are free parameters that can reshape the thrust profile for fuel optimization, constant-thrust arcs, or bang-bang operation without altering the averaged trajectory.

Ko and Scheeres (2014) further identified **6 essential TFCs** {β₁ᴿ, α₀ˢ, α₁ˢ, β₁ˢ, α₁ᵂ, β₁ᵂ} providing a minimum finite basis for arbitrary orbital maneuvers. This essential set enables dynamic interpolation across unknown maneuvers—critical for space situational awareness applications. The semi-major axis depends on just two coefficients (β₁ᴿ and α₀ˢ), while in-plane and out-of-plane dynamics decouple at first order.

Recent extensions have adapted TFCs to different element sets and anomaly variables. Ozawa (2017) developed Equinoctial TFCs (ETFCs) using the eccentric longitude K = ω̃ + E as the expansion variable, eliminating singularities at zero eccentricity. A 2024 Acta Astronautica paper reduced the nonsingular TFC count to **8 coefficients** for nearly circular orbits and incorporated these into Kalman filtering for orbit determination of continuously thrusting satellites. Nie and Gurfil (2021) proposed an alternative using mean anomaly M with resonant control laws, achieving fuel savings through artificial resonance between control and orbital periods. These parallel developments confirm that the choice of expansion variable materially affects the coefficient structure and computational efficiency.

---

## The unexplored bridge: Fourier parameterization in generalized eccentric longitude K

**No published work parameterizes thrust as Fourier series in the GEqOE generalized eccentric longitude K.** This gap is the central opportunity for a unified framework. The mathematical pathway is clear: expand P_r, P_f, P_h (thrust components in the GEqOE force decomposition) as Fourier series in K, substitute into the GEqOE equations of motion, and average over one revolution of K using the generalized Kepler equation L̃ = K − p₁ cos K + p₂ sin K (analogous to M = E − e sin E).

The expected advantages over classical TFCs in E are threefold. First, K advances more uniformly than E in perturbed environments because the generalized mean motion ν is constant under conservative perturbations alone—the classical mean motion n drifts secularly under J₂. Second, the GEqOE equations under thrust are **inherently smoother** because gravitational periodicities are absorbed, meaning the averaged equations more accurately capture the true secular evolution. Third, the generalized Kepler equation differential dL̃ = (1 − p₁ sin K − p₂ cos K) dK provides the same orthogonality mechanism that eliminates high-order harmonics in the classical formulation, but now with the generalized eccentricity components p₁, p₂ replacing e sin ω̃ and e cos ω̃. The number of surviving "generalized TFCs" and their relationship to the classical 14 coefficients remains to be derived—this is a concrete, well-defined mathematical problem.

A secondary advantage concerns the **dual interpretation for maneuver detection**: the same generalized TFC vector that parameterizes an optimal transfer also characterizes an unknown maneuver. If an observed orbit change is projected onto the generalized TFC basis via least-squares, the resulting coefficient magnitudes directly quantify the control effort—establishing a natural connection to the control-distance metric J.

---

## Heyoka provides the computational engine through Taylor integration and automatic differentiation

The heyoka Taylor integrator (Biscani & Izzo, 2021, MNRAS) uses automatic differentiation and LLVM just-in-time compilation to achieve **machine-precision integration** of arbitrary ODE systems. For a 20th-order Taylor expansion, heyoka requires roughly 16 steps per Keplerian orbit to maintain double-precision accuracy—compared to ~100 steps for IAS15 or DOP853. Its architecture decomposes ODE right-hand sides into abstract syntax trees, applies recurrence relations for elementary function Taylor coefficients, and compiles the result to SIMD-optimized machine code.

Since version 5.0, heyoka includes **built-in arbitrary-order variational equations** via the `var_ode_sys` class. Variations can be computed with respect to initial conditions, ODE parameters, or both, producing state transition tensors (STTs) to any desired order. The first-order variational equations yield the classical STM Φ(t,t₀); second-order yields the tensor Φ₂ encoding curvature of the flow; and so on. The `eval_taylor_map()` method evaluates the resulting high-order polynomial map δx_f = Pⁿ(δx₀, δp), enabling **jet transport**—propagation of entire neighborhoods of initial conditions from a single integration. This is functionally equivalent to the differential algebra (DA) approach implemented by DACE (Differential Algebra Computational Engine), but achieved through symbolic differentiation of variational equations rather than operator overloading, with the added benefit of LLVM optimization and SIMD vectorization.

For a GEqOE-based framework, heyoka's automatic variational equations are particularly valuable because the GEqOE equations of motion involve complex auxiliary quantities (generalized eccentricity g, generalized angular momentum c, generalized semi-major axis ã) whose manual differentiation would be extraordinarily tedious. Heyoka automates this entirely: the user specifies the GEqOE dynamics symbolically, and heyoka generates the variational equations to arbitrary order. **No published work yet implements GEqOE dynamics in heyoka**, but the tool's symbolic ODE specification capability makes this straightforward.

Recent versions (v7+) include operational Earth-orbit features: EGM2008 geopotential, Earth orientation parameters, atmospheric drag models (including neural-network "thermoNETs"), and JPL/VSOP ephemerides. The differentiable SGP4 propagator demonstrates that heyoka can provide gradients for optimization of satellite-related problems. Caleb et al. (2025) developed DADDy, a polynomial-based constrained solver combining DA with differential dynamic programming that achieved **23–94% runtime reduction** on low-thrust astrodynamics problems—demonstrating the viability of DA-based optimization for exactly this class of problems.

---

## The control-distance metric J bridges optimization and detection

Holzinger, Scheeres, and Alfriend (2012) defined the control-distance metric as **J = ∫ ½ u*ᵀu* dt**, the minimum energy-optimal control cost connecting two boundary states over a time interval. J = 0 implies a purely ballistic trajectory; J > 0 indicates either a maneuver or dynamics mismodeling. The metric emerges from solving an optimal control two-point boundary value problem with quadratic Lagrangian L = ½uᵀu, yielding the optimal control u*(t) = −(∂f/∂u)ᵀ p(t) through Pontryagin's maximum principle, where p(t) is the costate vector.

Under measurement uncertainty, J becomes a **quadratic form in Gaussian random variables**, approximately following a weighted chi-squared distribution. Holzinger et al. used Pearson's three-moment approximation J ~ c·χ²(ν) to set detection thresholds via hypothesis testing: the null hypothesis H₀ (no maneuver) is rejected when J exceeds a statistically determined threshold. Lubey and Scheeres (2013–2016) reformulated J in terms of pre-fit measurement residuals and integrated it into the Optimal Control-Based Estimator (OCBE), which simultaneously performs state estimation, control estimation, and maneuver detection.

The connection to TFCs is direct and powerful. When continuous thrust is parameterized as Fourier series and the dynamics are averaged, the control-distance metric reduces to **J ≈ ½cᵀWc**, a quadratic form in the TFC vector c with a weight matrix W determined by Parseval's identity and the orbit geometry. The minimum-norm TFC solution for a given orbit transfer directly yields J = ½c*ᵀWc*, providing an analytical or semi-analytical expression for the minimum control effort. Hefflin and DeMars (2024) developed a thrust-limited variant for tractable reachability computations, while Escribano et al. (2022, 2025) built stochastic hybrid systems using TFC-based admissible control regions for tracking low-thrust satellites with sparse optical data.

Reformulating J in terms of GEqOE and generalized TFCs would combine the advantages of both frameworks:

- **Smoother quadratic form**: Because GEqOE dynamics are more linear, the quadratic approximation J ~ c·χ²(ν) holds over longer time intervals and larger maneuver sizes
- **Physical interpretability**: Generalized TFCs directly quantify thrust effort against the embedded-gravity background, isolating genuine propulsive control from gravitational dynamics
- **DA-based distribution computation**: Heyoka's variational equations provide the high-order polynomial maps needed to propagate boundary-condition uncertainty through the J computation without Monte Carlo sampling

---

## DA tools and STTs enable uncertainty-aware optimization and detection

State transition tensors generalize the STM to nonlinear regimes: δx(t) = Φ·δx₀ + ½Φ₂[δx₀⊗δx₀] + ⅙Φ₃[δx₀⊗δx₀⊗δx₀] + ···, where Φₙ is the nth-order STT. Two computational approaches dominate. The **variational approach** (used by heyoka and OCEA) integrates the differential equations for each tensor order alongside the reference trajectory. The **DA/operator-overloading approach** (used by DACE) replaces floating-point numbers with truncated polynomials and propagates them through the integrator. Both produce equivalent polynomial maps; heyoka's approach benefits from LLVM optimization while DACE's is more mature in the astrodynamics community.

Boone and McMahon (2021–2025) demonstrated that STTs combined with differential dynamic programming (DDP) enable **rapid local trajectory optimization** for cislunar transfers, with directional STTs (DSTTs) reducing computational cost by aligning with sensitive directions identified via Cauchy-Green tensor eigendecomposition. For maneuver detection, Zhou et al. (2025) developed a Confidence-Dominance Maneuver Indicator (CDMI) using DA polynomial maps for recursive polynomial optimization of the observation likelihood function. Pirovano and Armellin (2024) cast maneuver estimation as a fuel-optimal second-order cone program solved in milliseconds.

The synthesis opportunity is clear: **GEqOE's superior linearization properties should directly improve DA/STT convergence**. Because GEqOE preserve Gaussianity ~36% longer than competing elements, polynomial maps in GEqOE space require lower expansion orders for equivalent accuracy. For the control-distance metric, this means the quadratic-form approximation J ~ cᵀWc remains valid over larger uncertainty domains, reducing the need for higher-order corrections and making the chi-squared threshold more reliable for maneuver detection.

---

## Toward a complete framework: identified gaps and the path forward

The table below summarizes the status of key method combinations as of early 2026:

| Method combination | Status |
|---|---|
| GEqOE + Fourier thrust parameterization | **Not yet published** |
| Fourier expansion in generalized eccentric longitude K | **Not yet published** |
| Taylor integration (heyoka) + GEqOE dynamics | **Not yet published** |
| DA/STT + GEqOE for uncertainty-aware optimization | **Not yet published** |
| Gauge freedom theory + thrust optimization | **Not yet published** |
| Control-distance metric in GEqOE + generalized TFCs | **Not yet published** |
| TFC-based maneuver detection + continuous thrust | Published (Ko & Scheeres 2016; Escribano et al. 2022) |
| DA + DDP for low-thrust optimization | Published (Caleb et al. 2025, DADDy solver) |
| D-matrix for low-thrust reachability | Partially published (Bombardelli 2024, Keplerian only) |

The proposed unified framework would proceed in four stages. **Stage 1**: Implement GEqOE equations of motion in heyoka, exploiting automatic variational equations for STM/STT computation. **Stage 2**: Derive the generalized TFC formulation by expanding thrust in K, averaging over the generalized Kepler equation, and identifying the surviving coefficients and their relationship to the classical 14. **Stage 3**: Express the control-distance metric J as a quadratic form in generalized TFCs, derive its chi-squared distribution parameters analytically, and validate against Monte Carlo sampling. **Stage 4**: Use heyoka's high-order polynomial maps to propagate boundary-condition uncertainty through J for robust maneuver detection thresholds.

The theoretical payoff is a formulation where the same generalized TFC infrastructure serves both mission design (fuel-optimal transfer via minimum-norm TFC selection with Lagrange multiplier boundary constraints) and space surveillance (maneuver detection via statistical testing of J against its noise-only distribution). The non-osculating gauge of GEqOE provides the crucial link: by absorbing gravitational complexity into element definitions, it simultaneously simplifies the optimization landscape, reduces the required Fourier truncation order, improves the validity of linearized uncertainty propagation, and produces a physically meaningful separation between gravitational dynamics and propulsive control—exactly what is needed for a control-distance metric that must distinguish genuine maneuvers from dynamics mismodeling.

---

## Conclusion

The mathematical ingredients for a GEqOE-Fourier-Taylor-DA framework are individually mature but have never been combined. Three concrete novelties would result from their integration. First, **generalized TFCs in K** would extend Hudson-Scheeres averaging theory to non-osculating elements, with the orthogonality mechanism preserved by the generalized Kepler equation and the coefficient structure simplified by the absorbed-gravity gauge. Second, **heyoka-computed variational equations of GEqOE dynamics** would eliminate the manual derivative burden that has historically limited adoption of complex orbital element formulations in optimization loops, while providing the exact high-order sensitivities needed for DDP and sequential convex programming. Third, a **control-distance metric reformulated in generalized TFC space** would unify fuel-optimal trajectory design and maneuver detection under a single quadratic-form cost function whose statistical distribution is analytically tractable through GEqOE's linearization properties. The gap analysis confirms that this territory is genuinely open as of 2026, with the closest existing work being Bombardelli's D-matrix reachability analysis (limited to Keplerian dynamics) and Caleb et al.'s DADDy solver (limited to Cartesian/MEE coordinates).