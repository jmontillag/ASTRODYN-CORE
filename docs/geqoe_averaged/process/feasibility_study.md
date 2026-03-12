# Feasibility Study: Mean / Averaged GEqOE for Fast Maneuver Metrics

## Objective

Assess whether the current GEqOE formulation in ASTRODYN-CORE can be turned into
an averaged or mean-element analytical model suitable for fast maneuver
detection metrics, with emphasis on:

- the GEqOE formulation of Bau, Hernando-Ayuso, and Bombardelli (2021),
- the thrust-coefficient / control-distance direction in
  `docs/V1_ManeuverDetection_TFC.pdf`,
- the secular-plus-periodic analytical propagation idea in
  `docs/V1_Continuous-Thrust Analytical State Propagation for J2-Perturbed Two-Body Problem.pdf`.

The question here is feasibility, not implementation.

## Short Answer

- A **formal first-order averaging of GEqOE is feasible**.
- A **closed-form analytical averaged GEqOE for arbitrary potential-based
  perturbations is not feasible in general**.
- A **restricted averaged GEqOE theory aimed at maneuver metrics is promising**
  if the scope is narrowed to:
  - a static conservative baseline, ideally `J2` or `J2-Jn`,
  - small non-conservative thrust-like forcing,
  - averaging with respect to the propagated generalized eccentric longitude `K`,
  - and an explicit distinction between secular dynamics and short-period
    reconstruction.

That restricted problem looks publishable and aligned with this branch. The
fully general one does not.

## Sources Reviewed

- `docs/s10569-021-10049-1.pdf`
- `docs/V1_ManeuverDetection_TFC.pdf`
- `docs/V1_Continuous-Thrust Analytical State Propagation for J2-Perturbed Two-Body Problem.pdf`
- `src/astrodyn_core/geqoe_taylor/rhs.py`
- `src/astrodyn_core/geqoe_taylor/conversions.py`
- `src/astrodyn_core/geqoe_taylor/perturbations/`
- `docs/geqoe_taylor_plan.md`
- `docs/geqoe_taylor/compass_continuous_markdown.md`

## What Matters In The Current GEqOE Formulation

### 1. The branch already uses `K` as the fast angle

The propagated state is:

```text
y = (nu, p1, p2, K, q1, q2)
```

This is not cosmetic. In the current implementation, `K` is used instead of `L`
because the generalized Kepler equation is implicit in `K` when `L` is given:

```math
L = K + p_1 \cos K - p_2 \sin K
```

For averaging work, this is actually helpful: `K` is the natural fast variable
for the current codebase and should remain the anomaly used for any averaged
theory in this branch.

### 2. GEqOE already absorb part of the conservative dynamics

GEqOE are non-osculating by construction. Conservative potential terms are
embedded in the element definitions through the generalized energy and
generalized Laplace geometry. This means:

- the variables are already smoother than classical osculating elements,
- some gravitational short-period structure is already removed,
- a future "mean GEqOE" would be the mean of an already non-osculating set.

That is good for smoothness, but it also means the transformation between
instantaneous Cartesian state and "mean GEqOE" will not be trivial.

### 3. For static conservative potentials, `nu` is especially attractive

From the GEqOE equations and the current implementation:

- if `U = U(r)` is conservative and time-independent,
- and `P = 0`,
- then `E_dot = 0`,
- hence `nu_dot = 0`.

So for static conservative baselines, `nu` is already an exact integral. That is
an unusually clean property for a mean-element theory.

### 4. The natural averaging measure is explicit in `K`

In the unperturbed limit:

```math
\dot{K}_0 = \frac{w}{r}, \qquad
w = \sqrt{\mu / a}, \qquad
r = a(1 - p_1 \sin K - p_2 \cos K)
```

Therefore:

```math
\frac{dt}{dK} = \frac{r}{w}
```

and a first-order orbital average of any quantity `f` can be written as:

```math
\langle f \rangle
= \frac{1}{T}\int_0^T f\,dt
= \frac{1}{2\pi a}\int_0^{2\pi} r(K)\,f(K)\,dK
```

with:

```math
\frac{r}{a} = 1 - p_1 \sin K - p_2 \cos K
```

This is the direct GEqOE analogue of the classical `dt = (1 - e cos E) dE / n`
weight used in averaged Gauss-equation theory.

## What The TFC / Analytical-Thrust Papers Actually Assume

The two thrust papers are useful, but their closure relies on assumptions much
narrower than "general potential-based perturbations".

### Common structure

They:

1. use a classical element set and Gauss-type variation equations,
2. average over a fast anomaly,
3. split the motion into secular plus periodic parts,
4. approximate thrust with a truncated Fourier series,
5. ignore or simplify at least some coupling terms,
6. derive a low-dimensional algebraic map from thrust coefficients to secular
   element drift.

### Critical simplification

In the J2-plus-thrust paper, the secular rates are approximated as the sum of:

- "J2-only secular drift", and
- "thrust-only secular drift".

The coupling between J2 and thrust is explicitly neglected.

That simplification is not a detail. It is one of the reasons an analytical
closure exists.

### Why they get compact coefficient sets

The classical averaged equations become linear in the thrust Fourier
coefficients because:

- the anomaly weight is simple,
- the Gauss kernels are low-order trigonometric functions,
- orthogonality kills higher harmonics.

This is the key feature we would want to recover, if possible, in GEqOE.

## Candidate GEqOE Averaging Framework

Let the slow state be:

```math
z = (\nu, p_1, p_2, q_1, q_2)
```

and let `K` be the fast phase. A standard first-order averaging setup would be:

```math
\dot{z} = \varepsilon f_1(z, K, t) + O(\varepsilon^2)
```

```math
\dot{K} = \frac{w(z)}{r(z,K)} + \varepsilon g_1(z, K, t) + O(\varepsilon^2)
```

Then the first-order secular system is:

```math
\dot{\bar{z}}
= \frac{1}{2\pi \bar{a}}
\int_0^{2\pi}
r(\bar{z},K)\,f_1(\bar{z},K,t)\,dK
+ O(\varepsilon^2)
```

with a separate equation for the mean phase if needed.

This part is straightforward conceptually. The hard part is not defining the
average. The hard part is getting a useful, closed, analytical expression for it.

## Feasibility By Scope

| Target problem | Feasibility | Assessment |
|---|---|---|
| Formal first-order averaged GEqOE for small perturbations | Yes | Standard single-fast-angle averaging applies with `K` as fast anomaly. |
| Closed-form mean GEqOE for arbitrary `U(r,t)` | No | No universal analytical closure exists for orbit averages of arbitrary potentials and their derivatives along the generalized ellipse. |
| Closed-form mean GEqOE for static zonal gravity (`J2`, maybe `J2-Jn`) | Probably yes, but hard | Term-by-term analytical averaging looks possible, but it is separate theory work. |
| Semi-analytical mean GEqOE for broad conservative models | Yes | Numerical quadrature or symbolic term-by-term averaging is possible, but that is not the same as a compact analytical law. |
| Averaged GEqOE thrust model on top of static conservative baseline | Yes, and most promising | This is the best match for fast maneuver metrics. |
| Fully general mean GEqOE including third-body ephemerides in closed form | No | Once the forcing depends on absolute time / ephemerides, a compact anomaly-only analytical average is lost. |

## Why A General Potential-Based Closed Form Fails

The GEqOE equations depend on combinations such as:

- `2U - r F_r`,
- `F_h`,
- `U_t`,
- `E_dot`,
- and `U(x(K), y(K), z(K), t)`.

For a truly general potential, the orbit averages require integrals of the form:

```math
\int_0^{2\pi}
\Phi\!\left(
U(x(K),y(K),z(K),t),
\nabla U(x(K),y(K),z(K),t),
U_t(x(K),y(K),z(K),t)
\right)
dK
```

There is no universal finite-dimensional basis that makes those integrals reduce
to a compact closed form for arbitrary `U`.

In practice, the difficulty comes from three places:

1. **Arbitrary spatial structure**
   - Even for static conservative potentials, the dependence on `K` is not
     generally a low-order trigonometric polynomial after substitution through
     `r`, `X`, `Y`, and `zhat`.

2. **Explicit time dependence**
   - If `U` depends on ephemerides or other time-varying inputs, the average is
     not an anomaly-only integral anymore.

3. **Mean/osculating transformation**
   - A useful mean-element theory needs both the secular flow and the
     short-period reconstruction. For arbitrary `U`, that transformation is
     model-specific.

So the obstacle is not "averaging theory does not apply". The obstacle is
"there is no single compact analytical closure for arbitrary potential models".

## Why A Maneuver-Focused Restricted Theory Looks Promising

This is the important positive result.

If the goal is fast maneuver detection metrics, the most relevant reduced model
is not "arbitrary conservative potential in one closed form". It is:

- a static conservative baseline,
- plus small thrust-like forcing,
- expressed directly in `K`,
- with an averaged secular law for the thrust contribution.

### Structural reason this is promising

In the current GEqOE formulation:

- `r / a` is first harmonic in `K`,
- `X / a` is affine in `sin K` and `cos K`,
- `Y / a` is affine in `sin K` and `cos K`,
- after multiplying by `dt / dK = r / w`, many `1 / r` factors cancel.

That means the thrust-driven averaged kernels are much more likely to collapse
to a **finite low-order harmonic dependence** than the gravity-driven kernels.

For example, under thrust-only forcing:

```math
\dot{\nu} \propto \dot{E}
= \dot{r} P_r + \frac{h}{r} P_f
```

and after converting to a `K` average, the `r` weight removes the explicit
`1/r` in the transverse term. The resulting integrand is built from:

- constants,
- `sin K`,
- `cos K`,
- and the thrust components `P_r(K), P_f(K), P_h(K)`.

That is exactly the kind of structure that can produce a reduced Fourier
coefficient set after averaging.

### Important nuance

This does **not** prove that the GEqOE analogue of the classical 14 TFCs exists
with the same count. It does strongly suggest that:

- a finite generalized TFC closure should exist for the thrust-driven secular
  terms,
- the coefficient count should stay low,
- and the right anomaly for that theory is `K`, not classical `E`.

That is likely the right research target.

## What Looks Unworkable vs. What Looks Workable

### Unworkable target

"Derive one analytical mean-element GEqOE propagator that is closed-form for a
general potential-based perturbation model."

This is too broad. For ephemeris-driven or otherwise arbitrary `U`, it is not a
realistic analytical target.

### Workable target

"Derive a first-order averaged GEqOE secular model in `K` for small
thrust-driven maneuvers on top of a static conservative baseline, together with
short-period reconstruction and a coefficient-space maneuver metric."

This is narrow enough to be tractable and broad enough to matter.

## Recommended Research Direction

### Phase 1: Define the exact problem narrowly

Start with:

- baseline conservative dynamics = `J2` first, then `J2-Jn`,
- small RTN thrust components,
- thrust expressed as Fourier series in `K`,
- first-order averaging only,
- coupling between zonals and thrust either neglected initially or retained only
  to first order if the algebra stays manageable.

This reproduces the simplifying logic of the reference thrust papers, but in a
GEqOE-native way.

### Phase 2: Derive secular GEqOE thrust rates

Use the existing `K`-based GEqOE equations and derive:

```math
\langle \dot{\nu} \rangle,\;
\langle \dot{p}_1 \rangle,\;
\langle \dot{p}_2 \rangle,\;
\langle \dot{q}_1 \rangle,\;
\langle \dot{q}_2 \rangle
```

as functions of:

- the slow GEqOE state,
- a finite set of Fourier-in-`K` thrust coefficients.

This is the step that decides whether a generalized TFC basis emerges cleanly.

### Phase 3: Derive short-period reconstruction

Do not stop at secular rates. For comparison against measurements or full
propagation, derive:

```math
x_{\text{full}} = x_{\text{mean}} + \varepsilon \eta(x_{\text{mean}}, K, t) + O(\varepsilon^2)
```

Without this, the result is an averaged slow-flow model, not a usable
mean-element theory.

### Phase 4: Build the maneuver metric in coefficient space

If the thrust basis reduces cleanly, the natural next step is:

```math
J \approx c^T W c
```

with boundary conditions enforced through the averaged GEqOE mapping. That is
the direct bridge to fast maneuver detection metrics.

### Phase 5: Only then consider broader conservative models

After the thrust-focused theory works for `J2` or `J2-Jn`, decide whether to
extend the conservative baseline by:

- term-by-term zonal averaging,
- semi-analytical quadrature,
- or a hybrid "analytical thrust + numerically averaged gravity" model.

That extension should be treated as optional.

## Practical Recommendation For This Branch

The best next research target is:

> A GEqOE averaged thrust theory in the propagated generalized eccentric
> longitude `K`, built on top of a static conservative baseline, with explicit
> short-period reconstruction and a quadratic control-distance-like metric.

The wrong next target is:

> A closed-form averaged GEqOE propagator for arbitrary potential-based
> perturbations.

## Final Verdict

### Feasible

- first-order averaging in GEqOE with `K` as the fast variable,
- a maneuver-focused mean/averaged GEqOE theory,
- a reduced Fourier-in-`K` thrust basis for secular drift,
- a fast control-distance-like metric derived from that reduced basis.

### Feasible but high effort

- a full semi-analytical GEqOE mean-element framework with short-period
  reconstruction for `J2-Jn`.

### Not realistically feasible as a single analytical theory

- one compact GEqOE mean-element closure valid for arbitrary general
  potential-based perturbations, especially time-dependent or ephemeris-driven
  ones.

## Bottom Line

The idea is worth pursuing, but only if it is framed correctly.

If the target is **fast maneuver detection metrics**, the promising path is not
"general potential-based mean GEqOE". The promising path is **averaged
thrust-in-`K` GEqOE over a static conservative baseline**. That is the part that
matches the analytical-thrust literature, the current branch architecture, and
the likely metric payoff.
