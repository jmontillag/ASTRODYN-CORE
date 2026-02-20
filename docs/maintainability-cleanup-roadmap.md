# ASTRODYN-CORE Maintainability Cleanup Roadmap

Last updated: 2026-02-20
Status: In progress

Progress snapshot:

- Phase A: completed (examples, façade integration test, delegation caching, docs sync)
- Phase B: completed (Slices 1 mission, 2 uncertainty, 3 states/orekit, 4 propagation/config)
- Phase C: next (API governance and boundary hardening)

## 1) Purpose and Intent

This document defines a structured cleanup and organization effort aimed at long-term maintainability, while preserving current functionality and backward compatibility.

Primary intent:

- Consolidate user-facing workflows under coherent façade classes.
- Reduce inter-module coupling and private cross-imports.
- Decompose oversized modules into clear responsibility slices.
- Keep Orekit-native expert workflows fully available.
- Improve documentation and examples so the intended architecture is the default path for users.

## 2) Scope

In scope:

- API ergonomics and architecture consistency (façades + module boundaries).
- Internal refactors that reduce complexity without changing results.
- Documentation and examples alignment.
- Targeted tests for integration and regression safety.

Out of scope:

- New mission physics capabilities.
- New force models or major algorithmic changes.
- Breaking API removals in this cycle.

## 3) Current-State Assessment Summary

Observed strengths:

- Strong typed model layer and broad test coverage.
- Clear separation of major domains (states, mission, uncertainty, tle, propagation).
- New façade pattern now exists (AstrodynClient, MissionClient, TLEClient, UncertaintyClient).

Observed maintainability hotspots:

1. Very large modules with mixed responsibilities:
   - uncertainty/propagator.py
   - mission/maneuvers.py
   - propagation/config.py
   - states/orekit.py
   - states/client.py
2. Cross-module private helper imports (underscore-prefixed symbols) in mission internals.
3. Examples still primarily use StateFileClient directly, so unified façade adoption is inconsistent.
4. Root public API is broad and mixes high-level and low-level exports.
5. Architecture docs and implementation plan are partially behind recent façade evolution.

## 4) Guiding Principles

- Preserve behavior first, improve structure second.
- Keep public API additive and backward compatible during cleanup.
- Prefer small, test-backed refactor increments.
- Avoid moving multiple unrelated concerns in one change set.
- Make intended usage obvious in examples and docs.

## 5) Proposed Change Program (Phased)

## Phase A — Consistency and Quick Wins (Low risk)

Objectives:

- Align examples/docs/tests with the façade-first architecture.
- Introduce low-cost structural improvements with immediate payoff.

Work items:

1. Example modernization
   - Migrate quickstart, scenario_missions, uncertainty examples to AstrodynClient-first usage.
   - Keep minimal notes showing direct StateFileClient compatibility for advanced users.

2. Façade integration coverage
   - Add one end-to-end test using app.state, app.mission, app.uncertainty, app.tle in one flow.
   - Keep focused domain tests unchanged.

3. Documentation alignment
   - Update architecture docs and implementation plan to reflect current client composition and intended usage.

4. Internal micro-cleanup
   - Cache composed clients in StateFileClient to avoid repeated re-instantiation on every delegated call.

Expected outcome:

- Architecture intent becomes explicit in examples and docs.
- User path is simplified without removing expert paths.

## Phase B — Module Decomposition (Medium risk)

Objectives:

- Reduce file size and responsibility overlap in hotspot modules.
- Lower coupling and improve testability.

Work items:

1. mission/maneuvers.py split
   - timeline resolution
   - trigger resolution
   - intent solvers
   - impulse/vector transforms
   - series simulation/export orchestration

2. uncertainty/propagator.py split
   - matrix/jacobian conversion helpers
   - orbit/frame covariance transform layer
   - STM runner abstraction
   - record conversion/output adapter

3. propagation/config.py split
   - universe config loading and normalization
   - dynamics parsing
   - spacecraft parsing
   - force parser registry

4. states/orekit.py split
   - date/frame/mu resolvers
   - state↔orbit conversion
   - ephemeris export/sampling

Expected outcome:

- Smaller, cohesive modules with clearer ownership.
- Easier onboarding and safer changes.

## Phase C — API Governance and Boundary Hardening (Medium risk)

Objectives:

- Make long-term API evolution predictable.
- Prevent accidental coupling regressions.

Work items:

1. Public vs internal contract definition
   - Introduce clear internal modules and avoid underscore imports across sibling modules.

2. Root API curation
   - Keep root exports ergonomic but intentional.
   - Consider dedicated public_api documentation page for stable symbols.

3. Compatibility policy
   - Introduce deprecation process and timeline for future API reshaping.

4. CI quality gates (if not already present)
   - Add checks for tests + lint + import hygiene (including rule against private cross-module imports where feasible).

Expected outcome:

- More stable upgrade path and lower maintenance entropy over time.

## 6) Proposed Execution Process

For each phase:

1. Define exact acceptance criteria before edits.
2. Implement in small PR-sized slices.
3. Run focused tests nearest to changed modules first.
4. Run broader regression suites.
5. Update docs/examples in the same slice when user-visible behavior changes.
6. Record decisions and follow-up items in docs/implementation-plan.md.

Refactor safety controls:

- No behavior changes without explicit test updates proving intended differences.
- Keep backward-compatible method names and signatures while migrating internals.
- Prefer adapter/shim approach during transitions.

## 7) Assumptions

- Current numerical behavior and outputs are correct enough to preserve.
- Existing test suite remains the baseline regression oracle.
- Orekit wrapper behavior is stable for current supported environment.
- Users rely on current public symbols from package root; removals are deferred.

## 8) Risks and Mitigations

Risk: Hidden coupling breaks during module splits.
Mitigation: Split by adapter-first strategy and preserve import shims until tests pass.

Risk: Example updates drift from tested paths.
Mitigation: Add tests that execute example-equivalent flows using façade API.

Risk: Public API bloat continues.
Mitigation: Introduce explicit API governance in Phase C and enforce in review.

Risk: Refactor fatigue due to broad scope.
Mitigation: Execute phase-by-phase with measurable, independently shippable milestones.

## 9) Acceptance Criteria for Program Completion

Phase A complete when:

- Main examples default to AstrodynClient usage.
- New façade integration test is green.
- Docs reflect façade-first architecture.
- StateFileClient delegated components are cached and tested.

Phase B complete when:

- Identified hotspot modules are decomposed into cohesive submodules.
- No regression in mission/uncertainty/state/tle suites.
- Private cross-module imports in mission lane are eliminated or reduced to documented exceptions.

Phase C complete when:

- Public/internal boundary policy is documented and enforced in CI/review checks.
- Root API curation decision is documented and applied.
- Deprecation workflow exists for future cleanup cycles.

## 10) Assumed Final State (Target Architecture)

User-facing architecture:

- AstrodynClient is the default ergonomic entry point for most workflows.
- Domain façades remain available for focused usage:
  - StateFileClient
  - MissionClient
  - UncertaintyClient
  - TLEClient

Internal architecture:

- Large modules are decomposed by responsibility.
- Cross-module private helper imports are removed from normal code paths.
- Shared conversion and utility layers are centralized and reusable.

Project hygiene:

- Examples and docs represent the intended API usage pattern.
- Tests include both domain-level and façade integration coverage.
- API evolution is governed by explicit compatibility and deprecation policy.

## 11) Proposed Immediate Next Step

Start Phase A with a single execution batch:

1. Update examples to AstrodynClient-first.
2. Add façade end-to-end integration test.
3. Cache delegated clients inside StateFileClient.
4. Update architecture and implementation docs accordingly.

This provides the highest maintainability gain for the least risk and sets a clean baseline for deeper module decomposition.