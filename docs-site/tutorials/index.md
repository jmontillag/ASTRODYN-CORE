# Tutorials Overview

Tutorials are guided versions of the example scripts with narrative context,
expected outputs, and decision guidance.

## Recommended learning paths

Choose a path based on what you want to do first.

### Path A: First propagation and mission workflows (most users)

1. [Propagation Quickstart](propagation-quickstart.md)
2. [Scenario + Mission Workflows](scenario-missions.md)
3. Then continue with cookbook recipes in [How-To / Cookbook](../how-to/index.md)

Use this path if you are building:

- basic orbit propagation scripts
- state-file pipelines
- scenario-driven mission workflows

### Path B: Covariance / STM workflows

1. [Propagation Quickstart](propagation-quickstart.md) (only if you are new)
2. [Uncertainty Workflows](uncertainty.md)
3. Then review API reference pages for `uncertainty` and `states`

Use this path if you are working on:

- covariance propagation
- STM extraction and validation
- uncertainty export/round-trip checks

### Path C: GEqOE analytical propagator workflows (advanced)

1. [Propagation Quickstart](propagation-quickstart.md) (for the facade pattern)
2. [GEqOE Propagator](geqoe.md)
3. Then compare against numerical/DSST examples in the cookbook

Use this path if you are evaluating:

- GEqOE provider integration (`kind="geqoe"`)
- direct adapter usage
- pure-numpy Taylor propagation
- performance / cached-coefficient behavior

## How tutorials relate to examples and reference docs

- `examples/` scripts are the executable truth
- tutorial pages explain why each step exists and what outputs mean
- API reference pages document function/class signatures and parameters

If a tutorial and an example differ, prefer the example and open an issue (or
patch the docs) so the tutorial can be updated.
