# ASTRODYN-CORE Documentation

ASTRODYN-CORE is builder-first astrodynamics tooling that keeps Orekit APIs
first-class while adding typed configuration, state-file workflows, and
mission/uncertainty helpers.

This docs site is the user-facing guide for learning and using the project.

## Start here

If you are new to the project, follow this path:

1. [Install](getting-started/install.md)
2. [Orekit data setup](getting-started/orekit-data.md)
3. [First propagation](getting-started/first-propagation.md)
4. [Run examples](getting-started/examples.md)

## Documentation map

- **Getting Started**: shortest path to a working propagation run
- **Tutorials**: guided workflows based on the example scripts
- **How-To / Cookbook**: focused recipes for specific tasks
- **Reference**: API docs, troubleshooting, and advanced links

## API tiers (recommended)

ASTRODYN-CORE exposes two practical usage tiers:

- **Facade-first (recommended)**: start with `AstrodynClient`
- **Advanced**: use `PropagatorFactory`, `ProviderRegistry`, and typed specs directly

See `README.md` in the repo root for a broader project overview and current
feature status.
