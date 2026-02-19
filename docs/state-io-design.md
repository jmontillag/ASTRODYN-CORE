# State I/O Design (Draft)

This document proposes a flexible state load/save subsystem for `astrodyn-core`.

## Goals

- Define orbital states in files instead of hardcoding them in scripts.
- Support single-epoch and multi-epoch state definitions.
- Keep schema independent from Orekit classes, then convert to Orekit at runtime.
- Leave room for mission events (maneuvers) and attitude directives.

## Non-Goals (initial versions)

- Full maneuver execution engine in v1.
- OEM/CCSDS round-trip fidelity in v1.
- Estimation-ready covariance/process-noise model in v1.

## Proposed module layout

- `src/astrodyn_core/states/models.py`
  - Dataclasses for serializable state schemas.
- `src/astrodyn_core/states/io.py`
  - YAML/JSON load/save functions.
- `src/astrodyn_core/states/orekit.py`
  - Conversion layer to Orekit objects.
- `src/astrodyn_core/states/validation.py`
  - Cross-field validation (frames, element sets, maneuver references).

## Core data model

### `OrbitStateRecord`

- `epoch: str` (ISO-8601 UTC)
- `frame: str` (ex: `GCRF`, `EME2000`, `ITRF_2020`)
- `representation: str` (`cartesian`, `keplerian`, `equinoctial`)
- `position_m: [x, y, z]` and `velocity_mps: [vx, vy, vz]` for cartesian
- `elements: {...}` for non-cartesian
- `mu_m3_s2: float | str` (float or predefined keyword like `WGS84`)
- `mass_kg: float | None`
- `metadata: dict[str, Any]`

### `StateSeries`

- `name: str`
- `states: list[OrbitStateRecord]` (sorted by epoch)
- `interpolation_hint: str | None` (future bounded propagator support)

### `ManeuverRecord` (v1 schema, execution later)

- `name: str`
- `trigger: {type, epoch|condition}`
- `model: {type, parameters...}` (impulsive/finite placeholder)
- `frame: str | None`

### `AttitudeRecord`

- `mode: str` (`nadir`, `lof`, `inertial`, `custom`)
- `frame: str | None`
- `params: dict[str, Any]`

### `ScenarioStateFile`

- `schema_version: int`
- `universe: dict | None`
- `spacecraft: dict | None`
- `initial_state: OrbitStateRecord | None`
- `state_series: list[StateSeries]`
- `maneuvers: list[ManeuverRecord]`
- `attitude_timeline: list[AttitudeRecord]`

## Minimal YAML shape (v1)

```yaml
schema_version: 1

initial_state:
  epoch: "2026-02-19T00:00:00Z"
  frame: "GCRF"
  representation: "keplerian"
  elements:
    a_m: 6878137.0
    e: 0.0012
    i_deg: 51.6
    argp_deg: 45.0
    raan_deg: 120.0
    anomaly_deg: 0.0
    anomaly_type: "MEAN"
  mu_m3_s2: "WGS84"
  mass_kg: 450.0

maneuvers:
  - name: "dv-1"
    trigger:
      type: "epoch"
      epoch: "2026-02-19T06:00:00Z"
    model:
      type: "impulsive"
      dv_mps: [0.0, 0.1, 0.0]
```

## Conversion strategy to Orekit

- Parse file -> typed dataclasses (`states/models.py`).
- Validate schema + cross-field consistency.
- Convert `OrbitStateRecord` to Orekit `Orbit` using:
  - `KeplerianOrbit`, `CartesianOrbit`, or `EquinoctialOrbit`.
- Resolve frame via `FramesFactory` + current universe config.
- Resolve `mu` via existing `get_mu(...)` helper.

This keeps file format stable even if Orekit internals evolve.

## Integration points with current propagation API

- New helper:
  - `BuildContext.from_state_record(record, universe=None, metadata=None)`
- New helper:
  - `load_initial_state(path) -> OrbitStateRecord`
- Example flow:
  1. `record = load_initial_state("states/leo_case.yaml")`
  2. `orbit = to_orekit_orbit(record, universe=...)`
  3. `ctx = BuildContext(initial_orbit=orbit, universe=...)`
  4. Existing factory path unchanged.

## Incremental implementation plan

1. v1 read/write:
   - Dataclasses + YAML parser + validation for `initial_state` only.
   - Orekit conversion for cartesian + keplerian.
2. v2 timeline:
   - Multi-epoch `state_series` support.
   - Optional interpolation helper.
3. v3 mission events:
   - Parse maneuver/attitude timelines.
   - Attach to propagation builders where supported.
4. v4 interoperability:
   - Import/export bridges for OEM/CCSDS where practical.

## Immediate next coding tasks

- Add `states/models.py` with `OrbitStateRecord`.
- Add `states/io.py` with `load_state_file` / `save_state_file`.
- Add `states/orekit.py` with `to_orekit_orbit`.
- Add tests:
  - valid keplerian parse
  - invalid frame/representation
  - cartesian conversion round-trip tolerance
- Add example:
  - `examples/demo_state_file_numerical.py`
