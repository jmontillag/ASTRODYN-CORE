# How-To: Batch NORAD TLE to High-Fidelity Ephemeris

This recipe runs the repository example that:

1. reads a YAML list of NORAD IDs and a target epoch
2. resolves/downloads TLE data for each NORAD
3. extracts states in `EME2000` at the target epoch
4. seeds high-fidelity numerical propagators
5. generates bounded ephemerides from `t0` to `t0 + 3 days`
6. exports trajectories to HDF5 at 60-second cadence

## Run

From the repository root:

```bash
conda run -n astrodyn-core-env python examples/tle_batch_high_fidelity_ephemeris.py
```

## Inputs and credentials

- Input config: `examples/state_files/tle_norad_batch.yaml`
- Space-Track credentials: repo-root `secrets.ini`
  - `credentials.spacetrack_identity`
  - `credentials.spacetrack_password`

## Outputs

Generated files are written under:

- `examples/generated/tle_hf_ephemerides/`

Naming pattern:

- `norad_<id>_hf_ephemeris_<YYYY-MM-DD>_to_<YYYY-MM-DD>.h5`

## Related references

- [Run Examples](../getting-started/examples.md)
- [How-To / Cookbook](index.md)
- `examples/README.md`
