# `tle` module

Download/cache/parse/resolve utilities for Two-Line Element (TLE) data.

## Purpose

This module resolves dynamic TLE data (local cache and optional Space-Track download)
into static line pairs consumed by the propagation layer.

This keeps architecture boundaries clean:
- `tle/` handles data retrieval and selection.
- `propagation/` remains deterministic and side-effect free.

## Key API

- `TLEQuery`: request model (`norad_id`, `target_epoch`, `base_dir`, `allow_download`)
- `resolve_tle_record(query, space_track_client=...)`
- `resolve_tle_spec(query, space_track_client=...)`
- `parse_tle_file(...)`, `find_best_tle_in_file(...)`
- `download_tles_for_month(...)`, `ensure_tles_available(...)`

## Typical flow

1. Create `TLEQuery` for NORAD + target epoch.
2. Resolve to `TLESpec` via `resolve_tle_spec(...)`.
3. Build propagator as usual with `PropagatorSpec(kind='tle', tle=...)`.

## Notes

- Downloads are optional and require an authenticated client exposing `gp_history(...)`.
- Without download mode, local cache files must already exist in `{base_dir}/{norad_id}/`.
