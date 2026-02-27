#!/usr/bin/env python
"""Batch NORAD TLE -> high-fidelity numerical ephemeris workflow.

Workflow covered end-to-end:
1. Read YAML config with a target epoch and a list of NORAD IDs.
2. Resolve each NORAD ID to a TLE (from local cache).
3. Build TLE propagators and propagate to the target epoch.
4. Extract each state in the EME2000 frame.
5. Use that state to initialize a high-fidelity numerical propagator
   (repo model template + standard SpacecraftSpec defaults).
6. Generate a bounded ephemeris from target epoch to +3 days.
7. Export each trajectory in binary HDF5 format at 60 s cadence.

Run with:
    conda run -n astrodyn-core-env python examples/tle_batch_high_fidelity_ephemeris.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from _common import init_orekit, make_generated_dir


def _load_config(path: Path) -> tuple[str, tuple[int, ...]]:
    from astrodyn_core.states import parse_epoch_utc

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    target_epoch_raw = raw.get("target_epoch")
    if not isinstance(target_epoch_raw, str) or not target_epoch_raw.strip():
        raise ValueError("Config must define 'target_epoch' as a non-empty ISO-8601 string.")
    target_epoch = target_epoch_raw.strip()
    # Validate using states module helper.
    parse_epoch_utc(target_epoch)

    norad_ids_raw = raw.get("norad_ids")
    if not isinstance(norad_ids_raw, list) or not norad_ids_raw:
        raise ValueError("Config must define non-empty 'norad_ids' list.")

    norad_ids: list[int] = []
    for idx, value in enumerate(norad_ids_raw):
        try:
            norad_id = int(value)
        except Exception as exc:  # noqa: BLE001
            raise TypeError(f"norad_ids[{idx}] is not an integer: {value!r}") from exc
        if norad_id <= 0:
            raise ValueError(f"norad_ids[{idx}] must be positive, got {norad_id}.")
        norad_ids.append(norad_id)

    return target_epoch, tuple(norad_ids)


def run_workflow(config_path: Path, tle_base_dir: Path, output_dir: Path) -> None:
    init_orekit()

    from configparser import ConfigParser

    from org.orekit.frames import FramesFactory
    from spacetrack import SpaceTrackClient

    from astrodyn_core import (
        AstrodynClient,
        BuildContext,
        OrbitStateRecord,
        OutputEpochSpec,
        PropagatorKind,
        PropagatorSpec,
        SpacecraftSpec,
        get_propagation_model,
        load_dynamics_config,
    )

    from astrodyn_core.states import parse_epoch_utc

    target_epoch_cfg, norad_ids = _load_config(config_path)
    target_epoch_dt = parse_epoch_utc(target_epoch_cfg)

    repo_root = Path(__file__).resolve().parents[1]
    secrets_path = repo_root / "secrets.ini"
    if not secrets_path.exists():
        raise RuntimeError(f"Missing credentials file: {secrets_path}")

    secrets_cfg = ConfigParser()
    secrets_cfg.read(secrets_path)
    identity = secrets_cfg.get("credentials", "spacetrack_identity", fallback="").strip()
    password = secrets_cfg.get("credentials", "spacetrack_password", fallback="").strip()
    if not identity or not password:
        raise RuntimeError(
            "secrets.ini is missing credentials.spacetrack_identity or credentials.spacetrack_password."
        )
    space_track_client = SpaceTrackClient(identity=identity, password=password)

    app = AstrodynClient(
        tle_base_dir=tle_base_dir,
        tle_allow_download=True,
        space_track_client=space_track_client,
    )
    eme2000 = FramesFactory.getEME2000()

    numerical_spec = load_dynamics_config(get_propagation_model("high_fidelity"))
    numerical_spec = numerical_spec.with_spacecraft(SpacecraftSpec())

    target_date = app.state.to_orekit_date(target_epoch_cfg)
    target_epoch_iso = app.state.from_orekit_date(target_date)
    end_date = target_date.shiftedBy(3.0 * 24.0 * 3600.0)
    end_epoch_iso = app.state.from_orekit_date(end_date)
    start_day_tag = target_epoch_iso[:10]
    end_day_tag = end_epoch_iso[:10]

    epoch_spec = OutputEpochSpec(
        start_epoch=target_epoch_iso,
        end_epoch=end_epoch_iso,
        step_seconds=60.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Batch TLE -> High-Fidelity Numerical Ephemeris")
    print("=" * 72)
    print(f"Config: {config_path}")
    print(f"TLE cache: {tle_base_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target epoch: {target_epoch_iso}")
    print(f"Window: {target_epoch_iso} -> {end_epoch_iso}")
    print(f"NORAD IDs: {list(norad_ids)}")

    printed_force_models = False
    ephemerides: dict[int, Any] = {}
    exported_files: list[Path] = []
    failures: list[tuple[int, str]] = []

    for norad_id in norad_ids:
        try:
            query = app.tle.build_query(norad_id=norad_id, target_epoch=target_epoch_dt, allow_download=True)
            tle_spec = app.tle.resolve_tle_spec(query)

            tle_propagator = app.propagation.build_propagator(
                PropagatorSpec(kind=PropagatorKind.TLE, tle=tle_spec),
                BuildContext(),
            )
            tle_state = tle_propagator.propagate(target_date)
            pv_eme2000 = tle_state.getPVCoordinates(eme2000)
            position = pv_eme2000.getPosition()
            velocity = pv_eme2000.getVelocity()

            initial_state = OrbitStateRecord(
                epoch=target_epoch_iso,
                frame="EME2000",
                representation="cartesian",
                position_m=(position.getX(), position.getY(), position.getZ()),
                velocity_mps=(velocity.getX(), velocity.getY(), velocity.getZ()),
                mu_m3_s2=float(tle_state.getOrbit().getMu()),
                mass_kg=float(tle_state.getMass()),
                metadata={"source": "tle", "norad_id": norad_id},
            )

            context = app.propagation.context_from_state(initial_state)
            builder = app.propagation.build_builder(numerical_spec, context)
            numerical_propagator = builder.buildPropagator(builder.getSelectedNormalizedParameters())

            if not printed_force_models:
                force_names = [fm.getClass().getSimpleName() for fm in numerical_propagator.getAllForceModels()]
                print(f"High-fidelity force models ({len(force_names)}): {force_names}")
                printed_force_models = True

            generator = numerical_propagator.getEphemerisGenerator()
            numerical_propagator.propagate(end_date)
            ephemeris = generator.getGeneratedEphemeris()
            ephemerides[norad_id] = ephemeris

            output_path = output_dir / (
                f"norad_{norad_id}_hf_ephemeris_{start_day_tag}_to_{end_day_tag}.h5"
            )
            app.state.export_trajectory_from_propagator(
                ephemeris,
                epoch_spec,
                output_path,
                series_name=f"norad-{norad_id}-hf-3day",
                representation="cartesian",
                frame="EME2000",
            )

            exported_files.append(output_path)
            print(f"[OK] NORAD {norad_id}: exported {output_path.name}")

        except Exception as exc:  # noqa: BLE001
            failures.append((norad_id, str(exc)))
            print(f"[FAIL] NORAD {norad_id}: {exc}")

    print("\nSummary")
    print(f"  Ephemerides generated: {len(ephemerides)}")
    print(f"  Trajectories exported: {len(exported_files)}")
    if failures:
        print(f"  Failures: {len(failures)}")
        for norad_id, message in failures:
            print(f"    - NORAD {norad_id}: {message}")

    if not exported_files:
        raise RuntimeError("No trajectories exported. Check TLE cache availability and config.")


def main() -> None:
    examples_dir = Path(__file__).resolve().parent
    generated_dir = make_generated_dir()

    parser = argparse.ArgumentParser(
        description="Resolve NORAD IDs to TLE, seed high-fidelity numerical propagators, and export HDF5 trajectories."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=examples_dir / "state_files" / "tle_norad_batch.yaml",
        help="YAML file with target_epoch and norad_ids list.",
    )
    parser.add_argument(
        "--tle-base-dir",
        type=Path,
        default=generated_dir / "tle_cache",
        help="Directory containing monthly TLE cache files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=generated_dir / "tle_hf_ephemerides",
        help="Directory where exported .h5 trajectory files are written.",
    )
    args = parser.parse_args()

    run_workflow(args.config, args.tle_base_dir, args.output_dir)


if __name__ == "__main__":
    main()
