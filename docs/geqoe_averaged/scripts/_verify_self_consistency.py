#!/usr/bin/env python3
"""Phase 3b: Verify the self-consistency hypothesis.

The round-trip osc -> mean -> osc should be accurate to O(eps^2) even though
the SP correction is missing O(eps) log terms, because the osc->mean inversion
absorbs the log terms into the mean state bias.

This script:
1. Takes an osculating state
2. Computes mean via osc_to_mean (rational-only)
3. Reconstructs osculating via mean_to_osc (rational-only)
4. Measures round-trip error (should be O(eps^2))
5. Compares against the log term magnitude (should be O(eps))
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrodyn_core.geqoe_taylor import (
    MU, RE, ZonalPerturbation, cart2geqoe, geqoe2cart,
)
from geqoe_mean.constants import J_COEFFS
from geqoe_mean.coordinates import kepler_to_rv
from geqoe_mean.short_period import (
    mean_to_osculating_state,
    osculating_to_mean_state,
)


def analyze_round_trip(name, a_km, e, inc_deg, raan_deg, argp_deg, anom_deg):
    """Analyze round-trip accuracy for one orbit point."""
    r0, v0 = kepler_to_rv(a_km, e, inc_deg, raan_deg, argp_deg, anom_deg)
    pert = ZonalPerturbation(J_COEFFS, mu=MU, re=RE)
    osc = cart2geqoe(r0, v0, MU, pert)

    # Round trip: osc -> mean -> osc_rec
    mean = osculating_to_mean_state(osc, J_COEFFS, re_val=RE, mu_val=MU)
    osc_rec = mean_to_osculating_state(mean, J_COEFFS, re_val=RE, mu_val=MU)

    # Element-level errors
    diff = osc_rec - osc
    # Convert to position error
    r_osc, v_osc = geqoe2cart(osc, MU, pert)
    r_rec, v_rec = geqoe2cart(osc_rec, MU, pert)
    pos_err = np.linalg.norm(r_rec - r_osc)

    # SP correction magnitude (for reference)
    sp_correction = osc_rec - mean
    sp_mag_pos = np.linalg.norm(
        geqoe2cart(mean + sp_correction, MU, pert)[0] - geqoe2cart(mean, MU, pert)[0]
    )

    # J2 epsilon for reference
    eps = abs(J_COEFFS[2]) * (RE / a_km) ** 2

    print(f"\n  {name}:")
    print(f"    eps = J2*(RE/a)^2 = {eps:.6e}")
    print(f"    Round-trip element errors:")
    labels = ['nu', 'p1', 'p2', 'K', 'q1', 'q2']
    for i, lab in enumerate(labels):
        print(f"      d{lab:3s} = {diff[i]:+.6e}")
    print(f"    Round-trip position error: {pos_err:.6e} km")
    print(f"    SP correction position:    {sp_mag_pos:.6e} km")
    print(f"    Ratio (round-trip / SP):    {pos_err/sp_mag_pos:.6e}")
    print(f"    Ratio (round-trip / eps^2): {pos_err/eps**2:.6e} km")

    return pos_err, sp_mag_pos, eps


def scaling_test(name, a_km, e, inc_deg, raan_deg, argp_deg, anom_deg):
    """Test how round-trip error scales with lambda."""
    from geqoe_mean.short_period import evaluate_truncated_short_period

    scales = [1.0, 0.5, 0.25, 0.125]
    errors = []

    for lam in scales:
        # Scale zonal coefficients
        scaled_coeffs = {k: v * lam for k, v in J_COEFFS.items()}
        r0, v0 = kepler_to_rv(a_km, e, inc_deg, raan_deg, argp_deg, anom_deg)
        pert = ZonalPerturbation(scaled_coeffs, mu=MU, re=RE)
        osc = cart2geqoe(r0, v0, MU, pert)

        mean = osculating_to_mean_state(osc, scaled_coeffs, re_val=RE, mu_val=MU)
        osc_rec = mean_to_osculating_state(mean, scaled_coeffs, re_val=RE, mu_val=MU)

        r_osc = geqoe2cart(osc, MU, pert)[0]
        r_rec = geqoe2cart(osc_rec, MU, pert)[0]
        errors.append(np.linalg.norm(r_rec - r_osc))

    # Fit slope
    scales = np.array(scales)
    errors = np.array(errors)
    mask = errors > 0
    if np.sum(mask) >= 2:
        slope = np.polyfit(np.log(scales[mask]), np.log(errors[mask]), 1)[0]
    else:
        slope = float('nan')

    print(f"\n  {name} round-trip scaling:")
    for lam, err in zip(scales, errors):
        print(f"    lambda={lam:.4f}: pos_err={err:.6e} km")
    print(f"    Slope: {slope:.2f} (expect 2.0 for O(eps^2))")


def main():
    print("Phase 3b: Self-consistency verification")
    print("=" * 70)

    print("\n--- Round-trip accuracy at single points ---")
    analyze_round_trip("low-e", 9000, 0.05, 40, 25, 60, 20)
    analyze_round_trip("high-e", 18000, 0.65, 63, 40, 250, 35)
    analyze_round_trip("mid-e", 9000, 0.30, 40, 25, 60, 90)

    print("\n--- Round-trip scaling with lambda ---")
    scaling_test("low-e", 9000, 0.05, 40, 25, 60, 20)
    scaling_test("high-e", 18000, 0.65, 63, 40, 250, 35)
    scaling_test("mid-e", 9000, 0.30, 40, 25, 60, 90)

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("If round-trip error scales as lambda^2, this confirms O(eps^2) self-")
    print("consistency: the log terms are absorbed into the mean state definition")
    print("and cancel in the round-trip. The rational-only SP defines a valid")
    print("near-identity transformation with O(eps^2) position accuracy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
