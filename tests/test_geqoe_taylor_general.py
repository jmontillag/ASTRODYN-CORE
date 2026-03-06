"""Tests for Phase 6: General perturbations in GEqOE Taylor propagator.

Validates:
1. General equations reduce to J2-only results
2. Third-body perturbations (Sun, Moon)
3. Composite perturbation model
4. Cowell ground truth with third-body
"""

import numpy as np
import pytest

from astrodyn_core.geqoe_taylor.constants import MU, A_J2
from astrodyn_core.geqoe_taylor.perturbations.j2 import J2Perturbation
from astrodyn_core.geqoe_taylor.perturbations.third_body import ThirdBodyPerturbation
from astrodyn_core.geqoe_taylor.perturbations.composite import CompositePerturbation
from astrodyn_core.geqoe_taylor.conversions import cart2geqoe, geqoe2cart
from astrodyn_core.geqoe_taylor.rhs import _is_j2_only
from astrodyn_core.geqoe_taylor.integrator import build_state_integrator

# Paper case (a): LEO circular i=45deg
R0_A = np.array([7178.1366, 0.0, 0.0])
V0_A = np.array([0.0, 5.269240572916780, 5.269240572916780])

# Reference final state after 12 days, J2 only (Appendix C)
RF_REF = np.array([-5398.929377366906, -390.257240638229, -4693.719111636971])
VF_REF = np.array([2.214482567493, -6.845637008953, -1.977748618717])

PERT_J2 = J2Perturbation()

# Arbitrary epoch for third-body tests (J2000 + 0 days)
EPOCH_JD = 2451545.0


class TestJ2GeneralPath:
    """Verify the general equations reproduce J2-only results."""

    def test_j2_is_detected_as_j2_only(self):
        """J2Perturbation should use the J2-only fast path."""
        assert _is_j2_only(PERT_J2) is True

    def test_composite_j2_only_is_not_j2_path(self):
        """CompositePerturbation with no non-conservative should still detect J2."""
        comp = CompositePerturbation(conservative=[PERT_J2])
        # Conservative + time-independent => J2-only path
        assert _is_j2_only(comp) is True

    def test_general_path_forced(self):
        """Force general path by adding a trivial non-conservative model."""
        comp = CompositePerturbation(
            conservative=[PERT_J2],
            non_conservative=[_ZeroPerturbation()],
        )
        assert _is_j2_only(comp) is False

    def test_general_j2_12day(self):
        """General equations with J2-only should match J2 fast path at 12 days."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT_J2)

        # Force general path via composite with zero non-conservative
        comp = CompositePerturbation(
            conservative=[PERT_J2],
            non_conservative=[_ZeroPerturbation()],
        )
        ta, _ = build_state_integrator(comp, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(12.0 * 86400.0)

        rf, vf = geqoe2cart(ta.state, MU, comp)
        pos_err = np.linalg.norm(rf - RF_REF)
        vel_err = np.linalg.norm(vf - VF_REF)

        # Should match to ~1e-6 km (same as J2-only test)
        assert pos_err < 1e-5, f"Position error {pos_err:.2e} km"
        assert vel_err < 1e-8, f"Velocity error {vel_err:.2e} km/s"

    def test_general_j2_nu_conserved(self):
        """nu is conserved in general equations with J2-only (E_dot=0)."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT_J2)
        nu0 = ic[0]

        comp = CompositePerturbation(
            conservative=[PERT_J2],
            non_conservative=[_ZeroPerturbation()],
        )
        ta, _ = build_state_integrator(comp, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(86400.0)

        assert ta.state[0] == pytest.approx(nu0, abs=1e-14)

    def test_non_j2_static_conservative_skips_j2_fast_path(self):
        """Static conservative models must not be routed into the J2 fast path."""
        assert _is_j2_only(_RadialQuarticPerturbation()) is False

    def test_non_j2_static_conservative_matches_forced_general(self):
        """Auto-selected path for non-J2 conservative models matches general path."""
        pert_auto = _RadialQuarticPerturbation()
        pert_forced = _RadialQuarticPerturbation(force_general=True)
        ic = cart2geqoe(R0_A, V0_A, MU, pert_auto)
        t_final = 5400.0

        ta_auto, _ = build_state_integrator(
            pert_auto, ic, tol=1e-15, compact_mode=True
        )
        ta_forced, _ = build_state_integrator(
            pert_forced, ic, tol=1e-15, compact_mode=True
        )
        ta_auto.propagate_until(t_final)
        ta_forced.propagate_until(t_final)

        rf_auto, vf_auto = geqoe2cart(ta_auto.state, MU, pert_auto)
        rf_forced, vf_forced = geqoe2cart(ta_forced.state, MU, pert_forced)

        assert np.linalg.norm(rf_auto - rf_forced) < 1e-8
        assert np.linalg.norm(vf_auto - vf_forced) < 1e-11


class TestThirdBody:
    """Third-body perturbation tests."""

    def test_sun_perturbation_builds(self):
        """Sun perturbation expression builds without error."""
        sun = ThirdBodyPerturbation("sun", EPOCH_JD)
        assert sun.is_conservative is False
        assert sun.is_time_dependent is True
        assert sun.mu_3b == pytest.approx(1.327e11, rel=1e-3)

    def test_moon_perturbation_builds(self):
        """Moon perturbation expression builds without error."""
        moon = ThirdBodyPerturbation("moon", EPOCH_JD)
        assert moon.mu_3b == pytest.approx(4902.8, rel=1e-3)

    def test_j2_sun_moon_propagation(self):
        """J2 + Sun + Moon propagation completes and differs from J2-only."""
        ic = cart2geqoe(R0_A, V0_A, MU, PERT_J2)

        # J2-only
        ta_j2, _ = build_state_integrator(PERT_J2, ic, tol=1e-15)
        ta_j2.propagate_until(12.0 * 86400.0)
        rf_j2, vf_j2 = geqoe2cart(ta_j2.state, MU, PERT_J2)

        # J2 + Sun + Moon (coarse ephemeris for fast JIT)
        sun = ThirdBodyPerturbation("sun", EPOCH_JD, thresh=1e-4)
        moon = ThirdBodyPerturbation("moon", EPOCH_JD, thresh=1e-2)
        comp = CompositePerturbation(
            conservative=[PERT_J2],
            non_conservative=[sun, moon],
        )
        ta_full, _ = build_state_integrator(comp, ic, tol=1e-15, compact_mode=True)
        ta_full.propagate_until(12.0 * 86400.0)
        rf_full, vf_full = geqoe2cart(ta_full.state, MU, comp)

        # Third-body should cause a measurable difference
        diff = np.linalg.norm(rf_full - rf_j2)
        assert diff > 1e-4, f"Sun+Moon effect too small: {diff:.2e} km"

        # But shouldn't be huge for 12-day LEO
        assert diff < 10.0, f"Sun+Moon effect too large: {diff:.2e} km"

    def test_j2_sun_moon_vs_cowell(self):
        """J2 + Sun + Moon GEqOE matches Cowell ground truth."""
        from astrodyn_core.geqoe_taylor.cowell import propagate_cowell_heyoka_full

        ic = cart2geqoe(R0_A, V0_A, MU, PERT_J2)
        t_final = 12.0 * 86400.0

        # Use coarse ephemeris for fast JIT (still adequate for 1e-2 km comparison)
        sun_thresh = 1e-4
        moon_thresh = 1e-2

        # GEqOE propagation
        sun = ThirdBodyPerturbation("sun", EPOCH_JD, thresh=sun_thresh)
        moon = ThirdBodyPerturbation("moon", EPOCH_JD, thresh=moon_thresh)
        comp = CompositePerturbation(
            conservative=[PERT_J2],
            non_conservative=[sun, moon],
        )
        ta, _ = build_state_integrator(comp, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(t_final)
        rf_geqoe, vf_geqoe = geqoe2cart(ta.state, MU, comp)

        # Cowell ground truth (same thresholds for apples-to-apples comparison)
        rf_cow, vf_cow = propagate_cowell_heyoka_full(
            R0_A, V0_A, t_final, EPOCH_JD,
            sun_thresh=sun_thresh, moon_thresh=moon_thresh,
        )

        pos_err = np.linalg.norm(rf_geqoe - rf_cow)
        vel_err = np.linalg.norm(vf_geqoe - vf_cow)

        # Should agree to < 1e-3 km
        assert pos_err < 1e-3, f"Position error vs Cowell: {pos_err:.2e} km"
        assert vel_err < 1e-6, f"Velocity error vs Cowell: {vel_err:.2e} km/s"

    def test_standalone_sun_vs_cowell(self):
        """Standalone third-body propagation should build and match Cowell."""
        from astrodyn_core.geqoe_taylor.cowell import propagate_cowell_heyoka_full

        t_final = 6.0 * 3600.0
        sun_thresh = 1e-4
        sun = ThirdBodyPerturbation("sun", EPOCH_JD, thresh=sun_thresh)
        ic = cart2geqoe(R0_A, V0_A, MU, sun)

        ta, _ = build_state_integrator(sun, ic, tol=1e-15, compact_mode=True)
        ta.propagate_until(t_final)
        rf_geqoe, vf_geqoe = geqoe2cart(ta.state, MU, sun)

        rf_cow, vf_cow = propagate_cowell_heyoka_full(
            R0_A,
            V0_A,
            t_final,
            EPOCH_JD,
            A=0.0,
            include_sun=True,
            include_moon=False,
            sun_thresh=sun_thresh,
        )

        assert np.linalg.norm(rf_geqoe - rf_cow) < 1e-5
        assert np.linalg.norm(vf_geqoe - vf_cow) < 1e-8

    def test_t0_does_not_shift_third_body_ephemeris(self):
        """Changing the integrator time origin must not change the dynamics."""
        t_rel = 6.0 * 3600.0
        t0_offset = 7.0 * 86400.0
        sun_thresh = 1e-4

        comp_0 = CompositePerturbation(
            conservative=[PERT_J2],
            non_conservative=[
                ThirdBodyPerturbation("sun", EPOCH_JD, thresh=sun_thresh)
            ],
        )
        comp_shifted = CompositePerturbation(
            conservative=[PERT_J2],
            non_conservative=[
                ThirdBodyPerturbation("sun", EPOCH_JD, thresh=sun_thresh)
            ],
        )

        ic_0 = cart2geqoe(R0_A, V0_A, MU, comp_0)
        ic_shifted = cart2geqoe(R0_A, V0_A, MU, comp_shifted)

        ta_0, _ = build_state_integrator(
            comp_0, ic_0, t0=0.0, tol=1e-15, compact_mode=True
        )
        ta_shifted, _ = build_state_integrator(
            comp_shifted,
            ic_shifted,
            t0=t0_offset,
            tol=1e-15,
            compact_mode=True,
        )

        ta_0.propagate_until(t_rel)
        ta_shifted.propagate_until(t0_offset + t_rel)

        rf_0, vf_0 = geqoe2cart(ta_0.state, MU, comp_0)
        rf_shifted, vf_shifted = geqoe2cart(ta_shifted.state, MU, comp_shifted)

        assert np.linalg.norm(rf_0 - rf_shifted) < 1e-8
        assert np.linalg.norm(vf_0 - vf_shifted) < 1e-11


class TestCompositePerturbation:
    """Composite perturbation model tests."""

    def test_j2_only_composite(self):
        """Composite with J2-only matches standalone J2."""
        comp = CompositePerturbation(conservative=[PERT_J2])
        assert comp.is_conservative is True
        assert comp.is_time_dependent is False
        assert comp.mu == PERT_J2.mu
        assert comp.A == PERT_J2.A

    def test_u_numeric_sum(self):
        """U_numeric sums conservative potentials."""
        comp = CompositePerturbation(conservative=[PERT_J2])
        u1 = PERT_J2.U_numeric(R0_A)
        u2 = comp.U_numeric(R0_A)
        assert u1 == pytest.approx(u2)


class _ZeroPerturbation:
    """Trivial non-conservative perturbation (zero force) for testing."""

    is_conservative = False
    is_time_dependent = False
    mu = MU
    A = A_J2

    def U_expr(self, x, y, z, r_mag, t, pars):
        return 0.0

    def U_numeric(self, r_vec, t=0.0):
        return 0.0

    def grad_U_expr(self, x, y, z, r_mag, t, pars):
        return 0.0, 0.0, 0.0

    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars):
        return 0.0, 0.0, 0.0

    def U_t_expr(self, x, y, z, r_mag, t, pars):
        return 0.0


class _RadialQuarticPerturbation:
    """Simple static conservative model used to verify path dispatch."""

    is_conservative = True
    is_time_dependent = False
    mu = MU
    A = 0.0

    def __init__(self, coeff: float = 1.0e14, force_general: bool = False):
        self.coeff = coeff
        self._force_general = force_general

    def U_expr(self, x, y, z, r_mag, t, pars):
        return -self.coeff / (r_mag ** 4)

    def U_numeric(self, r_vec, t=0.0):
        r_mag = np.linalg.norm(r_vec)
        return -self.coeff / (r_mag ** 4)

    def grad_U_expr(self, x, y, z, r_mag, t, pars):
        coeff = 4.0 * self.coeff / (r_mag ** 6)
        return coeff * x, coeff * y, coeff * z

    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars):
        return 0.0, 0.0, 0.0

    def U_t_expr(self, x, y, z, r_mag, t, pars):
        return 0.0
