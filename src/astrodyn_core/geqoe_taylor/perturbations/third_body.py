"""Third-body gravitational perturbation models (Sun, Moon).

Treated as non-conservative P in the GEqOE framework (Section 7.2 of
Bau et al. 2021) to avoid numerical instabilities in h and c.

Ephemeris: heyoka built-in VSOP2013 (Sun) and ELP2000 (Moon).
"""

from __future__ import annotations

import heyoka as hy
import numpy as np

from astrodyn_core.geqoe_taylor.constants import (
    AU_KM,
    COS_OBLIQUITY,
    DAYS_PER_JULIAN_CENTURY,
    DAYS_PER_JULIAN_MILLENNIUM,
    GM_MOON,
    GM_SUN,
    JD_J2000,
    MU,
    SECONDS_PER_DAY,
    SIN_OBLIQUITY,
)


def _ecliptic_to_equatorial(x_ecl, y_ecl, z_ecl):
    """Rotate from ecliptic J2000 to equatorial J2000 (heyoka expressions)."""
    x_eq = x_ecl
    y_eq = y_ecl * COS_OBLIQUITY - z_ecl * SIN_OBLIQUITY
    z_eq = y_ecl * SIN_OBLIQUITY + z_ecl * COS_OBLIQUITY
    return x_eq, y_eq, z_eq


def _ecliptic_to_equatorial_np(r_ecl: np.ndarray) -> np.ndarray:
    """Rotate from ecliptic J2000 to equatorial J2000 (NumPy)."""
    x, y, z = r_ecl
    return np.array([x,
                     y * COS_OBLIQUITY - z * SIN_OBLIQUITY,
                     y * SIN_OBLIQUITY + z * COS_OBLIQUITY])


def _sun_position_expr(epoch_jd: float, t_expr, thresh: float = 1e-9):
    """Geocentric Sun position in equatorial J2000 (km) as heyoka expressions.

    Uses VSOP2013 for the Earth-Moon barycenter, negated to get geocentric Sun.
    """
    # Convert ODE time (seconds from epoch) to Julian millennia from J2000
    t_jm = (epoch_jd - JD_J2000) / DAYS_PER_JULIAN_MILLENNIUM + (
        t_expr / (SECONDS_PER_DAY * DAYS_PER_JULIAN_MILLENNIUM)
    )

    # VSOP2013: heliocentric EMB in ecliptic J2000, AU and AU/day
    emb = hy.model.vsop2013_cartesian(3, time_expr=t_jm, thresh=thresh)

    # Geocentric Sun = -EMB (heliocentric), convert AU -> km
    sun_ecl = [-emb[i] * AU_KM for i in range(3)]
    return _ecliptic_to_equatorial(*sun_ecl)


def _moon_position_expr(epoch_jd: float, t_expr, thresh: float = 1e-6):
    """Geocentric Moon position in equatorial J2000 (km) as heyoka expressions.

    Uses ELP2000 theory.
    """
    # Convert ODE time (seconds from epoch) to Julian centuries from J2000
    t_jc = (epoch_jd - JD_J2000) / DAYS_PER_JULIAN_CENTURY + (
        t_expr / (SECONDS_PER_DAY * DAYS_PER_JULIAN_CENTURY)
    )

    # ELP2000: geocentric Moon in ecliptic J2000, km
    moon_ecl = hy.model.elp2000_cartesian_e2000(time_expr=t_jc, thresh=thresh)
    return _ecliptic_to_equatorial(moon_ecl[0], moon_ecl[1], moon_ecl[2])


class ThirdBodyPerturbation:
    """Third-body gravitational perturbation as non-conservative P.

    The third-body acceleration on a satellite at geocentric position r:
        P = mu_3b * ((r_3b - r)/|r_3b - r|^3 - r_3b/|r_3b|^3)

    The second term (indirect part) accounts for the acceleration of Earth
    by the third body.

    Args:
        body: "sun" or "moon".
        epoch_jd: Julian date corresponding to t=0 for this perturbation
            model (TDB).
        mu_3b: gravitational parameter (km^3/s^2). Auto-detected if None.
        thresh: ephemeris truncation threshold (larger = faster but less precise).
    """

    is_conservative = False
    is_time_dependent = True

    def __init__(
        self,
        body: str,
        epoch_jd: float,
        mu_3b: float | None = None,
        thresh: float | None = None,
    ):
        self.body = body.lower()
        self.epoch_jd = epoch_jd
        self.mu = MU

        if self.body == "sun":
            self.mu_3b = mu_3b if mu_3b is not None else GM_SUN
            self._thresh = thresh if thresh is not None else 1e-9
        elif self.body == "moon":
            self.mu_3b = mu_3b if mu_3b is not None else GM_MOON
            self._thresh = thresh if thresh is not None else 1e-6
        else:
            raise ValueError(f"Unknown body: {body!r}. Use 'sun' or 'moon'.")

    def _body_position_expr(self, t_expr) -> tuple:
        """Third-body position evaluated at t seconds from epoch_jd."""
        if self.body == "sun":
            return _sun_position_expr(self.epoch_jd, t_expr, self._thresh)
        return _moon_position_expr(self.epoch_jd, t_expr, self._thresh)

    def U_expr(self, x, y, z, r_mag, t, pars: dict):
        """No contribution to conservative potential U."""
        return 0.0

    def U_numeric(self, r_vec: np.ndarray, t: float = 0.0) -> float:
        """No contribution to U."""
        return 0.0

    def grad_U_expr(self, x, y, z, r_mag, t, pars: dict) -> tuple:
        """No conservative gradient."""
        return 0.0, 0.0, 0.0

    def P_expr(self, x, y, z, vx, vy, vz, r_mag, t, pars: dict) -> tuple:
        """Third-body acceleration P = mu * ((r3b-r)/|r3b-r|^3 - r3b/|r3b|^3)."""
        x3, y3, z3 = self._body_position_expr(t)

        # Relative position: third body - satellite
        dx = x3 - x
        dy = y3 - y
        dz = z3 - z
        d2 = dx * dx + dy * dy + dz * dz
        d_mag = hy.sqrt(d2)
        d3 = d2 * d_mag

        # Third body distance from Earth
        r3_2 = x3 * x3 + y3 * y3 + z3 * z3
        r3_mag = hy.sqrt(r3_2)
        r3_3 = r3_2 * r3_mag

        mu = self.mu_3b
        Px = mu * (dx / d3 - x3 / r3_3)
        Py = mu * (dy / d3 - y3 / r3_3)
        Pz = mu * (dz / d3 - z3 / r3_3)
        return Px, Py, Pz

    def U_t_expr(self, x, y, z, r_mag, t, pars: dict):
        """No time-dependent conservative potential."""
        return 0.0
