#include "conversions.hpp"
#include "kepler_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace astrodyn_core {
namespace geqoe {

// ============================================================================
// rv2geqoe  --  Cartesian -> GEqOE (per Giulio Bau's formulation with J2)
// ============================================================================
void rv2geqoe(
    const double* y_in,
    double* eq_out,
    size_t N,
    double J2, double Re, double mu
) {
    // Normalisation constants
    const double L     = Re;
    const double T     = std::sqrt(Re * Re * Re / mu);
    const double inv_L = 1.0 / L;
    const double T_over_L = T / L;           // = T * inv_L
    // mu_norm = 1.0 (implicit -- all mu_norm factors are unity)
    const double A     = J2 / 2.0;
    const double inv_T = 1.0 / T;

    for (size_t i = 0; i < N; ++i) {
        const double* y = y_in + i * 6;
        double* eq      = eq_out + i * 6;

        // --- Normalised position and velocity ---
        double rx = y[0] * inv_L;
        double ry = y[1] * inv_L;
        double rz = y[2] * inv_L;
        double vx = y[3] * T_over_L;
        double vy = y[4] * T_over_L;
        double vz = y[5] * T_over_L;

        // --- Geometric quantities ---
        double r2 = rx * rx + ry * ry + rz * rz;
        double r  = std::sqrt(r2);
        double inv_r = 1.0 / r;
        double rp = (rx * vx + ry * vy + rz * vz) * inv_r;   // radial velocity
        double v2 = vx * vx + vy * vy + vz * vz;

        // --- J2 potential ---
        double zg  = rz * inv_r;
        double r3  = r * r2;
        double U   = -A / r3 * (1.0 - 3.0 * zg * zg);

        // --- Radial unit vector ---
        double erx = rx * inv_r, ery = ry * inv_r, erz = rz * inv_r;

        // --- Angular momentum  h = r x v ---
        double hx = ry * vz - rz * vy;
        double hy = rz * vx - rx * vz;
        double hz = rx * vy - ry * vx;
        double h2 = hx * hx + hy * hy + hz * hz;
        double h  = std::sqrt(h2);
        double inv_h = 1.0 / h;

        double Ueff = h2 / (2.0 * r2) + U;

        // --- Angular momentum unit vector ---
        double ehx = hx * inv_h, ehy = hy * inv_h, ehz = hz * inv_h;

        // --- In-plane perpendicular unit vector  ef = eh x er ---
        double efx = ehy * erz - ehz * ery;
        double efy = ehz * erx - ehx * erz;
        double efz = ehx * ery - ehy * erx;

        // --- Total energy & mean motion ---
        double E_val   = v2 / 2.0 - 1.0 / r + U;    // mu_norm = 1
        double neg2E   = -2.0 * E_val;
        double nu_norm = std::pow(neg2E, 1.5);        // (1/mu) * (-2E)^1.5, mu=1

        // --- Orientation parameters q1, q2 ---
        // eh . ez = ehz,  eh . ex = ehx,  eh . ey = ehy
        double ehez = ehz;
        double inv_1pehez = 1.0 / (1.0 + ehez);
        double q1 =  ehx * inv_1pehez;
        double q2 = -ehy * inv_1pehez;

        // --- Equinoctial frame vectors eX, eY ---
        double q1s = q1 * q1, q2s = q2 * q2, q1q2 = q1 * q2;
        double eTerm = 1.0 / (1.0 + q1s + q2s);

        double eXx = eTerm * (1.0 - q1s + q2s);
        double eXy = eTerm * (2.0 * q1q2);
        double eXz = eTerm * (-2.0 * q1);

        double eYx = eTerm * (2.0 * q1q2);
        double eYy = eTerm * (1.0 + q1s - q2s);
        double eYz = eTerm * (2.0 * q2);

        // --- Generalised eccentricity vector ---
        double c = std::sqrt(2.0 * r2 * Ueff);
        double c_over_r = c * inv_r;

        // gv = rp * er + (c/r) * ef
        double gvx = rp * erx + c_over_r * efx;
        double gvy = rp * ery + c_over_r * efy;
        double gvz = rp * erz + c_over_r * efz;

        // cross1 = rv x gv
        double c1x = ry * gvz - rz * gvy;
        double c1y = rz * gvx - rx * gvz;
        double c1z = rx * gvy - ry * gvx;

        // g = gv x cross1 - er   (mu_norm = 1)
        double gx = (gvy * c1z - gvz * c1y) - erx;
        double gy = (gvz * c1x - gvx * c1z) - ery;
        double gz = (gvx * c1y - gvy * c1x) - erz;

        // --- Eccentricity-like parameters ---
        double p1_val = gx * eYx + gy * eYy + gz * eYz;    // g . eY
        double p2_val = gx * eXx + gy * eXy + gz * eXz;    // g . eX

        // --- Position projections onto equinoctial frame ---
        double X = rx * eXx + ry * eXy + rz * eXz;         // rv . eX
        double Y = rx * eYx + ry * eYy + rz * eYz;         // rv . eY

        // --- True longitude ---
        double p1s_val = p1_val * p1_val;
        double p2s_val = p2_val * p2_val;
        double beta     = std::sqrt(1.0 - p1s_val - p2s_val);
        double alpha    = 1.0 / (1.0 + beta);
        // Match Python: (mu_norm / nu_norm**2) ** (1/3)  with mu_norm=1
        double a        = std::pow(1.0 / (nu_norm * nu_norm), 1.0 / 3.0);
        double inv_a_beta = 1.0 / (a * beta);

        double cosK_val = p2_val + inv_a_beta * ((1.0 - alpha * p2s_val) * X
                                                - alpha * p1_val * p2_val * Y);
        double sinK_val = p1_val + inv_a_beta * ((1.0 - alpha * p1s_val) * Y
                                                - alpha * p1_val * p2_val * X);

        double Lr = std::atan2(sinK_val, cosK_val)
                  + inv_a_beta * (X * p1_val - Y * p2_val);

        // --- Output (denormalise nu) ---
        eq[0] = nu_norm * inv_T;
        eq[1] = q1;
        eq[2] = q2;
        eq[3] = p1_val;
        eq[4] = p2_val;
        eq[5] = Lr;
    }
}

// ============================================================================
// geqoe2rv  --  GEqOE -> Cartesian
// ============================================================================
void geqoe2rv(
    const double* eq_in,
    double* rv_out,
    double* rpv_out,
    size_t N,
    double J2, double Re, double mu
) {
    const double L        = Re;
    const double T        = std::sqrt(Re * Re * Re / mu);
    const double inv_T    = 1.0 / T;
    const double L_over_T = L * inv_T;
    const double A        = J2 / 2.0;
    const double TWO_PI   = 2.0 * M_PI;
    const double ONE_THIRD = 1.0 / 3.0;

    // --- Vectorised Kepler solve (matches Python np.all convergence) --------
    // Extract Lr, p1, p2 into temporary arrays, call the vectorised solver
    // so the convergence path is identical to the Python reference.
    std::vector<double> Lr_vec(N), p1_vec(N), p2_vec(N), K_vec(N);
    std::vector<double> nu_vec(N);

    for (size_t i = 0; i < N; ++i) {
        const double* eq = eq_in + i * 6;
        nu_vec[i] = eq[0] * T;

        double Lr_mod = std::fmod(eq[5], TWO_PI);
        if (Lr_mod < 0.0) Lr_mod += TWO_PI;
        Lr_vec[i] = Lr_mod;

        p1_vec[i] = eq[3];
        p2_vec[i] = eq[4];
    }

    solve_kep_gen(Lr_vec.data(), p1_vec.data(), p2_vec.data(),
                  K_vec.data(), N);

    // --- Per-state Cartesian reconstruction ---------------------------------
    for (size_t i = 0; i < N; ++i) {
        const double* eq = eq_in + i * 6;
        double* rv  = rv_out  + i * 3;
        double* rpv = rpv_out + i * 3;

        double nu_norm = nu_vec[i];
        double q1 = eq[1], q2 = eq[2];
        double p1 = p1_vec[i], p2 = p2_vec[i];

        double K    = K_vec[i];
        double sinK = std::sin(K);
        double cosK = std::cos(K);

        // Equinoctial frame
        double q1s = q1 * q1, q2s = q2 * q2, q1q2 = q1 * q2;
        double eTerm = 1.0 / (1.0 + q1s + q2s);

        double eXx = eTerm * (1.0 - q1s + q2s);
        double eXy = eTerm * (2.0 * q1q2);
        double eXz = eTerm * (-2.0 * q1);

        double eYx = eTerm * (2.0 * q1q2);
        double eYy = eTerm * (1.0 + q1s - q2s);
        double eYz = eTerm * (2.0 * q2);

        // Orbital shape
        double p1s_val = p1 * p1, p2s_val = p2 * p2;
        double gs    = p1s_val + p2s_val;
        double beta  = std::sqrt(1.0 - gs);
        double alpha = 1.0 / (1.0 + beta);
        // Match Python: (mu_norm / nu_norm**2) ** (1/3)  with mu_norm=1
        double a     = std::pow(1.0 / (nu_norm * nu_norm), ONE_THIRD);

        double X = a * (alpha * p1 * p2 * sinK + (1.0 - alpha * p1s_val) * cosK - p2);
        double Y = a * (alpha * p1 * p2 * cosK + (1.0 - alpha * p2s_val) * sinK - p1);

        // Normalised position
        double rvx = X * eXx + Y * eYx;
        double rvy = X * eXy + Y * eYy;
        double rvz = X * eXz + Y * eYz;

        double z = rvz;
        double r = a * (1.0 - p1 * sinK - p2 * cosK);
        double inv_r = 1.0 / r;
        // sqrt(mu_norm * a) / r * (...), mu_norm=1
        double rp_val = std::sqrt(a) * inv_r * (p2 * sinK - p1 * cosK);

        double cosL = X * inv_r;
        double sinL = Y * inv_r;
        // (mu_norm**2 / nu_norm) ** (1/3) * beta, mu_norm=1
        double c    = std::pow(1.0 / nu_norm, ONE_THIRD) * beta;

        // J2 potential at this position
        double zg  = z * inv_r;
        double r3  = r * r * r;
        double U   = -A / r3 * (1.0 - 3.0 * zg * zg);
        double h   = std::sqrt(c * c - 2.0 * r * r * U);
        double h_over_r = h * inv_r;

        // Normalised velocity
        double Xp = rp_val * cosL - h_over_r * sinL;
        double Yp = rp_val * sinL + h_over_r * cosL;

        double rpvx = Xp * eXx + Yp * eYx;
        double rpvy = Xp * eXy + Yp * eYy;
        double rpvz = Xp * eXz + Yp * eYz;

        // Denormalise to SI
        rv[0] = rvx * L;
        rv[1] = rvy * L;
        rv[2] = rvz * L;

        rpv[0] = rpvx * L_over_T;
        rpv[1] = rpvy * L_over_T;
        rpv[2] = rpvz * L_over_T;
    }
}

} // namespace geqoe
} // namespace astrodyn_core
