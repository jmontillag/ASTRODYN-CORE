#include "jacobians.hpp"
#include "kepler_solver.hpp"

#include <cmath>
#include <vector>

// Helper: write into a 6x6 row-major Jacobian at (row, col)
#define JAC(row, col) jac[(row) * 6 + (col)]

namespace astrodyn_core {
namespace geqoe {

// ============================================================================
// get_pEqpY  --  d(Eq)/d(Y)  Cartesian -> GEqOE Jacobian
// ============================================================================
void get_pEqpY(
    const double* y_in,
    double* jac_out,
    size_t N,
    double J2, double Re, double mu
) {
    const double L     = Re;
    const double T     = std::sqrt(Re * Re * Re / mu);
    const double inv_L = 1.0 / L;
    const double T_over_L = T * inv_L;
    const double A     = J2 / 2.0;
    // Scaling factors for Jacobian assembly
    const double inv_LT = 1.0 / (L * T);
    const double T_over_L_jac = T / L;  // for q/p/Lr velocity partials

    for (size_t idx = 0; idx < N; ++idx) {
        const double* y = y_in + idx * 6;
        double* jac     = jac_out + idx * 36;

        // Zero the Jacobian
        for (int k = 0; k < 36; ++k) jac[k] = 0.0;

        // --- Normalised position and velocity ---
        double rx = y[0] * inv_L, ry = y[1] * inv_L, rz = y[2] * inv_L;
        double vx = y[3] * T_over_L, vy = y[4] * T_over_L, vz = y[5] * T_over_L;

        double r2 = rx*rx + ry*ry + rz*rz;
        double r  = std::sqrt(r2);
        double inv_r = 1.0 / r;
        double rp = (rx*vx + ry*vy + rz*vz) * inv_r;
        double v2 = vx*vx + vy*vy + vz*vz;
        double r3 = r * r2;

        double zg = rz * inv_r;
        double U = -A / r3 * (1.0 - 3.0 * zg * zg);

        // er, ez, ex, ey
        double erx = rx*inv_r, ery = ry*inv_r, erz = rz*inv_r;

        // hv = rv x rpv
        double hx = ry*vz - rz*vy;
        double hy = rz*vx - rx*vz;
        double hz = rx*vy - ry*vx;
        double h2 = hx*hx + hy*hy + hz*hz;
        double h  = std::sqrt(h2);
        double inv_h = 1.0 / h;
        double Ueff = h2 / (2.0 * r2) + U;

        double ehx = hx*inv_h, ehy = hy*inv_h, ehz = hz*inv_h;

        // ef = eh x er
        double efx = ehy*erz - ehz*ery;
        double efy = ehz*erx - ehx*erz;
        double efz = ehx*ery - ehy*erx;

        double E_val = v2 / 2.0 - 1.0 / r + U;
        double neg2E = -2.0 * E_val;
        double nu_norm = std::pow(neg2E, 1.5);

        double ehez = ehz;
        double q1 =  ehx / (1.0 + ehez);
        double q2 = -ehy / (1.0 + ehez);

        double q1s = q1*q1, q2s = q2*q2, q1q2 = q1*q2;
        double eTerm = 1.0 / (1.0 + q1s + q2s);

        double eXx = eTerm*(1.0-q1s+q2s), eXy = eTerm*2.0*q1q2,         eXz = eTerm*(-2.0*q1);
        double eYx = eTerm*2.0*q1q2,      eYy = eTerm*(1.0+q1s-q2s),    eYz = eTerm*2.0*q2;

        double c = std::sqrt(2.0 * r2 * Ueff);
        double c_over_r = c * inv_r;

        // gv = rp*er + (c/r)*ef
        double gvx = rp*erx + c_over_r*efx;
        double gvy = rp*ery + c_over_r*efy;
        double gvz = rp*erz + c_over_r*efz;

        // g = cross(gv, cross(rv, gv)) / mu - er  (mu=1)
        double c1x = ry*gvz - rz*gvy;
        double c1y = rz*gvx - rx*gvz;
        double c1z = rx*gvy - ry*gvx;
        double gx = (gvy*c1z - gvz*c1y) - erx;
        double gy = (gvz*c1x - gvx*c1z) - ery;
        double gz = (gvx*c1y - gvy*c1x) - erz;

        double p1 = gx*eYx + gy*eYy + gz*eYz;
        double p2 = gx*eXx + gy*eXy + gz*eXz;
        double p1s = p1*p1, p2s = p2*p2;
        double gs = p1s + p2s;

        double X = rx*eXx + ry*eXy + rz*eXz;
        double Y_val = rx*eYx + ry*eYy + rz*eYz;

        double beta = std::sqrt(1.0 - gs);
        double alpha = 1.0 / (1.0 + beta);
        double a = std::pow(1.0 / (nu_norm * nu_norm), 1.0 / 3.0);

        double Xp = vx*eXx + vy*eXy + vz*eXz;
        double Yp = vx*eYx + vy*eYy + vz*eYz;

        double cosL = X * inv_r;
        double sinL = Y_val * inv_r;
        double rou = a * (1.0 - gs);

        // Derivative terms
        double Lambda   = Y_val*q2 - X*q1;
        double LAMBDA_p = Yp*q2 - Xp*q1;
        double epsi     = p2 + cosL;
        double zeta     = p1 + sinL;
        double gamma_   = 1.0 + q1s + q2s;

        // pU_r = -A / r^4 * (-6*zg*(ez - zg*er) - 3*(1-3*zg^2)*er)
        // For each component j:
        double ez_arr[3] = {0.0, 0.0, 1.0};
        double er_arr[3] = {erx, ery, erz};
        double pU_r[3];
        {
            double coeff = -A / (r * r * r * r);
            double t1_coeff = -6.0 * zg;
            double t2_coeff = -3.0 * (1.0 - 3.0 * zg * zg);
            for (int j = 0; j < 3; ++j) {
                pU_r[j] = coeff * (t1_coeff * (ez_arr[j] - zg * er_arr[j])
                                 + t2_coeff * er_arr[j]);
            }
        }

        // delta vectors (3-component each)
        double sqrt_mu_a = std::sqrt(a);   // sqrt(mu*a), mu=1
        double delta_0[3], delta_1[3], delta_2[3], delta_3[3];

        // delta_0 = -3 / sqrt(mu*a) * pU_r
        {
            double c0 = -3.0 / sqrt_mu_a;
            for (int j = 0; j < 3; ++j) delta_0[j] = c0 * pU_r[j];
        }

        // delta_1 = (r/(rou*mu) * (r*zeta + rou*sinL)) * pU_r
        {
            double s = r / rou * (r * zeta + rou * sinL);
            for (int j = 0; j < 3; ++j) delta_1[j] = s * pU_r[j];
        }

        // delta_2 = (r/(rou*mu) * (r*epsi + rou*cosL)) * pU_r
        {
            double s = r / rou * (r * epsi + rou * cosL);
            for (int j = 0; j < 3; ++j) delta_2[j] = s * pU_r[j];
        }

        // delta_3 = (r*rp/(c*mu) * (rou+r) * alpha) * pU_r
        {
            double s = r * rp / c * (rou + r) * alpha;
            for (int j = 0; j < 3; ++j) delta_3[j] = s * pU_r[j];
        }

        // --- Position derivatives (3-vectors) ---
        double ef_arr[3] = {efx, efy, efz};
        double eh_arr[3] = {ehx, ehy, ehz};

        // pnu_r = -3*a*nu_norm/r2 * er + delta_0
        double pnu_r[3];
        {
            double c0 = -3.0 * a * nu_norm / r2;
            for (int j = 0; j < 3; ++j) pnu_r[j] = c0 * er_arr[j] + delta_0[j];
        }

        // pp1_r = (zeta/r)*er - (h/(c*r)*((2-c/h)*p2+X/a))*ef
        //       - (p2*LAMBDA_p/h)*eh + delta_1
        double pp1_r[3];
        {
            double c1 = zeta / r;
            double c2 = h / (c * r) * ((2.0 - c / h) * p2 + X / a);
            double c3 = p2 * LAMBDA_p * inv_h;
            for (int j = 0; j < 3; ++j)
                pp1_r[j] = c1*er_arr[j] - c2*ef_arr[j] - c3*eh_arr[j] + delta_1[j];
        }

        // pp2_r = (epsi/r)*er + (h/(c*r)*((2-c/h)*p1+Y/a))*ef
        //       + (p1*LAMBDA_p/h)*eh + delta_2
        double pp2_r[3];
        {
            double c1 = epsi / r;
            double c2 = h / (c * r) * ((2.0 - c / h) * p1 + Y_val / a);
            double c3 = p1 * LAMBDA_p * inv_h;
            for (int j = 0; j < 3; ++j)
                pp2_r[j] = c1*er_arr[j] + c2*ef_arr[j] + c3*eh_arr[j] + delta_2[j];
        }

        // pLr_r
        double pLr_r[3];
        {
            double c1 = rp / (c * r) * (rou * alpha - r * beta);
            double c2 = h / (c * r) * (2.0 - c / h + alpha / a * (rou - r));
            double c3 = LAMBDA_p * inv_h;
            for (int j = 0; j < 3; ++j)
                pLr_r[j] = c1*er_arr[j] - c2*ef_arr[j] - c3*eh_arr[j] + delta_3[j];
        }

        // pq1_r = (-gamma_*Yp/(2*h)) * eh
        double pq1_r[3];
        {
            double s = -gamma_ * Yp / (2.0 * h);
            for (int j = 0; j < 3; ++j) pq1_r[j] = s * eh_arr[j];
        }

        // pq2_r = (-gamma_*Xp/(2*h)) * eh
        double pq2_r[3];
        {
            double s = -gamma_ * Xp / (2.0 * h);
            for (int j = 0; j < 3; ++j) pq2_r[j] = s * eh_arr[j];
        }

        // --- Velocity derivatives (3-vectors) ---
        double rpv_arr[3] = {vx, vy, vz};

        // pnu_rp = (-3/sqrt(mu*a)) * rpv
        double pnu_rp[3];
        {
            double s = -3.0 / sqrt_mu_a;
            for (int j = 0; j < 3; ++j) pnu_rp[j] = s * rpv_arr[j];
        }

        // pp1_rp = (-c/mu*cosL)*er + (h/mu*(2*sinL - rp/c*X))*ef + (Lambda*p2/h)*eh
        double pp1_rp[3];
        {
            double c1 = -c * cosL;       // mu=1
            double c2 = h * (2.0*sinL - rp/c * X);
            double c3 = Lambda * p2 * inv_h;
            for (int j = 0; j < 3; ++j)
                pp1_rp[j] = c1*er_arr[j] + c2*ef_arr[j] + c3*eh_arr[j];
        }

        // pp2_rp = -(-c/mu*sinL)*er + (h/mu*(2*cosL+rp/c*Y))*ef - (Lambda*p1/h)*eh
        double pp2_rp[3];
        {
            double c1 = c * sinL;         // negated from Python: -pp2_rp_t1
            double c2 = h * (2.0*cosL + rp/c * Y_val);
            double c3 = Lambda * p1 * inv_h;
            for (int j = 0; j < 3; ++j)
                pp2_rp[j] = c1*er_arr[j] + c2*ef_arr[j] - c3*eh_arr[j];
        }

        // pLr_rp
        double pLr_rp[3];
        {
            double c1 = c / r * alpha * (r - rou) - 2.0 * r / sqrt_mu_a;
            double c2 = h * rp / c * alpha * (rou + r);
            double c3 = Lambda * inv_h;
            for (int j = 0; j < 3; ++j)
                pLr_rp[j] = c1*er_arr[j] + c2*ef_arr[j] + c3*eh_arr[j];
        }

        // pq1_rp = (gamma_*Y_val/(2*h)) * eh
        double pq1_rp[3];
        {
            double s = gamma_ * Y_val / (2.0 * h);
            for (int j = 0; j < 3; ++j) pq1_rp[j] = s * eh_arr[j];
        }

        // pq2_rp = (gamma_*X/(2*h)) * eh
        double pq2_rp[3];
        {
            double s = gamma_ * X / (2.0 * h);
            for (int j = 0; j < 3; ++j) pq2_rp[j] = s * eh_arr[j];
        }

        // --- Assemble Jacobian (6x6) ---
        // Row 0: nu  derivatives
        for (int j = 0; j < 3; ++j) JAC(0, j)     = pnu_r[j]  * inv_LT;
        for (int j = 0; j < 3; ++j) JAC(0, 3 + j)  = pnu_rp[j] * inv_L;

        // Row 1: q1
        for (int j = 0; j < 3; ++j) JAC(1, j)     = pq1_r[j]  * inv_L;
        for (int j = 0; j < 3; ++j) JAC(1, 3 + j)  = pq1_rp[j] * T_over_L_jac;

        // Row 2: q2
        for (int j = 0; j < 3; ++j) JAC(2, j)     = pq2_r[j]  * inv_L;
        for (int j = 0; j < 3; ++j) JAC(2, 3 + j)  = pq2_rp[j] * T_over_L_jac;

        // Row 3: p1
        for (int j = 0; j < 3; ++j) JAC(3, j)     = pp1_r[j]  * inv_L;
        for (int j = 0; j < 3; ++j) JAC(3, 3 + j)  = pp1_rp[j] * T_over_L_jac;

        // Row 4: p2
        for (int j = 0; j < 3; ++j) JAC(4, j)     = pp2_r[j]  * inv_L;
        for (int j = 0; j < 3; ++j) JAC(4, 3 + j)  = pp2_rp[j] * T_over_L_jac;

        // Row 5: Lr
        for (int j = 0; j < 3; ++j) JAC(5, j)     = pLr_r[j]  * inv_L;
        for (int j = 0; j < 3; ++j) JAC(5, 3 + j)  = pLr_rp[j] * T_over_L_jac;
    }
}

// ============================================================================
// get_pYpEq  --  d(Y)/d(Eq)  GEqOE -> Cartesian Jacobian
// ============================================================================
void get_pYpEq(
    const double* eq_in,
    double* jac_out,
    size_t N,
    double J2, double Re, double mu
) {
    const double L       = Re;
    const double T       = std::sqrt(Re * Re * Re / mu);
    const double inv_T   = 1.0 / T;
    const double A       = J2 / 2.0;
    const double ONE_THIRD = 1.0 / 3.0;

    // Vectorised Kepler solve
    std::vector<double> Lr_vec(N), p1_vec(N), p2_vec(N), K_vec(N);

    for (size_t i = 0; i < N; ++i) {
        const double* eq = eq_in + i * 6;
        double Lr_mod = std::fmod(eq[5], 2.0 * M_PI);
        if (Lr_mod < 0.0) Lr_mod += 2.0 * M_PI;
        Lr_vec[i] = Lr_mod;
        p1_vec[i] = eq[3];
        p2_vec[i] = eq[4];
    }
    solve_kep_gen(Lr_vec.data(), p1_vec.data(), p2_vec.data(), K_vec.data(), N);

    for (size_t idx = 0; idx < N; ++idx) {
        const double* eq = eq_in + idx * 6;
        double* jac      = jac_out + idx * 36;
        for (int k = 0; k < 36; ++k) jac[k] = 0.0;

        double nu_norm = eq[0] * T;
        double q1 = eq[1], q2 = eq[2];
        double p1 = p1_vec[idx], p2 = p2_vec[idx];

        double K    = K_vec[idx];
        double sinK = std::sin(K);
        double cosK = std::cos(K);

        double q1s = q1*q1, q2s = q2*q2, q1q2 = q1*q2;
        double gamma_ = 1.0 + q1s + q2s;
        double eTerm  = 1.0 / gamma_;

        double eXx = eTerm*(1.0-q1s+q2s), eXy = eTerm*2.0*q1q2,         eXz = eTerm*(-2.0*q1);
        double eYx = eTerm*2.0*q1q2,      eYy = eTerm*(1.0+q1s-q2s),    eYz = eTerm*2.0*q2;

        double p1s = p1*p1, p2s = p2*p2;
        double gs    = p1s + p2s;
        double beta  = std::sqrt(1.0 - gs);
        double alpha = 1.0 / (1.0 + beta);
        double a     = std::pow(1.0 / (nu_norm * nu_norm), ONE_THIRD);

        double X = a*(alpha*p1*p2*sinK + (1.0-alpha*p1s)*cosK - p2);
        double Y_val = a*(alpha*p1*p2*cosK + (1.0-alpha*p2s)*sinK - p1);

        // rv_norm
        double rvx = X*eXx + Y_val*eYx;
        double rvy = X*eXy + Y_val*eYy;
        double rvz = X*eXz + Y_val*eYz;

        double r = std::sqrt(rvx*rvx + rvy*rvy + rvz*rvz);
        double inv_r = 1.0 / r;
        double sqrt_a = std::sqrt(a);
        double rp = sqrt_a * inv_r * (p2*sinK - p1*cosK);

        double cosL = X * inv_r;
        double sinL = Y_val * inv_r;
        double c    = std::pow(1.0 / nu_norm, ONE_THIRD) * beta;
        double rou  = a * (1.0 - gs);

        double z  = rvz;
        double zg = z * inv_r;
        double r3 = r * r * r;
        double U  = -A / r3 * (1.0 - 3.0 * zg * zg);
        double h  = std::sqrt(c*c - 2.0*r*r*U);
        double inv_h = 1.0 / h;

        double Xp = rp*cosL - h*inv_r*sinL;
        double Yp = rp*sinL + h*inv_r*cosL;

        double rpvx = Xp*eXx + Yp*eYx;
        double rpvy = Xp*eXy + Yp*eYy;
        double rpvz = Xp*eXz + Yp*eYz;

        // er, eh, ef
        double erx = rvx*inv_r, ery = rvy*inv_r, erz = rvz*inv_r;
        double hvx = rvy*rpvz - rvz*rpvy;
        double hvy = rvz*rpvx - rvx*rpvz;
        double hvz = rvx*rpvy - rvy*rpvx;
        double hv_norm = std::sqrt(hvx*hvx + hvy*hvy + hvz*hvz);
        double inv_hv = 1.0 / hv_norm;
        double ehx = hvx*inv_hv, ehy = hvy*inv_hv, ehz = hvz*inv_hv;
        double efx = ehy*erz - ehz*ery;
        double efy = ehz*erx - ehx*erz;
        double efz = ehx*ery - ehy*erx;

        double er_arr[3] = {erx, ery, erz};
        double ef_arr[3] = {efx, efy, efz};
        double eh_arr[3] = {ehx, ehy, ehz};
        double eX_arr[3] = {eXx, eXy, eXz};
        double eY_arr[3] = {eYx, eYy, eYz};

        // v = rp*er + (c/r)*ef
        double vx_ = rp*erx + c*inv_r*efx;
        double vy_ = rp*ery + c*inv_r*efy;
        double vz_ = rp*erz + c*inv_r*efz;
        double v_arr[3] = {vx_, vy_, vz_};

        double epsi = p2 + cosL;
        double zeta = p1 + sinL;

        // s = (h/r)*er - rp*ef
        double sx = h*inv_r*erx - rp*efx;
        double sy = h*inv_r*ery - rp*efy;
        double sz = h*inv_r*erz - rp*efz;
        double s_arr[3] = {sx, sy, sz};

        // ---------- Position derivatives ----------

        // pr_nu = (-2*r/(3*nu_norm)) * er
        double pr_nu[3];
        {
            double coeff = -2.0 * r / (3.0 * nu_norm);
            for (int j = 0; j < 3; ++j) pr_nu[j] = coeff * er_arr[j];
        }

        // pr_p1
        double pr_p1[3];
        {
            double s1 = -(alpha*rp/nu_norm*p2 + a*sinL);
            double s2 = -a*((a*alpha*beta*inv_r + r/rou)*p2 + X/rou + cosL);
            for (int j = 0; j < 3; ++j)
                pr_p1[j] = s1*er_arr[j] + s2*ef_arr[j];
        }

        // pr_p2
        double pr_p2[3];
        {
            double s1 = alpha*rp/nu_norm*p1 - a*cosL;
            double s2 = a*((a*alpha*beta*inv_r + r/rou)*p1 + Y_val/rou + sinL);
            for (int j = 0; j < 3; ++j)
                pr_p2[j] = s1*er_arr[j] + s2*ef_arr[j];
        }

        // pr_Lr = (1/nu_norm) * v
        double pr_Lr[3];
        {
            double coeff = 1.0 / nu_norm;
            for (int j = 0; j < 3; ++j) pr_Lr[j] = coeff * v_arr[j];
        }

        // pr_q1 = (-2/gamma_) * (r*q2*ef + X*eh)
        double pr_q1[3];
        {
            double coeff = -2.0 / gamma_;
            for (int j = 0; j < 3; ++j)
                pr_q1[j] = coeff * (r*q2*ef_arr[j] + X*eh_arr[j]);
        }

        // pr_q2 = (2/gamma_) * (r*q1*ef + Y_val*eh)
        double pr_q2[3];
        {
            double coeff = 2.0 / gamma_;
            for (int j = 0; j < 3; ++j)
                pr_q2[j] = coeff * (r*q1*ef_arr[j] + Y_val*eh_arr[j]);
        }

        // ---------- Perturbation potential derivatives ----------
        double pU_r[3];
        {
            double coeff = 3.0 * A / (r*r*r*r);
            double ez_arr[3] = {0.0, 0.0, 1.0};
            for (int j = 0; j < 3; ++j)
                pU_r[j] = coeff * (2.0*zg*ez_arr[j] + (1.0-5.0*zg*zg)*er_arr[j]);
        }

        // pU_* = dot(pU_r, pr_*)
        auto dot3 = [](const double* a, const double* b) {
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        };
        double pU_nu = dot3(pU_r, pr_nu);
        double pU_p1 = dot3(pU_r, pr_p1);
        double pU_p2 = dot3(pU_r, pr_p2);
        double pU_Lr = dot3(pU_r, pr_Lr);
        double pU_q1 = dot3(pU_r, pr_q1);
        double pU_q2 = dot3(pU_r, pr_q2);

        // ---------- Delta vectors ----------
        // delta_0 = (r/h*(2/(3*nu_norm)*U - pU_nu)) * ef
        double delta_0[3];
        {
            double coeff = r * inv_h * (2.0/(3.0*nu_norm)*U - pU_nu);
            for (int j = 0; j < 3; ++j) delta_0[j] = coeff * ef_arr[j];
        }

        // delta_1
        double delta_1[3];
        {
            // t1 = (a*(a*alpha*beta/r*p2+cosL))*er + (alpha*rp/nu_norm*p2)*ef
            double t1[3];
            double t1_er = a*(a*alpha*beta*inv_r*p2 + cosL);
            double t1_ef = alpha*rp/nu_norm*p2;
            for (int j = 0; j < 3; ++j)
                t1[j] = t1_er*er_arr[j] + t1_ef*ef_arr[j];

            double s1 = (h - c) / (r*r);
            double s2_coeff = 1.0*inv_h*(2.0*(alpha*rp/nu_norm*p2+a*sinL)*U - r*pU_p1);
            for (int j = 0; j < 3; ++j)
                delta_1[j] = s1*t1[j] + s2_coeff*ef_arr[j];
        }

        // delta_2
        double delta_2[3];
        {
            double t2[3];
            double t2_er = a*(a*alpha*beta*inv_r*p1 + sinL);
            double t2_ef = alpha*rp/nu_norm*p1;
            for (int j = 0; j < 3; ++j)
                t2[j] = t2_er*er_arr[j] + t2_ef*ef_arr[j];

            double s1 = (c - h) / (r*r);
            double s2_coeff = -inv_h*(2.0*(alpha*rp/nu_norm*p1-a*cosL)*U + r*pU_p2);
            for (int j = 0; j < 3; ++j)
                delta_2[j] = s1*t2[j] + s2_coeff*ef_arr[j];
        }

        // delta_3
        double delta_3[3];
        {
            // t3 = (1/nu_norm) * ((c/r)*er + rp*ef)
            double t3[3];
            double inv_nu = 1.0 / nu_norm;
            for (int j = 0; j < 3; ++j)
                t3[j] = inv_nu * (c*inv_r*er_arr[j] + rp*ef_arr[j]);

            double s1 = (c - h) / (r*r);
            double s2_coeff = -inv_h*(2.0*rp/nu_norm*U + r*pU_Lr);
            for (int j = 0; j < 3; ++j)
                delta_3[j] = s1*t3[j] + s2_coeff*ef_arr[j];
        }

        // delta_4 = (-r/h * pU_q1) * ef
        double delta_4[3];
        {
            double coeff = -r * inv_h * pU_q1;
            for (int j = 0; j < 3; ++j) delta_4[j] = coeff * ef_arr[j];
        }

        // delta_5 = (-r/h * pU_q2) * ef
        double delta_5[3];
        {
            double coeff = -r * inv_h * pU_q2;
            for (int j = 0; j < 3; ++j) delta_5[j] = coeff * ef_arr[j];
        }

        // ---------- Velocity derivatives ----------
        double rpv_arr[3] = {rpvx, rpvy, rpvz};

        // prp_nu = (1/(3*nu_norm))*rpv + delta_0
        double prp_nu[3];
        {
            double coeff = 1.0 / (3.0 * nu_norm);
            for (int j = 0; j < 3; ++j) prp_nu[j] = coeff*rpv_arr[j] + delta_0[j];
        }

        // prp_p1
        double prp_p1[3];
        {
            double c_er = a*inv_r*alpha*sqrt_a*inv_r*p2;
            double c_ef = -a*inv_r*(1.0*inv_h*p1 + Xp);    // mu/h*p1 + Xp, mu=1
            double c_s  = a/rou*epsi;
            for (int j = 0; j < 3; ++j)
                prp_p1[j] = c_er*er_arr[j] + c_ef*ef_arr[j] + c_s*s_arr[j] + delta_1[j];
        }

        // prp_p2
        double prp_p2[3];
        {
            double c_er = -a*inv_r*alpha*sqrt_a*inv_r*p1;
            double c_ef = -a*inv_r*(inv_h*p2 - Yp);         // mu/h*p2 - Yp, mu=1
            double c_s  = -a/rou*zeta;
            for (int j = 0; j < 3; ++j)
                prp_p2[j] = c_er*er_arr[j] + c_ef*ef_arr[j] + c_s*s_arr[j] + delta_2[j];
        }

        // prp_Lr = (-mu/(r^2*nu_norm))*er + delta_3   (mu=1)
        double prp_Lr[3];
        {
            double coeff = -1.0 / (r*r*nu_norm);
            for (int j = 0; j < 3; ++j) prp_Lr[j] = coeff*er_arr[j] + delta_3[j];
        }

        // prp_q1 = (2/gamma_)*(q2*s - Xp*eh) + delta_4
        double prp_q1[3];
        {
            double coeff = 2.0 / gamma_;
            for (int j = 0; j < 3; ++j)
                prp_q1[j] = coeff*(q2*s_arr[j] - Xp*eh_arr[j]) + delta_4[j];
        }

        // prp_q2 = (-2/gamma_)*(q1*s - Yp*eh) + delta_5
        double prp_q2[3];
        {
            double coeff = -2.0 / gamma_;
            for (int j = 0; j < 3; ++j)
                prp_q2[j] = coeff*(q1*s_arr[j] - Yp*eh_arr[j]) + delta_5[j];
        }

        // --- Assemble Jacobian ---
        // Python layout: pYpEq[:, :3, col] for positions, pYpEq[:, 3:, col] for velocities
        // Column 0: nu
        for (int j = 0; j < 3; ++j) JAC(j,     0) = pr_nu[j]  * L * T;
        for (int j = 0; j < 3; ++j) JAC(3 + j, 0) = prp_nu[j] * L;

        // Column 1: q1
        for (int j = 0; j < 3; ++j) JAC(j,     1) = pr_q1[j]  * L;
        for (int j = 0; j < 3; ++j) JAC(3 + j, 1) = prp_q1[j] * L * inv_T;

        // Column 2: q2
        for (int j = 0; j < 3; ++j) JAC(j,     2) = pr_q2[j]  * L;
        for (int j = 0; j < 3; ++j) JAC(3 + j, 2) = prp_q2[j] * L * inv_T;

        // Column 3: p1
        for (int j = 0; j < 3; ++j) JAC(j,     3) = pr_p1[j]  * L;
        for (int j = 0; j < 3; ++j) JAC(3 + j, 3) = prp_p1[j] * L * inv_T;

        // Column 4: p2
        for (int j = 0; j < 3; ++j) JAC(j,     4) = pr_p2[j]  * L;
        for (int j = 0; j < 3; ++j) JAC(3 + j, 4) = prp_p2[j] * L * inv_T;

        // Column 5: Lr
        for (int j = 0; j < 3; ++j) JAC(j,     5) = pr_Lr[j]  * L;
        for (int j = 0; j < 3; ++j) JAC(3 + j, 5) = prp_Lr[j] * L * inv_T;
    }
}

#undef JAC

} // namespace geqoe
} // namespace astrodyn_core
