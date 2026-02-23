#include "taylor_order_1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "kepler_solver.hpp"
#include "math_utils.hpp"

namespace astrodyn_core {
namespace geqoe {

namespace {

constexpr std::size_t STATE_DIM = 6;

inline std::size_t idx_yprop(std::size_t step, std::size_t comp) {
    return step * STATE_DIM + comp;
}

inline double derivatives_of_inverse_1(double a) {
    double out = 0.0;
    const double in[1] = {a};
    astrodyn_core::math::compute_derivatives_of_inverse(in, &out, 1, true);
    return out;
}

inline double derivatives_of_inverse_wrt_param_1(double a, double a_d) {
    double out = 0.0;
    const double in[1] = {a};
    const double in_d[1] = {a_d};
    astrodyn_core::math::compute_derivatives_of_inverse_wrt_param(in, in_d, &out, 1, true);
    return out;
}

inline double derivatives_of_inverse_wrt_param_2(double a, double ap, double a_d, double ap_d) {
    double out = 0.0;
    const double in[2] = {a, ap};
    const double in_d[2] = {a_d, ap_d};
    astrodyn_core::math::compute_derivatives_of_inverse_wrt_param(in, in_d, &out, 2, true);
    return out;
}

inline double derivatives_of_product_wrt_param_1(double a, double ap, double a_d, double ap_d) {
    double out = 0.0;
    const double in[2] = {a, ap};
    const double in_d[2] = {a_d, ap_d};
    astrodyn_core::math::compute_derivatives_of_product_wrt_param(in, in_d, &out, 2, true);
    return out;
}

inline double wrap_to_2pi(double angle) {
    const double two_pi = 2.0 * M_PI;
    double wrapped = std::fmod(angle, two_pi);
    if (wrapped < 0.0) {
        wrapped += two_pi;
    }
    return wrapped;
}

} // namespace

void Order1EvaluationScratch::resize(std::size_t M) {
    nu_nu.assign(M, 0.0);

    q1_nu.assign(M, 0.0);
    q1_q1.assign(M, 0.0);
    q1_q2.assign(M, 0.0);
    q1_p1.assign(M, 0.0);
    q1_p2.assign(M, 0.0);
    q1_Lr.assign(M, 0.0);

    q2_nu.assign(M, 0.0);
    q2_q1.assign(M, 0.0);
    q2_q2.assign(M, 0.0);
    q2_p1.assign(M, 0.0);
    q2_p2.assign(M, 0.0);
    q2_Lr.assign(M, 0.0);

    p1_nu.assign(M, 0.0);
    p1_q1.assign(M, 0.0);
    p1_q2.assign(M, 0.0);
    p1_p1.assign(M, 0.0);
    p1_p2.assign(M, 0.0);
    p1_Lr.assign(M, 0.0);

    p2_nu.assign(M, 0.0);
    p2_q1.assign(M, 0.0);
    p2_q2.assign(M, 0.0);
    p2_p1.assign(M, 0.0);
    p2_p2.assign(M, 0.0);
    p2_Lr.assign(M, 0.0);

    Lr_nu.assign(M, 0.0);
    Lr_q1.assign(M, 0.0);
    Lr_q2.assign(M, 0.0);
    Lr_p1.assign(M, 0.0);
    Lr_p2.assign(M, 0.0);
    Lr_Lr.assign(M, 0.0);
}

StmAccumulatorView Order1EvaluationScratch::view() const {
    return StmAccumulatorView{
        nu_nu.data(),

        q1_nu.data(),
        q1_q1.data(),
        q1_q2.data(),
        q1_p1.data(),
        q1_p2.data(),
        q1_Lr.data(),

        q2_nu.data(),
        q2_q1.data(),
        q2_q2.data(),
        q2_p1.data(),
        q2_p2.data(),
        q2_Lr.data(),

        p1_nu.data(),
        p1_q1.data(),
        p1_q2.data(),
        p1_p1.data(),
        p1_p2.data(),
        p1_Lr.data(),

        p2_nu.data(),
        p2_q1.data(),
        p2_q2.data(),
        p2_p1.data(),
        p2_p2.data(),
        p2_Lr.data(),

        Lr_nu.data(),
        Lr_q1.data(),
        Lr_q2.data(),
        Lr_p1.data(),
        Lr_p2.data(),
        Lr_Lr.data(),
    };
}

void compute_coefficients_1(
    const double* y0,
    const PropagationConstants& constants,
    Order1Coefficients& out
) {
    const double T = constants.time_scale;
    const double mu_norm = constants.mu_norm;
    const double A = constants.a_half_j2;

    const double nu_0 = y0[0] * T;
    const double q1_0 = y0[1];
    const double q2_0 = y0[2];
    const double p1_0 = y0[3];
    const double p2_0 = y0[4];
    const double Lr_0 = wrap_to_2pi(y0[5]);

    std::array<double, 1> Lr_vec = {Lr_0};
    std::array<double, 1> p1_vec = {p1_0};
    std::array<double, 1> p2_vec = {p2_0};
    std::array<double, 1> K_vec = {0.0};
    solve_kep_gen(Lr_vec.data(), p1_vec.data(), p2_vec.data(), K_vec.data(), 1, 1e-14, 1000);
    const double K_0 = K_vec[0];

    const double q1s = q1_0 * q1_0;
    const double q2s = q2_0 * q2_0;
    const double p1s = p1_0 * p1_0;
    const double p2s = p2_0 * p2_0;
    const double gs = p1s + p2s;
    const double beta = std::sqrt(1.0 - gs);
    const double alpha = 1.0 / (1.0 + beta);
    const double sinK = std::sin(K_0);
    const double cosK = std::cos(K_0);

    const double a = std::pow(mu_norm / (nu_0 * nu_0), 1.0 / 3.0);
    const double X = a * (alpha * p1_0 * p2_0 * sinK + (1.0 - alpha * p1s) * cosK - p2_0);
    const double Y = a * (alpha * p1_0 * p2_0 * cosK + (1.0 - alpha * p2s) * sinK - p1_0);

    const double r = a * (1.0 - p1_0 * sinK - p2_0 * cosK);
    const double rp = std::sqrt(mu_norm * a) / r * (p2_0 * sinK - p1_0 * cosK);

    const double cosL = X / r;
    const double sinL = Y / r;

    const double c = std::pow((mu_norm * mu_norm) / nu_0, 1.0 / 3.0) * beta;
    const double zg = 2.0 * (Y * q2_0 - X * q1_0) / (r * (1.0 + q1s + q2s));

    const double U = -A / (r * r * r) * (1.0 - 3.0 * zg * zg);

    const double h = std::sqrt(c * c - 2.0 * r * r * U);

    const double r2 = r * r;
    const double r3 = r2 * r;

    const double fir = derivatives_of_inverse_1(r);
    const double fir3 = derivatives_of_inverse_1(r3);
    const double f2rp = r * rp;
    const double fir2 = derivatives_of_inverse_1(r2);

    const double fib = 1.0 / beta;
    const double fic = 1.0 / c;
    const double fialpha = 1.0 / alpha;

    const double hr3 = h * r3;
    const double delta = 1.0 - q1s - q2s;
    const double I = 3.0 * A / hr3 * zg * delta;
    const double d = (h - c) / r2;
    const double wh = I * zg;
    const double xi1 = X / a + 2.0 * p2_0;
    const double xi2 = Y / a + 2.0 * p1_0;
    const double GAMMA_ = fialpha + alpha * (1.0 - r / a);

    const double p1p_0 = p2_0 * (d - wh) - fic * xi1 * U;
    const double p2p_0 = p1_0 * (wh - d) + fic * xi2 * U;
    const double Lrp_0 = nu_0 + d - wh - fic * GAMMA_ * U;
    const double q1p_0 = -I * sinL;
    const double q2p_0 = -I * cosL;

    const double a_nu = std::pow(mu_norm, 1.0 / 3.0) * (-2.0 / 3.0) / std::pow(nu_0, 5.0 / 3.0);

    const double beta_nu = 0.0;
    const double beta_q1 = 0.0;
    const double beta_q2 = 0.0;
    const double beta_Lr = 0.0;
    const double beta_p1 = -p1_0 * fib;
    const double beta_p2 = -p2_0 * fib;

    const double c_q1 = 0.0;
    const double c_q2 = 0.0;
    const double c_Lr = 0.0;
    const double c_nu = -beta * std::pow(mu_norm, 2.0 / 3.0) / (3.0 * std::pow(nu_0, 4.0 / 3.0));
    const double c_p1 = std::pow(mu_norm, 2.0 / 3.0) / std::pow(nu_0, 1.0 / 3.0) * beta_p1;
    const double c_p2 = std::pow(mu_norm, 2.0 / 3.0) / std::pow(nu_0, 1.0 / 3.0) * beta_p2;

    const double fic_nu = derivatives_of_inverse_wrt_param_1(c, c_nu);
    const double fic_p1 = derivatives_of_inverse_wrt_param_1(c, c_p1);
    const double fic_p2 = derivatives_of_inverse_wrt_param_1(c, c_p2);
    const double fic_q1 = 0.0;
    const double fic_q2 = 0.0;
    const double fic_Lr = 0.0;

    const double K_Lr = 1.0 / (1.0 - p1_0 * sinK - p2_0 * cosK);
    const double K_p1 = cosK / (-1.0 + p1_0 * sinK + p2_0 * cosK);
    const double K_p2 = sinK / (1.0 - p1_0 * sinK - p2_0 * cosK);

    const double r_q1 = 0.0;
    const double r_q2 = 0.0;
    const double r_nu = a_nu * (1.0 - p1_0 * sinK - p2_0 * cosK);
    const double r_Lr = a * (-p1_0 * cosK * K_Lr + p2_0 * sinK * K_Lr);
    const double r_p1 = a * (-sinK - p1_0 * cosK * K_p1 + p2_0 * sinK * K_p1);
    const double r_p2 = a * (-p1_0 * cosK * K_p2 - cosK + p2_0 * sinK * K_p2);

    const double alpha_nu = 0.0;
    const double alpha_q1 = 0.0;
    const double alpha_q2 = 0.0;
    const double alpha_Lr = 0.0;
    const double alpha_p1 = -beta_p1 / ((1.0 + beta) * (1.0 + beta));
    const double alpha_p2 = -beta_p2 / ((1.0 + beta) * (1.0 + beta));

    const double fialpha_p1 = derivatives_of_inverse_wrt_param_1(alpha, alpha_p1);
    const double fialpha_p2 = derivatives_of_inverse_wrt_param_1(alpha, alpha_p2);
    const double fialpha_nu = 0.0;
    const double fialpha_q1 = 0.0;
    const double fialpha_q2 = 0.0;
    const double fialpha_Lr = 0.0;

    const double X_q1 = 0.0;
    const double X_q2 = 0.0;
    const double X_nu = a_nu * (alpha * p1_0 * p2_0 * sinK + (1.0 - alpha * p1s) * cosK - p2_0);
    const double X_p1 = a * (
        alpha_p1 * p1_0 * p2_0 * sinK
        + alpha * (p2_0 * sinK + p1_0 * p2_0 * cosK * K_p1)
        - (alpha_p1 * p1s + alpha * 2.0 * p1_0) * cosK
        - (1.0 - alpha * p1s) * sinK * K_p1
    );
    const double X_p2 = a * (
        alpha_p2 * p1_0 * p2_0 * sinK
        + alpha * (p1_0 * sinK + p1_0 * p2_0 * cosK * K_p2)
        - alpha_p2 * p1s * cosK
        - (1.0 - alpha * p1s) * sinK * K_p2
        - 1.0
    );
    const double X_Lr = a * (alpha * p1_0 * p2_0 * cosK * K_Lr - (1.0 - alpha * p1s) * sinK * K_Lr);

    const double Y_q1 = 0.0;
    const double Y_q2 = 0.0;
    const double Y_nu = a_nu * (alpha * p1_0 * p2_0 * cosK + (1.0 - alpha * p2s) * sinK - p1_0);
    const double Y_p1 = a * (
        alpha_p1 * p1_0 * p2_0 * cosK
        + alpha * (p2_0 * cosK - p1_0 * p2_0 * sinK * K_p1)
        - alpha_p1 * p2s * sinK
        + (1.0 - alpha * p2s) * cosK * K_p1
        - 1.0
    );
    const double Y_p2 = a * (
        alpha_p2 * p1_0 * p2_0 * cosK
        + alpha * (p1_0 * cosK - p1_0 * p2_0 * sinK * K_p2)
        - (alpha_p2 * p2_0 * p2_0 + alpha * 2.0 * p2_0) * sinK
        + (1.0 - alpha * p2s) * cosK * K_p2
    );
    const double Y_Lr = a * (-alpha * p1_0 * p2_0 * sinK * K_Lr + (1.0 - alpha * p2s) * cosK * K_Lr);

    const double fir_nu = derivatives_of_inverse_wrt_param_1(r, r_nu);
    const double fir_Lr = derivatives_of_inverse_wrt_param_1(r, r_Lr);
    const double fir_p1 = derivatives_of_inverse_wrt_param_1(r, r_p1);
    const double fir_p2 = derivatives_of_inverse_wrt_param_1(r, r_p2);
    const double fir_q1 = 0.0;
    const double fir_q2 = 0.0;

    const double cosL_q1 = 0.0;
    const double cosL_q2 = 0.0;
    const double cosL_nu = X_nu * fir + X * fir_nu;
    const double cosL_p1 = X_p1 * fir + X * fir_p1;
    const double cosL_p2 = X_p2 * fir + X * fir_p2;
    const double cosL_Lr = X_Lr * fir + X * fir_Lr;

    const double sinL_q1 = 0.0;
    const double sinL_q2 = 0.0;
    const double sinL_nu = Y_nu * fir + Y * fir_nu;
    const double sinL_p1 = Y_p1 * fir + Y * fir_p1;
    const double sinL_p2 = Y_p2 * fir + Y * fir_p2;
    const double sinL_Lr = Y_Lr * fir + Y * fir_Lr;

    const double C_nu = 2.0 * (Y_nu * q2_0 - X_nu * q1_0);
    const double C_p1 = 2.0 * (Y_p1 * q2_0 - X_p1 * q1_0);
    const double C_p2 = 2.0 * (Y_p2 * q2_0 - X_p2 * q1_0);
    const double C_Lr = 2.0 * (Y_Lr * q2_0 - X_Lr * q1_0);
    const double C_q1 = -2.0 * X;
    const double C_q2 = 2.0 * Y;

    const double qs = 1.0 + q1s + q2s;

    const double D_nu = r_nu * qs;
    const double D_p1 = r_p1 * qs;
    const double D_p2 = r_p2 * qs;
    const double D_Lr = r_Lr * qs;
    const double D_q1 = r * 2.0 * q1_0;
    const double D_q2 = r * 2.0 * q2_0;

    const double C = 2.0 * (Y * q2_0 - X * q1_0);
    const double D = r * qs;
    const double fiD = 1.0 / D;

    const double fiD_nu = derivatives_of_inverse_wrt_param_1(D, D_nu);
    const double fiD_Lr = derivatives_of_inverse_wrt_param_1(D, D_Lr);
    const double fiD_q1 = derivatives_of_inverse_wrt_param_1(D, D_q1);
    const double fiD_q2 = derivatives_of_inverse_wrt_param_1(D, D_q2);
    const double fiD_p1 = derivatives_of_inverse_wrt_param_1(D, D_p1);
    const double fiD_p2 = derivatives_of_inverse_wrt_param_1(D, D_p2);

    const double zg_nu = C_nu * fiD + C * fiD_nu;
    const double zg_Lr = C_Lr * fiD + C * fiD_Lr;
    const double zg_q1 = C_q1 * fiD + C * fiD_q1;
    const double zg_q2 = C_q2 * fiD + C * fiD_q2;
    const double zg_p1 = C_p1 * fiD + C * fiD_p1;
    const double zg_p2 = C_p2 * fiD + C * fiD_p2;

    const double fUz = 1.0 - 3.0 * zg * zg;
    const double fUz_nu = -6.0 * zg * zg_nu;
    const double fUz_Lr = -6.0 * zg * zg_Lr;
    const double fUz_q1 = -6.0 * zg * zg_q1;
    const double fUz_q2 = -6.0 * zg * zg_q2;
    const double fUz_p1 = -6.0 * zg * zg_p1;
    const double fUz_p2 = -6.0 * zg * zg_p2;

    const double r3_nu = 3.0 * r2 * r_nu;
    const double r3_Lr = 3.0 * r2 * r_Lr;
    const double r3_q1 = 3.0 * r2 * r_q1;
    const double r3_q2 = 3.0 * r2 * r_q2;
    const double r3_p1 = 3.0 * r2 * r_p1;
    const double r3_p2 = 3.0 * r2 * r_p2;

    const double fir3_nu = derivatives_of_inverse_wrt_param_1(r3, r3_nu);
    const double fir3_Lr = derivatives_of_inverse_wrt_param_1(r3, r3_Lr);
    const double fir3_q1 = derivatives_of_inverse_wrt_param_1(r3, r3_q1);
    const double fir3_q2 = derivatives_of_inverse_wrt_param_1(r3, r3_q2);
    const double fir3_p1 = derivatives_of_inverse_wrt_param_1(r3, r3_p1);
    const double fir3_p2 = derivatives_of_inverse_wrt_param_1(r3, r3_p2);

    const double U_nu = -A * (fUz_nu * fir3 + fUz * fir3_nu);
    const double U_Lr = -A * (fUz_Lr * fir3 + fUz * fir3_Lr);
    const double U_q1 = -A * (fUz_q1 * fir3 + fUz * fir3_q1);
    const double U_q2 = -A * (fUz_q2 * fir3 + fUz * fir3_q2);
    const double U_p1 = -A * (fUz_p1 * fir3 + fUz * fir3_p1);
    const double U_p2 = -A * (fUz_p2 * fir3 + fUz * fir3_p2);

    const double h_nu = (c * c_nu - 2.0 * r * r_nu * U - r2 * U_nu) / h;
    const double h_Lr = (c * c_Lr - 2.0 * r * r_Lr * U - r2 * U_Lr) / h;
    const double h_q1 = (c * c_q1 - 2.0 * r * r_q1 * U - r2 * U_q1) / h;
    const double h_q2 = (c * c_q2 - 2.0 * r * r_q2 * U - r2 * U_q2) / h;
    const double h_p1 = (c * c_p1 - 2.0 * r * r_p1 * U - r2 * U_p1) / h;
    const double h_p2 = (c * c_p2 - 2.0 * r * r_p2 * U - r2 * U_p2) / h;

    const double fihr3 = 1.0 / hr3;

    const double hr3_nu = h_nu * r3 + h * r3_nu;
    const double hr3_Lr = h_Lr * r3 + h * r3_Lr;
    const double hr3_q1 = h_q1 * r3 + h * r3_q1;
    const double hr3_q2 = h_q2 * r3 + h * r3_q2;
    const double hr3_p1 = h_p1 * r3 + h * r3_p1;
    const double hr3_p2 = h_p2 * r3 + h * r3_p2;

    const double fihr3_nu = derivatives_of_inverse_wrt_param_1(hr3, hr3_nu);
    const double fihr3_Lr = derivatives_of_inverse_wrt_param_1(hr3, hr3_Lr);
    const double fihr3_q1 = derivatives_of_inverse_wrt_param_1(hr3, hr3_q1);
    const double fihr3_q2 = derivatives_of_inverse_wrt_param_1(hr3, hr3_q2);
    const double fihr3_p1 = derivatives_of_inverse_wrt_param_1(hr3, hr3_p1);
    const double fihr3_p2 = derivatives_of_inverse_wrt_param_1(hr3, hr3_p2);

    const double delta_nu = 0.0;
    const double delta_Lr = 0.0;
    const double delta_p1 = 0.0;
    const double delta_p2 = 0.0;
    const double delta_q1 = -2.0 * q1_0;
    const double delta_q2 = -2.0 * q2_0;

    const double I_nu = 3.0 * A * (zg_nu * delta * fihr3 + zg * delta * fihr3_nu);
    const double I_Lr = 3.0 * A * (zg_Lr * delta * fihr3 + zg * delta * fihr3_Lr);
    const double I_p1 = 3.0 * A * (zg_p1 * delta * fihr3 + zg * delta * fihr3_p1);
    const double I_p2 = 3.0 * A * (zg_p2 * delta * fihr3 + zg * delta * fihr3_p2);
    const double I_q1 = 3.0 * A * ((zg_q1 * delta + zg * delta_q1) * fihr3 + zg * delta * fihr3_q1);
    const double I_q2 = 3.0 * A * ((zg_q2 * delta + zg * delta_q2) * fihr3 + zg * delta * fihr3_q2);

    const double r2_nu = 2.0 * r * r_nu;
    const double r2_Lr = 2.0 * r * r_Lr;
    const double r2_q1 = 2.0 * r * r_q1;
    const double r2_q2 = 2.0 * r * r_q2;
    const double r2_p1 = 2.0 * r * r_p1;
    const double r2_p2 = 2.0 * r * r_p2;

    const double fir2_nu = derivatives_of_inverse_wrt_param_1(r2, r2_nu);
    const double fir2_Lr = derivatives_of_inverse_wrt_param_1(r2, r2_Lr);
    const double fir2_q1 = derivatives_of_inverse_wrt_param_1(r2, r2_q1);
    const double fir2_q2 = derivatives_of_inverse_wrt_param_1(r2, r2_q2);
    const double fir2_p1 = derivatives_of_inverse_wrt_param_1(r2, r2_p1);
    const double fir2_p2 = derivatives_of_inverse_wrt_param_1(r2, r2_p2);

    const double rpn = p2_0 * sinL - p1_0 * cosL;

    const double rpn_nu = p2_0 * sinL_nu - p1_0 * cosL_nu;
    const double rpn_p1 = p2_0 * sinL_p1 - cosL - p1_0 * cosL_p1;
    const double rpn_p2 = sinL + p2_0 * sinL_p2 - p1_0 * cosL_p2;
    const double rpn_Lr = p2_0 * sinL_Lr - p1_0 * cosL_Lr;
    const double rpn_q1 = p2_0 * sinL_q1 - p1_0 * cosL_q1;
    const double rpn_q2 = p2_0 * sinL_q2 - p1_0 * cosL_q2;

    const double rp_nu = mu_norm * (rpn_nu * fic + rpn * fic_nu);
    const double rp_Lr = mu_norm * (rpn_Lr * fic + rpn * fic_Lr);
    const double rp_q1 = mu_norm * (rpn_q1 * fic + rpn * fic_q1);
    const double rp_q2 = mu_norm * (rpn_q2 * fic + rpn * fic_q2);
    const double rp_p1 = mu_norm * (rpn_p1 * fic + rpn * fic_p1);
    const double rp_p2 = mu_norm * (rpn_p2 * fic + rpn * fic_p2);

    const double f2rp_nu = derivatives_of_product_wrt_param_1(r, rp, r_nu, rp_nu);
    const double f2rp_Lr = derivatives_of_product_wrt_param_1(r, rp, r_Lr, rp_Lr);
    const double f2rp_q1 = derivatives_of_product_wrt_param_1(r, rp, r_q1, rp_q1);
    const double f2rp_q2 = derivatives_of_product_wrt_param_1(r, rp, r_q2, rp_q2);
    const double f2rp_p1 = derivatives_of_product_wrt_param_1(r, rp, r_p1, rp_p1);
    const double f2rp_p2 = derivatives_of_product_wrt_param_1(r, rp, r_p2, rp_p2);

    const double firp_nu = derivatives_of_inverse_wrt_param_2(r, rp, r_nu, rp_nu);
    const double firp_Lr = derivatives_of_inverse_wrt_param_2(r, rp, r_Lr, rp_Lr);
    const double firp_q1 = derivatives_of_inverse_wrt_param_2(r, rp, r_q1, rp_q1);
    const double firp_q2 = derivatives_of_inverse_wrt_param_2(r, rp, r_q2, rp_q2);
    const double firp_p1 = derivatives_of_inverse_wrt_param_2(r, rp, r_p1, rp_p1);
    const double firp_p2 = derivatives_of_inverse_wrt_param_2(r, rp, r_p2, rp_p2);

    const double fir2p_nu = derivatives_of_inverse_wrt_param_2(r2, 2.0 * f2rp, r2_nu, 2.0 * f2rp_nu);
    const double fir2p_Lr = derivatives_of_inverse_wrt_param_2(r2, 2.0 * f2rp, r2_Lr, 2.0 * f2rp_Lr);
    const double fir2p_q1 = derivatives_of_inverse_wrt_param_2(r2, 2.0 * f2rp, r2_q1, 2.0 * f2rp_q1);
    const double fir2p_q2 = derivatives_of_inverse_wrt_param_2(r2, 2.0 * f2rp, r2_q2, 2.0 * f2rp_q2);
    const double fir2p_p1 = derivatives_of_inverse_wrt_param_2(r2, 2.0 * f2rp, r2_p1, 2.0 * f2rp_p1);
    const double fir2p_p2 = derivatives_of_inverse_wrt_param_2(r2, 2.0 * f2rp, r2_p2, 2.0 * f2rp_p2);

    const double r3p_nu = 3.0 * (2.0 * r * r_nu * rp + r2 * rp_nu);
    const double r3p_Lr = 3.0 * (2.0 * r * r_Lr * rp + r2 * rp_Lr);
    const double r3p_q1 = 3.0 * (2.0 * r * r_q1 * rp + r2 * rp_q1);
    const double r3p_q2 = 3.0 * (2.0 * r * r_q2 * rp + r2 * rp_q2);
    const double r3p_p1 = 3.0 * (2.0 * r * r_p1 * rp + r2 * rp_p1);
    const double r3p_p2 = 3.0 * (2.0 * r * r_p2 * rp + r2 * rp_p2);

    const double fir3_p_nu = derivatives_of_inverse_wrt_param_2(r3, 3.0 * r2 * rp, r3_nu, r3p_nu);
    const double fir3_p_Lr = derivatives_of_inverse_wrt_param_2(r3, 3.0 * r2 * rp, r3_Lr, r3p_Lr);
    const double fir3_p_q1 = derivatives_of_inverse_wrt_param_2(r3, 3.0 * r2 * rp, r3_q1, r3p_q1);
    const double fir3_p_q2 = derivatives_of_inverse_wrt_param_2(r3, 3.0 * r2 * rp, r3_q2, r3p_q2);
    const double fir3_p_p1 = derivatives_of_inverse_wrt_param_2(r3, 3.0 * r2 * rp, r3_p1, r3p_p1);
    const double fir3_p_p2 = derivatives_of_inverse_wrt_param_2(r3, 3.0 * r2 * rp, r3_p2, r3p_p2);

    const double d_nu = (h_nu - c_nu) * fir2 + (h - c) * fir2_nu;
    const double d_Lr = (h_Lr - c_Lr) * fir2 + (h - c) * fir2_Lr;
    const double d_q1 = (h_q1 - c_q1) * fir2 + (h - c) * fir2_q1;
    const double d_q2 = (h_q2 - c_q2) * fir2 + (h - c) * fir2_q2;
    const double d_p1 = (h_p1 - c_p1) * fir2 + (h - c) * fir2_p1;
    const double d_p2 = (h_p2 - c_p2) * fir2 + (h - c) * fir2_p2;

    const double wh_nu = I_nu * zg + I * zg_nu;
    const double wh_Lr = I_Lr * zg + I * zg_Lr;
    const double wh_q1 = I_q1 * zg + I * zg_q1;
    const double wh_q2 = I_q2 * zg + I * zg_q2;
    const double wh_p1 = I_p1 * zg + I * zg_p1;
    const double wh_p2 = I_p2 * zg + I * zg_p2;

    const double GAMMA_nu = fialpha_nu + alpha_nu * (1.0 - r / a) + alpha * (-r_nu / a + r / (a * a) * a_nu);
    const double GAMMA_Lr = fialpha_Lr + alpha_Lr * (1.0 - r / a) - alpha * r_Lr / a;
    const double GAMMA_q1 = fialpha_q1 + alpha_q1 * (1.0 - r / a) - alpha * r_q1 / a;
    const double GAMMA_q2 = fialpha_q2 + alpha_q2 * (1.0 - r / a) - alpha * r_q2 / a;
    const double GAMMA_p1 = fialpha_p1 + alpha_p1 * (1.0 - r / a) - alpha * r_p1 / a;
    const double GAMMA_p2 = fialpha_p2 + alpha_p2 * (1.0 - r / a) - alpha * r_p2 / a;

    const double xi1_nu = X_nu / a - X / (a * a) * a_nu;
    const double xi2_nu = Y_nu / a - Y / (a * a) * a_nu;
    const double xi1_Lr = X_Lr / a;
    const double xi2_Lr = Y_Lr / a;
    const double xi1_q1 = X_q1 / a;
    const double xi2_q1 = Y_q1 / a;
    const double xi1_q2 = X_q2 / a;
    const double xi2_q2 = Y_q2 / a;
    const double xi1_p1 = X_p1 / a;
    const double xi2_p1 = Y_p1 / a + 2.0;
    const double xi1_p2 = X_p2 / a + 2.0;
    const double xi2_p2 = Y_p2 / a;

    const double q1p_nu = -I_nu * sinL - I * sinL_nu;
    const double q1p_Lr = -I_Lr * sinL - I * sinL_Lr;
    const double q1p_q1 = -I_q1 * sinL - I * sinL_q1;
    const double q1p_q2 = -I_q2 * sinL - I * sinL_q2;
    const double q1p_p1 = -I_p1 * sinL - I * sinL_p1;
    const double q1p_p2 = -I_p2 * sinL - I * sinL_p2;

    const double q2p_nu = -I_nu * cosL - I * cosL_nu;
    const double q2p_Lr = -I_Lr * cosL - I * cosL_Lr;
    const double q2p_q1 = -I_q1 * cosL - I * cosL_q1;
    const double q2p_q2 = -I_q2 * cosL - I * cosL_q2;
    const double q2p_p1 = -I_p1 * cosL - I * cosL_p1;
    const double q2p_p2 = -I_p2 * cosL - I * cosL_p2;

    const double p1p_nu = p2_0 * (d_nu - wh_nu) - (fic_nu * xi1 + fic * xi1_nu) * U - fic * xi1 * U_nu;
    const double p1p_Lr = p2_0 * (d_Lr - wh_Lr) - (fic_Lr * xi1 + fic * xi1_Lr) * U - fic * xi1 * U_Lr;
    const double p1p_q1 = p2_0 * (d_q1 - wh_q1) - (fic_q1 * xi1 + fic * xi1_q1) * U - fic * xi1 * U_q1;
    const double p1p_q2 = p2_0 * (d_q2 - wh_q2) - (fic_q2 * xi1 + fic * xi1_q2) * U - fic * xi1 * U_q2;
    const double p1p_p1 = p2_0 * (d_p1 - wh_p1) - (fic_p1 * xi1 + fic * xi1_p1) * U - fic * xi1 * U_p1;
    const double p1p_p2 = (d - wh) + p2_0 * (d_p2 - wh_p2) - (fic_p2 * xi1 + fic * xi1_p2) * U - fic * xi1 * U_p2;

    const double p2p_nu = p1_0 * (-d_nu + wh_nu) + (fic_nu * xi2 + fic * xi2_nu) * U + fic * xi2 * U_nu;
    const double p2p_Lr = p1_0 * (-d_Lr + wh_Lr) + (fic_Lr * xi2 + fic * xi2_Lr) * U + fic * xi2 * U_Lr;
    const double p2p_q1 = p1_0 * (-d_q1 + wh_q1) + (fic_q1 * xi2 + fic * xi2_q1) * U + fic * xi2 * U_q1;
    const double p2p_q2 = p1_0 * (-d_q2 + wh_q2) + (fic_q2 * xi2 + fic * xi2_q2) * U + fic * xi2 * U_q2;
    const double p2p_p1 = (-d + wh) + p1_0 * (-d_p1 + wh_p1) + (fic_p1 * xi2 + fic * xi2_p1) * U + fic * xi2 * U_p1;
    const double p2p_p2 = p1_0 * (-d_p2 + wh_p2) + (fic_p2 * xi2 + fic * xi2_p2) * U + fic * xi2 * U_p2;

    const double Lrp_nu = 1.0 + d_nu - wh_nu - (fic_nu * GAMMA_ + fic * GAMMA_nu) * U - fic * GAMMA_ * U_nu;
    const double Lrp_Lr = d_Lr - wh_Lr - (fic_Lr * GAMMA_ + fic * GAMMA_Lr) * U - fic * GAMMA_ * U_Lr;
    const double Lrp_q1 = d_q1 - wh_q1 - (fic_q1 * GAMMA_ + fic * GAMMA_q1) * U - fic * GAMMA_ * U_q1;
    const double Lrp_q2 = d_q2 - wh_q2 - (fic_q2 * GAMMA_ + fic * GAMMA_q2) * U - fic * GAMMA_ * U_q2;
    const double Lrp_p1 = d_p1 - wh_p1 - (fic_p1 * GAMMA_ + fic * GAMMA_p1) * U - fic * GAMMA_ * U_p1;
    const double Lrp_p2 = d_p2 - wh_p2 - (fic_p2 * GAMMA_ + fic * GAMMA_p2) * U - fic * GAMMA_ * U_p2;

    out.nu_0 = nu_0;
    out.q1_0 = q1_0;
    out.q2_0 = q2_0;
    out.p1_0 = p1_0;
    out.p2_0 = p2_0;
    out.Lr_0 = Lr_0;

    out.q1p_0 = q1p_0;
    out.q2p_0 = q2p_0;
    out.p1p_0 = p1p_0;
    out.p2p_0 = p2p_0;
    out.Lrp_0 = Lrp_0;

    out.q1p_nu = q1p_nu;
    out.q1p_Lr = q1p_Lr;
    out.q1p_q1 = q1p_q1;
    out.q1p_q2 = q1p_q2;
    out.q1p_p1 = q1p_p1;
    out.q1p_p2 = q1p_p2;

    out.q2p_nu = q2p_nu;
    out.q2p_Lr = q2p_Lr;
    out.q2p_q1 = q2p_q1;
    out.q2p_q2 = q2p_q2;
    out.q2p_p1 = q2p_p1;
    out.q2p_p2 = q2p_p2;

    out.p1p_nu = p1p_nu;
    out.p1p_Lr = p1p_Lr;
    out.p1p_q1 = p1p_q1;
    out.p1p_q2 = p1p_q2;
    out.p1p_p1 = p1p_p1;
    out.p1p_p2 = p1p_p2;

    out.p2p_nu = p2p_nu;
    out.p2p_Lr = p2p_Lr;
    out.p2p_q1 = p2p_q1;
    out.p2p_q2 = p2p_q2;
    out.p2p_p1 = p2p_p1;
    out.p2p_p2 = p2p_p2;

    out.Lrp_nu = Lrp_nu;
    out.Lrp_Lr = Lrp_Lr;
    out.Lrp_q1 = Lrp_q1;
    out.Lrp_q2 = Lrp_q2;
    out.Lrp_p1 = Lrp_p1;
    out.Lrp_p2 = Lrp_p2;

    out.map_components_col0[0] = 0.0;
    out.map_components_col0[1] = q1p_0;
    out.map_components_col0[2] = q2p_0;
    out.map_components_col0[3] = p1p_0;
    out.map_components_col0[4] = p2p_0;
    out.map_components_col0[5] = Lrp_0;

    (void)beta_nu;
    (void)beta_q1;
    (void)beta_q2;
    (void)beta_Lr;
    (void)delta_nu;
    (void)delta_Lr;
    (void)delta_p1;
    (void)delta_p2;
    (void)firp_nu;
    (void)firp_Lr;
    (void)firp_q1;
    (void)firp_q2;
    (void)firp_p1;
    (void)firp_p2;
    (void)fir2p_nu;
    (void)fir2p_Lr;
    (void)fir2p_q1;
    (void)fir2p_q2;
    (void)fir2p_p1;
    (void)fir2p_p2;
    (void)fir3_p_nu;
    (void)fir3_p_Lr;
    (void)fir3_p_q1;
    (void)fir3_p_q2;
    (void)fir3_p_p1;
    (void)fir3_p_p2;
}

void evaluate_order_1(
    const Order1Coefficients& coeffs,
    const double* dt_norm,
    std::size_t M,
    double* y_prop,
    Order1EvaluationScratch& scratch
) {
    scratch.resize(M);

    for (std::size_t i = 0; i < M; ++i) {
        const double dt = dt_norm[i];

        y_prop[idx_yprop(i, 0)] = coeffs.nu_0;
        y_prop[idx_yprop(i, 1)] = coeffs.q1_0 + coeffs.q1p_0 * dt;
        y_prop[idx_yprop(i, 2)] = coeffs.q2_0 + coeffs.q2p_0 * dt;
        y_prop[idx_yprop(i, 3)] = coeffs.p1_0 + coeffs.p1p_0 * dt;
        y_prop[idx_yprop(i, 4)] = coeffs.p2_0 + coeffs.p2p_0 * dt;
        y_prop[idx_yprop(i, 5)] = coeffs.Lr_0 + coeffs.Lrp_0 * dt;

        scratch.nu_nu[i] = 1.0;

        scratch.Lr_nu[i] = coeffs.Lrp_nu * dt;
        scratch.Lr_Lr[i] = 1.0 + coeffs.Lrp_Lr * dt;
        scratch.Lr_q1[i] = coeffs.Lrp_q1 * dt;
        scratch.Lr_q2[i] = coeffs.Lrp_q2 * dt;
        scratch.Lr_p1[i] = coeffs.Lrp_p1 * dt;
        scratch.Lr_p2[i] = coeffs.Lrp_p2 * dt;

        scratch.q1_nu[i] = coeffs.q1p_nu * dt;
        scratch.q1_Lr[i] = coeffs.q1p_Lr * dt;
        scratch.q1_q1[i] = 1.0 + coeffs.q1p_q1 * dt;
        scratch.q1_q2[i] = coeffs.q1p_q2 * dt;
        scratch.q1_p1[i] = coeffs.q1p_p1 * dt;
        scratch.q1_p2[i] = coeffs.q1p_p2 * dt;

        scratch.q2_nu[i] = coeffs.q2p_nu * dt;
        scratch.q2_Lr[i] = coeffs.q2p_Lr * dt;
        scratch.q2_q1[i] = coeffs.q2p_q1 * dt;
        scratch.q2_q2[i] = 1.0 + coeffs.q2p_q2 * dt;
        scratch.q2_p1[i] = coeffs.q2p_p1 * dt;
        scratch.q2_p2[i] = coeffs.q2p_p2 * dt;

        scratch.p1_nu[i] = coeffs.p1p_nu * dt;
        scratch.p1_Lr[i] = coeffs.p1p_Lr * dt;
        scratch.p1_q1[i] = coeffs.p1p_q1 * dt;
        scratch.p1_q2[i] = coeffs.p1p_q2 * dt;
        scratch.p1_p1[i] = 1.0 + coeffs.p1p_p1 * dt;
        scratch.p1_p2[i] = coeffs.p1p_p2 * dt;

        scratch.p2_nu[i] = coeffs.p2p_nu * dt;
        scratch.p2_Lr[i] = coeffs.p2p_Lr * dt;
        scratch.p2_q1[i] = coeffs.p2p_q1 * dt;
        scratch.p2_q2[i] = coeffs.p2p_q2 * dt;
        scratch.p2_p1[i] = coeffs.p2p_p1 * dt;
        scratch.p2_p2[i] = 1.0 + coeffs.p2p_p2 * dt;
    }
}

} // namespace geqoe
} // namespace astrodyn_core
