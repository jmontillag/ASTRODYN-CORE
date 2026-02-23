#include "taylor_pipeline.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "conversions.hpp"
#include "jacobians.hpp"

namespace astrodyn_core {
namespace geqoe {

namespace {

constexpr std::size_t STATE_DIM = 6;

inline std::size_t idx3(std::size_t i, std::size_t j, std::size_t k, std::size_t M) {
    return i * STATE_DIM * M + j * M + k;
}

inline std::size_t idx2(std::size_t i, std::size_t j) {
    return i * STATE_DIM + j;
}

inline std::size_t idx_nij(std::size_t n, std::size_t i, std::size_t j) {
    return n * STATE_DIM * STATE_DIM + i * STATE_DIM + j;
}

void matmul6(const double* A, const double* B, double* C) {
    for (std::size_t i = 0; i < STATE_DIM; ++i) {
        for (std::size_t j = 0; j < STATE_DIM; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < STATE_DIM; ++k) {
                sum += A[idx2(i, k)] * B[idx2(k, j)];
            }
            C[idx2(i, j)] = sum;
        }
    }
}

void extract_eq0_from_rv2geqoe(
    const double* y0_cart,
    double j2,
    double re,
    double mu,
    double* eq0_out
) {
    std::array<double, 6> eq_tmp{};
    rv2geqoe(y0_cart, eq_tmp.data(), 1, j2, re, mu);
    for (std::size_t i = 0; i < STATE_DIM; ++i) {
        eq0_out[i] = eq_tmp[i];
    }
}

} // namespace

PreparedTaylorCoefficients::PreparedTaylorCoefficients() : constants{}, order(1), initial_geqoe{0.0}, map_components{0.0}, order1{} {}

std::shared_ptr<PreparedTaylorCoefficients> prepare_taylor_coefficients_cpp(
    const double* y0,
    double j2,
    double re,
    double mu,
    int order
) {
    if (!is_valid_order(order)) {
        throw std::runtime_error("Taylor order must be in [1, 4].");
    }
    if (order != 1) {
        throw std::runtime_error("C++ staged propagator currently supports order=1 only.");
    }

    auto prepared = std::make_shared<PreparedTaylorCoefficients>();
    prepared->constants = make_constants(j2, re, mu);
    prepared->order = order;

    for (std::size_t i = 0; i < STATE_DIM; ++i) {
        prepared->initial_geqoe[i] = y0[i];
    }

    std::fill(std::begin(prepared->map_components), std::end(prepared->map_components), 0.0);

    compute_coefficients_1(y0, prepared->constants, prepared->order1);
    for (std::size_t row = 0; row < STATE_DIM; ++row) {
        prepared->map_components[row * 4 + 0] = prepared->order1.map_components_col0[row];
    }

    return prepared;
}

void evaluate_taylor_cpp(
    const PreparedTaylorCoefficients& coeffs,
    const double* dt_seconds,
    std::size_t M,
    double* y_prop,
    double* y_y0,
    double* map_components
) {
    if (coeffs.order != 1) {
        throw std::runtime_error("C++ staged propagator currently supports order=1 only.");
    }

    std::vector<double> dt_norm(M, 0.0);
    normalize_time_grid(dt_seconds, M, coeffs.constants.time_scale, dt_norm.data());

    std::fill(y_y0, y_y0 + STATE_DIM * STATE_DIM * M, 0.0);

    Order1EvaluationScratch scratch;
    evaluate_order_1(coeffs.order1, dt_norm.data(), M, y_prop, scratch);

    assemble_stm_and_denormalize_nu(scratch.view(), M, coeffs.constants.time_scale, y_prop, y_y0);

    for (std::size_t row = 0; row < STATE_DIM; ++row) {
        for (int col = 0; col < coeffs.order; ++col) {
            map_components[row * coeffs.order + col] = coeffs.map_components[row * 4 + static_cast<std::size_t>(col)];
        }
    }
}

std::shared_ptr<PreparedTaylorCoefficients> prepare_cart_coefficients_cpp(
    const double* y0_cart,
    double j2,
    double re,
    double mu,
    int order,
    double* peq_py_0
) {
    std::array<double, 6> eq0{};
    extract_eq0_from_rv2geqoe(y0_cart, j2, re, mu, eq0.data());

    get_pEqpY(y0_cart, peq_py_0, 1, j2, re, mu);

    return prepare_taylor_coefficients_cpp(eq0.data(), j2, re, mu, order);
}

void evaluate_cart_taylor_cpp(
    const PreparedTaylorCoefficients& coeffs,
    const double* peq_py_0,
    const double* tspan,
    std::size_t M,
    double* y_out,
    double* dy_dy0
) {
    std::vector<double> eq_taylor(M * STATE_DIM, 0.0);
    std::vector<double> eq_eq0(STATE_DIM * STATE_DIM * M, 0.0);
    std::vector<double> map_components(STATE_DIM * static_cast<std::size_t>(coeffs.order), 0.0);

    evaluate_taylor_cpp(coeffs, tspan, M, eq_taylor.data(), eq_eq0.data(), map_components.data());

    std::vector<double> py_peq(M * STATE_DIM * STATE_DIM, 0.0);
    get_pYpEq(eq_taylor.data(), py_peq.data(), M, coeffs.constants.j2, coeffs.constants.re, coeffs.constants.mu);

    std::vector<double> rv(M * 3, 0.0);
    std::vector<double> rpv(M * 3, 0.0);
    geqoe2rv(eq_taylor.data(), rv.data(), rpv.data(), M, coeffs.constants.j2, coeffs.constants.re, coeffs.constants.mu);

    for (std::size_t k = 0; k < M; ++k) {
        y_out[k * STATE_DIM + 0] = rv[k * 3 + 0];
        y_out[k * STATE_DIM + 1] = rv[k * 3 + 1];
        y_out[k * STATE_DIM + 2] = rv[k * 3 + 2];
        y_out[k * STATE_DIM + 3] = rpv[k * 3 + 0];
        y_out[k * STATE_DIM + 4] = rpv[k * 3 + 1];
        y_out[k * STATE_DIM + 5] = rpv[k * 3 + 2];
    }

    std::array<double, 36> eq_eq0_k{};
    std::array<double, 36> py_peq_k{};
    std::array<double, 36> tmp{};
    std::array<double, 36> out{};

    for (std::size_t k = 0; k < M; ++k) {
        for (std::size_t i = 0; i < STATE_DIM; ++i) {
            for (std::size_t j = 0; j < STATE_DIM; ++j) {
                eq_eq0_k[idx2(i, j)] = eq_eq0[idx3(i, j, k, M)];
                py_peq_k[idx2(i, j)] = py_peq[idx_nij(k, i, j)];
            }
        }

        matmul6(py_peq_k.data(), eq_eq0_k.data(), tmp.data());
        matmul6(tmp.data(), peq_py_0, out.data());

        for (std::size_t i = 0; i < STATE_DIM; ++i) {
            for (std::size_t j = 0; j < STATE_DIM; ++j) {
                dy_dy0[idx3(i, j, k, M)] = out[idx2(i, j)];
            }
        }
    }
}

} // namespace geqoe
} // namespace astrodyn_core
