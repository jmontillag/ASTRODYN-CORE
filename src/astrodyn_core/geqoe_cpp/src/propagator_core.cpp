#include "propagator_core.hpp"

#include <cmath>

namespace astrodyn_core {
namespace geqoe {

namespace {

constexpr std::size_t STATE_DIM = 6;

inline std::size_t idx_yprop(std::size_t step, std::size_t comp) {
    return step * STATE_DIM + comp;
}

inline std::size_t idx_yy0(std::size_t row, std::size_t col, std::size_t step, std::size_t M) {
    return row * STATE_DIM * M + col * M + step;
}

} // namespace

bool is_valid_order(int order) {
    return order >= 1 && order <= 4;
}

PropagationConstants make_constants(double j2, double re, double mu) {
    const double time_scale = std::sqrt(re * re * re / mu);
    return PropagationConstants{
        j2,
        re,
        mu,
        re,
        time_scale,
        1.0,
        j2 / 2.0,
    };
}

void normalize_time_grid(
    const double* dt_seconds,
    std::size_t M,
    double time_scale,
    double* dt_norm_out
) {
    for (std::size_t i = 0; i < M; ++i) {
        dt_norm_out[i] = dt_seconds[i] / time_scale;
    }
}

void assemble_stm_and_denormalize_nu(
    const StmAccumulatorView& s,
    std::size_t M,
    double time_scale,
    double* y_prop,
    double* y_y0
) {
    for (std::size_t k = 0; k < M; ++k) {
        y_y0[idx_yy0(0, 0, k, M)] = s.nu_nu[k];

        y_y0[idx_yy0(1, 0, k, M)] = s.q1_nu[k] * time_scale;
        y_y0[idx_yy0(1, 1, k, M)] = s.q1_q1[k];
        y_y0[idx_yy0(1, 2, k, M)] = s.q1_q2[k];
        y_y0[idx_yy0(1, 3, k, M)] = s.q1_p1[k];
        y_y0[idx_yy0(1, 4, k, M)] = s.q1_p2[k];
        y_y0[idx_yy0(1, 5, k, M)] = s.q1_Lr[k];

        y_y0[idx_yy0(2, 0, k, M)] = s.q2_nu[k] * time_scale;
        y_y0[idx_yy0(2, 1, k, M)] = s.q2_q1[k];
        y_y0[idx_yy0(2, 2, k, M)] = s.q2_q2[k];
        y_y0[idx_yy0(2, 3, k, M)] = s.q2_p1[k];
        y_y0[idx_yy0(2, 4, k, M)] = s.q2_p2[k];
        y_y0[idx_yy0(2, 5, k, M)] = s.q2_Lr[k];

        y_y0[idx_yy0(3, 0, k, M)] = s.p1_nu[k] * time_scale;
        y_y0[idx_yy0(3, 1, k, M)] = s.p1_q1[k];
        y_y0[idx_yy0(3, 2, k, M)] = s.p1_q2[k];
        y_y0[idx_yy0(3, 3, k, M)] = s.p1_p1[k];
        y_y0[idx_yy0(3, 4, k, M)] = s.p1_p2[k];
        y_y0[idx_yy0(3, 5, k, M)] = s.p1_Lr[k];

        y_y0[idx_yy0(4, 0, k, M)] = s.p2_nu[k] * time_scale;
        y_y0[idx_yy0(4, 1, k, M)] = s.p2_q1[k];
        y_y0[idx_yy0(4, 2, k, M)] = s.p2_q2[k];
        y_y0[idx_yy0(4, 3, k, M)] = s.p2_p1[k];
        y_y0[idx_yy0(4, 4, k, M)] = s.p2_p2[k];
        y_y0[idx_yy0(4, 5, k, M)] = s.p2_Lr[k];

        y_y0[idx_yy0(5, 0, k, M)] = s.Lr_nu[k] * time_scale;
        y_y0[idx_yy0(5, 1, k, M)] = s.Lr_q1[k];
        y_y0[idx_yy0(5, 2, k, M)] = s.Lr_q2[k];
        y_y0[idx_yy0(5, 3, k, M)] = s.Lr_p1[k];
        y_y0[idx_yy0(5, 4, k, M)] = s.Lr_p2[k];
        y_y0[idx_yy0(5, 5, k, M)] = s.Lr_Lr[k];

        y_prop[idx_yprop(k, 0)] /= time_scale;
    }
}

} // namespace geqoe
} // namespace astrodyn_core
