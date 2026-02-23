#pragma once

#include <cstddef>

namespace astrodyn_core {
namespace geqoe {

struct PropagationConstants {
    double j2;
    double re;
    double mu;
    double length_scale;
    double time_scale;
    double mu_norm;
    double a_half_j2;
};

struct StmAccumulatorView {
    const double* nu_nu;

    const double* q1_nu;
    const double* q1_q1;
    const double* q1_q2;
    const double* q1_p1;
    const double* q1_p2;
    const double* q1_Lr;

    const double* q2_nu;
    const double* q2_q1;
    const double* q2_q2;
    const double* q2_p1;
    const double* q2_p2;
    const double* q2_Lr;

    const double* p1_nu;
    const double* p1_q1;
    const double* p1_q2;
    const double* p1_p1;
    const double* p1_p2;
    const double* p1_Lr;

    const double* p2_nu;
    const double* p2_q1;
    const double* p2_q2;
    const double* p2_p1;
    const double* p2_p2;
    const double* p2_Lr;

    const double* Lr_nu;
    const double* Lr_q1;
    const double* Lr_q2;
    const double* Lr_p1;
    const double* Lr_p2;
    const double* Lr_Lr;
};

bool is_valid_order(int order);
PropagationConstants make_constants(double j2, double re, double mu);

void normalize_time_grid(
    const double* dt_seconds,
    std::size_t M,
    double time_scale,
    double* dt_norm_out
);

void assemble_stm_and_denormalize_nu(
    const StmAccumulatorView& s,
    std::size_t M,
    double time_scale,
    double* y_prop,
    double* y_y0
);

} // namespace geqoe
} // namespace astrodyn_core
