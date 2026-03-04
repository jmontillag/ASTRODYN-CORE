#pragma once

#include <cstddef>
#include <vector>

#include "propagator_core.hpp"

namespace astrodyn_core {
namespace geqoe {

struct Order1Coefficients {
    double nu_0;
    double q1_0;
    double q2_0;
    double p1_0;
    double p2_0;
    double Lr_0;

    double q1p_0;
    double q2p_0;
    double p1p_0;
    double p2p_0;
    double Lrp_0;

    double q1p_nu;
    double q1p_Lr;
    double q1p_q1;
    double q1p_q2;
    double q1p_p1;
    double q1p_p2;

    double q2p_nu;
    double q2p_Lr;
    double q2p_q1;
    double q2p_q2;
    double q2p_p1;
    double q2p_p2;

    double p1p_nu;
    double p1p_Lr;
    double p1p_q1;
    double p1p_q2;
    double p1p_p1;
    double p1p_p2;

    double p2p_nu;
    double p2p_Lr;
    double p2p_q1;
    double p2p_q2;
    double p2p_p1;
    double p2p_p2;

    double Lrp_nu;
    double Lrp_Lr;
    double Lrp_q1;
    double Lrp_q2;
    double Lrp_p1;
    double Lrp_p2;

    double map_components_col0[6];
};

struct Order1EvaluationScratch {
    std::vector<double> nu_nu;

    std::vector<double> q1_nu;
    std::vector<double> q1_q1;
    std::vector<double> q1_q2;
    std::vector<double> q1_p1;
    std::vector<double> q1_p2;
    std::vector<double> q1_Lr;

    std::vector<double> q2_nu;
    std::vector<double> q2_q1;
    std::vector<double> q2_q2;
    std::vector<double> q2_p1;
    std::vector<double> q2_p2;
    std::vector<double> q2_Lr;

    std::vector<double> p1_nu;
    std::vector<double> p1_q1;
    std::vector<double> p1_q2;
    std::vector<double> p1_p1;
    std::vector<double> p1_p2;
    std::vector<double> p1_Lr;

    std::vector<double> p2_nu;
    std::vector<double> p2_q1;
    std::vector<double> p2_q2;
    std::vector<double> p2_p1;
    std::vector<double> p2_p2;
    std::vector<double> p2_Lr;

    std::vector<double> Lr_nu;
    std::vector<double> Lr_q1;
    std::vector<double> Lr_q2;
    std::vector<double> Lr_p1;
    std::vector<double> Lr_p2;
    std::vector<double> Lr_Lr;

    void resize(std::size_t M);
    StmAccumulatorView view() const;
};

struct Order1Intermediates {
    double nu_0;
    double q1_0;
    double q2_0;
    double p1_0;
    double p2_0;
    double Lr_0;
    double T;
    double mu_norm;
    double A;
    double a;
    double a_nu;
    double r;
    double rp;
    double r2;
    double r3;
    double X;
    double Y;
    double Xp;
    double Yp;
    double cosL;
    double sinL;
    double alpha;
    double beta;
    double fib;
    double fic;
    double fih;
    double fir;
    double firp;
    double fir2;
    double fir2p;
    double fir3;
    double fir3p;
    double fihr3;
    double fiD;
    double fiDp;
    double c;
    double h;
    double hr;
    double hr3;
    double d;
    double wh;
    double I;
    double U;
    double delta;
    double zg;
    double GAMMA_;
    double xi1;
    double xi2;
    double fUz;
    double qs;
    double q1s;
    double q2s;
    double p1s;
    double p2s;
    double C;
    double D;
    double rpn;
    double fialpha;
    double f2rp;
    double p1p_0;
    double p2p_0;
    double Lrp_0;
    double q1p_0;
    double q2p_0;
    double r_nu;
    double r_Lr;
    double r_q1;
    double r_q2;
    double r_p1;
    double r_p2;
    double rp_nu;
    double rp_Lr;
    double rp_q1;
    double rp_q2;
    double rp_p1;
    double rp_p2;
    double r2_nu;
    double r2_Lr;
    double r2_q1;
    double r2_q2;
    double r2_p1;
    double r2_p2;
    double r3_nu;
    double r3_Lr;
    double r3_q1;
    double r3_q2;
    double r3_p1;
    double r3_p2;
    double beta_nu;
    double beta_Lr;
    double beta_q1;
    double beta_q2;
    double beta_p1;
    double beta_p2;
    double alpha_nu;
    double alpha_Lr;
    double alpha_q1;
    double alpha_q2;
    double alpha_p1;
    double alpha_p2;
    double fialpha_nu;
    double fialpha_Lr;
    double fialpha_q1;
    double fialpha_q2;
    double fialpha_p1;
    double fialpha_p2;
    double c_nu;
    double c_Lr;
    double c_q1;
    double c_q2;
    double c_p1;
    double c_p2;
    double fic_nu;
    double fic_Lr;
    double fic_q1;
    double fic_q2;
    double fic_p1;
    double fic_p2;
    double h_nu;
    double h_Lr;
    double h_q1;
    double h_q2;
    double h_p1;
    double h_p2;
    double fih_nu;
    double fih_Lr;
    double fih_q1;
    double fih_q2;
    double fih_p1;
    double fih_p2;
    double X_nu;
    double X_Lr;
    double X_q1;
    double X_q2;
    double X_p1;
    double X_p2;
    double Y_nu;
    double Y_Lr;
    double Y_q1;
    double Y_q2;
    double Y_p1;
    double Y_p2;
    double cosL_nu;
    double cosL_Lr;
    double cosL_q1;
    double cosL_q2;
    double cosL_p1;
    double cosL_p2;
    double sinL_nu;
    double sinL_Lr;
    double sinL_q1;
    double sinL_q2;
    double sinL_p1;
    double sinL_p2;
    double zg_nu;
    double zg_Lr;
    double zg_q1;
    double zg_q2;
    double zg_p1;
    double zg_p2;
    double fUz_nu;
    double fUz_Lr;
    double fUz_q1;
    double fUz_q2;
    double fUz_p1;
    double fUz_p2;
    double U_nu;
    double U_Lr;
    double U_q1;
    double U_q2;
    double U_p1;
    double U_p2;
    double hr3_nu;
    double hr3_Lr;
    double hr3_q1;
    double hr3_q2;
    double hr3_p1;
    double hr3_p2;
    double fihr3_nu;
    double fihr3_Lr;
    double fihr3_q1;
    double fihr3_q2;
    double fihr3_p1;
    double fihr3_p2;
    double delta_nu;
    double delta_Lr;
    double delta_q1;
    double delta_q2;
    double delta_p1;
    double delta_p2;
    double I_nu;
    double I_Lr;
    double I_q1;
    double I_q2;
    double I_p1;
    double I_p2;
    double d_nu;
    double d_Lr;
    double d_q1;
    double d_q2;
    double d_p1;
    double d_p2;
    double wh_nu;
    double wh_Lr;
    double wh_q1;
    double wh_q2;
    double wh_p1;
    double wh_p2;
    double GAMMA_nu;
    double GAMMA_Lr;
    double GAMMA_q1;
    double GAMMA_q2;
    double GAMMA_p1;
    double GAMMA_p2;
    double xi1_nu;
    double xi1_Lr;
    double xi1_q1;
    double xi1_q2;
    double xi1_p1;
    double xi1_p2;
    double xi2_nu;
    double xi2_Lr;
    double xi2_q1;
    double xi2_q2;
    double xi2_p1;
    double xi2_p2;
    double q1p_nu;
    double q1p_Lr;
    double q1p_q1;
    double q1p_q2;
    double q1p_p1;
    double q1p_p2;
    double q2p_nu;
    double q2p_Lr;
    double q2p_q1;
    double q2p_q2;
    double q2p_p1;
    double q2p_p2;
    double p1p_nu;
    double p1p_Lr;
    double p1p_q1;
    double p1p_q2;
    double p1p_p1;
    double p1p_p2;
    double p2p_nu;
    double p2p_Lr;
    double p2p_q1;
    double p2p_q2;
    double p2p_p1;
    double p2p_p2;
    double Lrp_nu;
    double Lrp_Lr;
    double Lrp_q1;
    double Lrp_q2;
    double Lrp_p1;
    double Lrp_p2;
    double rpn_nu;
    double rpn_Lr;
    double rpn_q1;
    double rpn_q2;
    double rpn_p1;
    double rpn_p2;
    double D_nu;
    double D_Lr;
    double D_q1;
    double D_q2;
    double D_p1;
    double D_p2;
    double fiD_nu;
    double fiD_Lr;
    double fiD_q1;
    double fiD_q2;
    double fiD_p1;
    double fiD_p2;
    double C_nu;
    double C_Lr;
    double C_q1;
    double C_q2;
    double C_p1;
    double C_p2;
    double fir3_p_nu;
    double fir3_p_Lr;
    double fir3_p_q1;
    double fir3_p_q2;
    double fir3_p_p1;
    double fir3_p_p2;
    double fir_nu;
    double fir_Lr;
    double fir_q1;
    double fir_q2;
    double fir_p1;
    double fir_p2;
    double firp_nu;
    double firp_Lr;
    double firp_q1;
    double firp_q2;
    double firp_p1;
    double firp_p2;
    double fir2_nu;
    double fir2_Lr;
    double fir2_q1;
    double fir2_q2;
    double fir2_p1;
    double fir2_p2;
    double fir2p_nu;
    double fir2p_Lr;
    double fir2p_q1;
    double fir2p_q2;
    double fir2p_p1;
    double fir2p_p2;
    double fir3_nu;
    double fir3_Lr;
    double fir3_q1;
    double fir3_q2;
    double fir3_p1;
    double fir3_p2;
    double f2rp_nu;
    double f2rp_Lr;
    double f2rp_q1;
    double f2rp_q2;
    double f2rp_p1;
    double f2rp_p2;
    // Vector-only locals: variables stored in vectors but not written to scratch.
    // Needed by higher orders for derivative calls on inherited vectors.
    double fibm1;       // = alpha (element of alpha_vector)
    double r3p;         // = 3*r2*rp (element of r3_vector)
    double r3p_nu;      // partials of r3p (elements of r3_XX_vector)
    double r3p_Lr;
    double r3p_q1;
    double r3p_q2;
    double r3p_p1;
    double r3p_p2;
};

void compute_coefficients_1(
    const double* y0,
    const PropagationConstants& constants,
    Order1Coefficients& out
);

void compute_coefficients_1(
    const double* y0,
    const PropagationConstants& constants,
    Order1Coefficients& out,
    Order1Intermediates& inter
);

void evaluate_order_1(
    const Order1Coefficients& coeffs,
    const double* dt_norm,
    std::size_t M,
    double* y_prop,
    Order1EvaluationScratch& scratch
);

} // namespace geqoe
} // namespace astrodyn_core
