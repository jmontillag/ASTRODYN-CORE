#include "math_utils.hpp"
#include <cmath>
#include <stdexcept>

namespace astrodyn_core {
namespace math {

void compute_derivatives_of_inverse(const double* a_ptr, double* out_ptr, size_t n, bool do_one) {
    if (n == 0) {
        if (do_one) out_ptr[0] = 0.0;
        return;
    }

    double a = a_ptr[0];

    if (do_one) {
        if (n == 1) { out_ptr[0] = 1.0 / a; return; }
        double ap = a_ptr[1];
        if (n == 2) { out_ptr[0] = -ap / (a * a); return; }
        double ap2 = a_ptr[2];
        if (n == 3) { out_ptr[0] = 2.0 * std::pow(ap, 2) / std::pow(a, 3) - ap2 / std::pow(a, 2); return; }
        double ap3 = a_ptr[3];
        if (n == 4) { out_ptr[0] = 6.0 * ap2 * ap / std::pow(a, 3) - ap3 / std::pow(a, 2) - 6.0 * std::pow(ap, 3) / std::pow(a, 4); return; }
        double ap4 = a_ptr[4];
        if (n == 5) {
            out_ptr[0] = (8.0 * ap3 * ap / std::pow(a, 3)
                        + 6.0 * std::pow(ap2, 2) / std::pow(a, 3)
                        - 36.0 * std::pow(ap, 2) * ap2 / std::pow(a, 4)
                        - ap4 / std::pow(a, 2)
                        + 24.0 * std::pow(ap, 4) / std::pow(a, 5));
            return;
        }
        throw std::runtime_error("Derivatives of inverse beyond 4th order not implemented for do_one=True.");
    }

    out_ptr[0] = 1.0 / a;
    if (n >= 2) out_ptr[1] = -a_ptr[1] / (a * a);
    if (n >= 3) out_ptr[2] = 2.0 * std::pow(a_ptr[1], 2) / std::pow(a, 3) - a_ptr[2] / std::pow(a, 2);
    if (n >= 4) out_ptr[3] = 6.0 * a_ptr[2] * a_ptr[1] / std::pow(a, 3) - a_ptr[3] / std::pow(a, 2) - 6.0 * std::pow(a_ptr[1], 3) / std::pow(a, 4);
    if (n >= 5) {
        out_ptr[4] = (8.0 * a_ptr[3] * a_ptr[1] / std::pow(a, 3)
                    + 6.0 * std::pow(a_ptr[2], 2) / std::pow(a, 3)
                    - 36.0 * std::pow(a_ptr[1], 2) * a_ptr[2] / std::pow(a, 4)
                    - a_ptr[4] / std::pow(a, 2)
                    + 24.0 * std::pow(a_ptr[1], 4) / std::pow(a, 5));
    }
    if (n > 5) throw std::runtime_error("Derivatives of inverse beyond 4th order not implemented for do_one=False.");
}

void compute_derivatives_of_inverse_wrt_param(const double* a_ptr, const double* ad_ptr, double* out_ptr, size_t n, bool do_one) {
    if (n == 0) {
        if (do_one) out_ptr[0] = 0.0;
        return;
    }

    double a = a_ptr[0];
    double a_d = ad_ptr[0];

    if (do_one) {
        if (n == 1) { out_ptr[0] = -a_d / (a * a); return; }
        double ap = a_ptr[1], ap_d = ad_ptr[1];
        if (n == 2) { out_ptr[0] = -ap_d / (a * a) + 2.0 * ap * a_d / std::pow(a, 3); return; }
        double ap2 = a_ptr[2], ap2_d = ad_ptr[2];
        if (n == 3) { out_ptr[0] = -ap2_d / (a * a) + 2.0 * (ap2 * a_d + 2.0 * ap * ap_d) / std::pow(a, 3) - 6.0 * std::pow(ap, 2) * a_d / std::pow(a, 4); return; }
        double ap3 = a_ptr[3], ap3_d = ad_ptr[3];
        if (n == 4) { out_ptr[0] = -ap3_d / (a * a) + 2.0 * (ap3 * a_d + 3.0 * (ap2_d * ap + ap2 * ap_d)) / std::pow(a, 3) - 18.0 * (ap2 * ap * a_d + std::pow(ap, 2) * ap_d) / std::pow(a, 4) + 24.0 * std::pow(ap, 3) * a_d / std::pow(a, 5); return; }
        double ap4 = a_ptr[4], ap4_d = ad_ptr[4];
        if (n == 5) {
            out_ptr[0] = -ap4_d / (a * a) + 2.0 * (ap4 * a_d + 6.0 * ap2 * ap2_d + 4.0 * (ap3_d * ap + ap3 * ap_d)) / std::pow(a, 3) 
                         - 6.0 * (4.0 * ap3 * ap * a_d + 6.0 * (2.0 * ap * ap_d * ap2 + std::pow(ap, 2) * ap2_d) + 3.0 * std::pow(ap2, 2) * a_d) / std::pow(a, 4) 
                         + 12.0 * (12.0 * std::pow(ap, 2) * ap2 * a_d - 8.0 * std::pow(ap, 3) * ap_d) / std::pow(a, 5) 
                         + 120.0 * std::pow(ap, 4) * a_d / std::pow(a, 6);
            return;
        }
        throw std::runtime_error("Only up to 4th-order available");
    }

    out_ptr[0] = -a_d / (a * a);
    if (n >= 2) out_ptr[1] = -ad_ptr[1] / (a * a) + 2.0 * a_ptr[1] * a_d / std::pow(a, 3);
    if (n >= 3) out_ptr[2] = -ad_ptr[2] / (a * a) + 2.0 * (a_ptr[2] * a_d + 2.0 * a_ptr[1] * ad_ptr[1]) / std::pow(a, 3) - 6.0 * std::pow(a_ptr[1], 2) * a_d / std::pow(a, 4);
    if (n >= 4) out_ptr[3] = -ad_ptr[3] / (a * a) + 2.0 * (a_ptr[3] * a_d + 3.0 * (ad_ptr[2] * a_ptr[1] + a_ptr[2] * ad_ptr[1])) / std::pow(a, 3) - 18.0 * (a_ptr[2] * a_ptr[1] * a_d + std::pow(a_ptr[1], 2) * ad_ptr[1]) / std::pow(a, 4) + 24.0 * std::pow(a_ptr[1], 3) * a_d / std::pow(a, 5);
    if (n >= 5) {
        out_ptr[4] = -ad_ptr[4] / (a * a) + 2.0 * (a_ptr[4] * a_d + 6.0 * a_ptr[2] * ad_ptr[2] + 4.0 * (ad_ptr[3] * a_ptr[1] + a_ptr[3] * ad_ptr[1])) / std::pow(a, 3) 
                     - 6.0 * (4.0 * a_ptr[3] * a_ptr[1] * a_d + 6.0 * (2.0 * a_ptr[1] * ad_ptr[1] * a_ptr[2] + std::pow(a_ptr[1], 2) * ad_ptr[2]) + 3.0 * std::pow(a_ptr[2], 2) * a_d) / std::pow(a, 4) 
                     + 12.0 * (12.0 * std::pow(a_ptr[1], 2) * a_ptr[2] * a_d - 8.0 * std::pow(a_ptr[1], 3) * ad_ptr[1]) / std::pow(a, 5) 
                     + 120.0 * std::pow(a_ptr[1], 4) * a_d / std::pow(a, 6);
    }
    if (n > 5) throw std::runtime_error("Beyond 4th order not implemented.");
}

void compute_derivatives_of_product(const double* a_ptr, double* out_ptr, size_t m, bool do_one) {
    int n = static_cast<int>(m) - 1;
    if (n < 1) {
        if (do_one) out_ptr[0] = 0.0;
        return;
    }

    double a = a_ptr[0];
    double ap = a_ptr[1];

    if (do_one) {
        if (n == 1) { out_ptr[0] = a * ap; return; }
        double ap2 = a_ptr[2];
        if (n == 2) { out_ptr[0] = std::pow(ap, 2) + a * ap2; return; }
        double ap3 = a_ptr[3];
        if (n == 3) { out_ptr[0] = 3.0 * ap * ap2 + a * ap3; return; }
        double ap4 = a_ptr[4];
        if (n == 4) { out_ptr[0] = 3.0 * std::pow(ap2, 2) + 4.0 * ap * ap3 + a * ap4; return; }
        double ap5 = a_ptr[5];
        if (n == 5) { out_ptr[0] = 10.0 * ap2 * ap3 + 5.0 * ap * ap4 + a * ap5; return; }
        throw std::runtime_error("Derivatives of product beyond 5th order not implemented.");
    }

    out_ptr[0] = a * ap;
    if (n >= 2) out_ptr[1] = std::pow(ap, 2) + a * a_ptr[2];
    if (n >= 3) out_ptr[2] = 3.0 * ap * a_ptr[2] + a * a_ptr[3];
    if (n >= 4) out_ptr[3] = 3.0 * std::pow(a_ptr[2], 2) + 4.0 * ap * a_ptr[3] + a * a_ptr[4];
    if (n >= 5) out_ptr[4] = 10.0 * a_ptr[2] * a_ptr[3] + 5.0 * ap * a_ptr[4] + a * a_ptr[5];
    if (n > 5) throw std::runtime_error("Derivatives of product beyond 5th order not implemented.");
}

void compute_derivatives_of_product_wrt_param(const double* a_ptr, const double* ad_ptr, double* out_ptr, size_t m, bool do_one) {
    int n = static_cast<int>(m) - 1;
    if (n < 1) {
        if (do_one) out_ptr[0] = 0.0;
        return;
    }

    double a = a_ptr[0];
    double a_d = ad_ptr[0];
    double ap = a_ptr[1], ap_d = ad_ptr[1];

    if (do_one) {
        if (n == 1) { out_ptr[0] = a_d * ap + a * ap_d; return; }
        double ap2 = a_ptr[2], ap2_d = ad_ptr[2];
        if (n == 2) { out_ptr[0] = 2.0 * ap * ap_d + a_d * ap2 + a * ap2_d; return; }
        double ap3 = a_ptr[3], ap3_d = ad_ptr[3];
        if (n == 3) { out_ptr[0] = 3.0 * ap_d * ap2 + 3.0 * ap * ap2_d + a_d * ap3 + a * ap3_d; return; }
        double ap4 = a_ptr[4], ap4_d = ad_ptr[4];
        if (n == 4) { out_ptr[0] = 6.0 * ap2 * ap2_d + 4.0 * ap_d * ap3 + 4.0 * ap * ap3_d + a_d * ap4 + a * ap4_d; return; }
        double ap5 = a_ptr[5], ap5_d = ad_ptr[5];
        if (n == 5) { out_ptr[0] = 10.0 * ap2_d * ap3 + 10.0 * ap2 * ap3_d + 5.0 * ap_d * ap4 + 5.0 * ap * ap4_d + a_d * ap5 + a * ap5_d; return; }
        throw std::runtime_error("Partials of product derivatives beyond 5th order not implemented.");
    }

    out_ptr[0] = a_d * ap + a * ap_d;
    if (n >= 2) {
        double ap2 = a_ptr[2], ap2_d = ad_ptr[2];
        out_ptr[1] = 2.0 * ap * ap_d + a_d * ap2 + a * ap2_d;
    }
    if (n >= 3) {
        double ap2 = a_ptr[2], ap2_d = ad_ptr[2];
        double ap3 = a_ptr[3], ap3_d = ad_ptr[3];
        out_ptr[2] = 3.0 * ap_d * ap2 + 3.0 * ap * ap2_d + a_d * ap3 + a * ap3_d;
    }
    if (n >= 4) {
        double ap2 = a_ptr[2], ap2_d = ad_ptr[2];
        double ap3 = a_ptr[3], ap3_d = ad_ptr[3];
        double ap4 = a_ptr[4], ap4_d = ad_ptr[4];
        out_ptr[3] = 6.0 * ap2 * ap2_d + 4.0 * ap_d * ap3 + 4.0 * ap * ap3_d + a_d * ap4 + a * ap4_d;
    }
    if (n >= 5) {
        double ap2 = a_ptr[2], ap2_d = ad_ptr[2];
        double ap3 = a_ptr[3], ap3_d = ad_ptr[3];
        double ap4 = a_ptr[4], ap4_d = ad_ptr[4];
        double ap5 = a_ptr[5], ap5_d = ad_ptr[5];
        out_ptr[4] = 10.0 * ap2_d * ap3 + 10.0 * ap2 * ap3_d + 5.0 * ap_d * ap4 + 5.0 * ap * ap4_d + a_d * ap5 + a * ap5_d;
    }
    if (n > 5) throw std::runtime_error("Partials of product derivatives beyond 5th order not implemented.");
}

} // namespace math
} // namespace astrodyn_core