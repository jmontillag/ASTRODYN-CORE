#pragma once

#include <cstddef>

namespace astrodyn_core {
namespace math {

// Pure C++ core functions. 
// They operate on raw memory pointers and know nothing about Python or NumPy.
// Memory must be allocated by the caller (the Python wrapper or another C++ function).

/**
 * Computes the derivatives of the inverse of a scalar function, f(t) = 1/a(t).
 * 
 * @param a_ptr Pointer to the input array [a, a', a'', ...]
 * @param out_ptr Pointer to the output array where results will be written
 * @param n Length of the input array
 * @param do_one If true, only computes and writes the highest order derivative to out_ptr[0]
 */
void compute_derivatives_of_inverse(const double* a_ptr, double* out_ptr, size_t n, bool do_one);

/**
 * Computes the partial derivatives of the derivatives of f(t) = 1/a(t) with respect to a parameter.
 * 
 * @param a_ptr Pointer to the function array [a, a', a'', ...]
 * @param ad_ptr Pointer to the partial derivatives array [∂a/∂x, ∂a'/∂x, ...]
 * @param out_ptr Pointer to the output array where results will be written
 * @param n Length of the arrays
 * @param do_one If true, only computes and writes the highest order derivative to out_ptr[0]
 */
void compute_derivatives_of_inverse_wrt_param(const double* a_ptr, const double* ad_ptr, double* out_ptr, size_t n, bool do_one);

/**
 * Computes derivatives of the product f(t) = a(t) * a'(t).
 * 
 * @param a_ptr Pointer to the input array [a, a', a'', ...]
 * @param out_ptr Pointer to the output array (length n-1)
 * @param m Length of the input array (n = m - 1)
 * @param do_one If true, only computes and writes the highest order derivative to out_ptr[0]
 */
void compute_derivatives_of_product(const double* a_ptr, double* out_ptr, size_t m, bool do_one);

/**
 * Computes the partial derivatives of the derivatives of f(t) = a(t) * a'(t) with respect to a parameter.
 * 
 * @param a_ptr Pointer to the function array [a, a', a'', ...]
 * @param ad_ptr Pointer to the partial derivatives array [∂a/∂x, ∂a'/∂x, ...]
 * @param out_ptr Pointer to the output array (length n-1)
 * @param m Length of the input arrays (n = m - 1)
 * @param do_one If true, only computes and writes the highest order derivative to out_ptr[0]
 */
void compute_derivatives_of_product_wrt_param(const double* a_ptr, const double* ad_ptr, double* out_ptr, size_t m, bool do_one);

} // namespace math
} // namespace astrodyn_core
