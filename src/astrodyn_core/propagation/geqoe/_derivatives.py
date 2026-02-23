"""Pure-Python derivative chain-rule utilities for the GEqOE propagator.

Provides four functions for computing Taylor-coefficient derivatives of
composite expressions (inverse, product) and their parameter sensitivities.
These are bit-identical to the legacy ``extra_utils/math_utils.py``.

The C++ accelerated versions in ``math_cpp`` are independent and serve
their own consumers; the Python propagator uses these directly.
"""

from __future__ import annotations

import numpy as np


def derivatives_of_inverse(
    a_vector: np.ndarray, do_one: bool = False
):
    n = len(a_vector)
    a = a_vector[0]

    if do_one:
        if n == 1:
            return 1.0 / a
        ap = a_vector[1]
        if n == 2:
            return -ap / a**2
        ap2 = a_vector[2]
        if n == 3:
            return 2 * ap**2 / a**3 - ap2 / a**2
        ap3 = a_vector[3]
        if n == 4:
            return 6 * ap2 * ap / a**3 - ap3 / a**2 - 6 * ap**3 / a**4
        ap4 = a_vector[4]
        if n == 5:
            return (
                8 * ap3 * ap / a**3
                + 6 * ap2**2 / a**3
                - 36 * ap**2 * ap2 / a**4
                - ap4 / a**2
                + 24 * ap**4 / a**5
            )
        raise NotImplementedError

    fia = np.zeros(n)
    fia[0] = 1.0 / a
    if n >= 2:
        fia[1] = -a_vector[1] / a**2
    if n >= 3:
        ap, ap2 = a_vector[1], a_vector[2]
        fia[2] = 2 * ap**2 / a**3 - ap2 / a**2
    if n >= 4:
        ap, ap2, ap3 = a_vector[1], a_vector[2], a_vector[3]
        fia[3] = 6 * ap2 * ap / a**3 - ap3 / a**2 - 6 * ap**3 / a**4
    if n >= 5:
        ap, ap2, ap3, ap4 = (
            a_vector[1],
            a_vector[2],
            a_vector[3],
            a_vector[4],
        )
        fia[4] = (
            8 * ap3 * ap / a**3
            + 6 * ap2**2 / a**3
            - 36 * ap**2 * ap2 / a**4
            - ap4 / a**2
            + 24 * ap**4 / a**5
        )
    return fia


def derivatives_of_inverse_wrt_param(
    a_vector, a_d_vector, do_one=False
):
    n = len(a_vector)
    a, a_d = a_vector[0], a_d_vector[0]

    if do_one:
        if n == 1:
            return -a_d / a**2
        ap, ap_d = a_vector[1], a_d_vector[1]
        if n == 2:
            return -ap_d / a**2 + 2 * ap * a_d / a**3
        ap2, ap2_d = a_vector[2], a_d_vector[2]
        if n == 3:
            return (
                -ap2_d / a**2
                + 2 * (ap2 * a_d + 2 * ap * ap_d) / a**3
                - 6 * ap**2 * a_d / a**4
            )
        ap3, ap3_d = a_vector[3], a_d_vector[3]
        if n == 4:
            return (
                -ap3_d / a**2
                + 2 * (ap3 * a_d + 3 * (ap2_d * ap + ap2 * ap_d)) / a**3
                - 18 * (ap2 * ap * a_d + ap**2 * ap_d) / a**4
                + 24 * ap**3 * a_d / a**5
            )
        ap4, ap4_d = a_vector[4], a_d_vector[4]
        if n == 5:
            return (
                -ap4_d / a**2
                + 2
                * (ap4 * a_d + 6 * ap2 * ap2_d + 4 * (ap3_d * ap + ap3 * ap_d))
                / a**3
                - 6
                * (
                    4 * ap3 * ap * a_d
                    + 6 * (2 * ap * ap_d * ap2 + ap**2 * ap2_d)
                    + 3 * ap2**2 * a_d
                )
                / a**4
                + 12 * (12 * ap**2 * ap2 * a_d - 8 * ap**3 * ap_d) / a**5
                + 120 * ap**4 * a_d / a**6
            )
        raise NotImplementedError

    fia_d = np.zeros(n)
    fia_d[0] = -a_d / a**2
    if n >= 2:
        ap, ap_d = a_vector[1], a_d_vector[1]
        fia_d[1] = -ap_d / a**2 + 2 * ap * a_d / a**3
    if n >= 3:
        ap2, ap2_d = a_vector[2], a_d_vector[2]
        fia_d[2] = (
            -ap2_d / a**2
            + 2 * (ap2 * a_d + 2 * ap * ap_d) / a**3
            - 6 * ap**2 * a_d / a**4
        )
    if n >= 4:
        ap3, ap3_d = a_vector[3], a_d_vector[3]
        fia_d[3] = (
            -ap3_d / a**2
            + 2 * (ap3 * a_d + 3 * (ap2_d * ap + ap2 * ap_d)) / a**3
            - 18 * (ap2 * ap * a_d + ap**2 * ap_d) / a**4
            + 24 * ap**3 * a_d / a**5
        )
    if n >= 5:
        ap4, ap4_d = a_vector[4], a_d_vector[4]
        fia_d[4] = (
            -ap4_d / a**2
            + 2
            * (ap4 * a_d + 6 * ap2 * ap2_d + 4 * (ap3_d * ap + ap3 * ap_d))
            / a**3
            - 6
            * (
                4 * ap3 * ap * a_d
                + 6 * (2 * ap * ap_d * ap2 + ap**2 * ap2_d)
                + 3 * ap2**2 * a_d
            )
            / a**4
            + 12 * (12 * ap**2 * ap2 * a_d - 8 * ap**3 * ap_d) / a**5
            + 120 * ap**4 * a_d / a**6
        )
    return fia_d


def derivatives_of_product(
    a_vector: np.ndarray, do_one: bool = False
):
    m = len(a_vector)
    n = m - 1
    if n < 1:
        return np.array([]) if not do_one else 0.0

    a, ap = a_vector[0], a_vector[1]

    if do_one:
        if n == 1:
            return a * ap
        ap2 = a_vector[2]
        if n == 2:
            return ap**2 + a * ap2
        ap3 = a_vector[3]
        if n == 3:
            return 3 * ap * ap2 + a * ap3
        ap4 = a_vector[4]
        if n == 4:
            return 3 * ap2**2 + 4 * ap * ap3 + a * ap4
        ap5 = a_vector[5]
        if n == 5:
            return 10 * ap2 * ap3 + 5 * ap * ap4 + a * ap5
        raise NotImplementedError

    fa2p = np.zeros(n, dtype=a_vector.dtype)
    fa2p[0] = a * ap
    if n >= 2:
        ap2 = a_vector[2]
        fa2p[1] = ap**2 + a * ap2
    if n >= 3:
        ap2, ap3 = a_vector[2], a_vector[3]
        fa2p[2] = 3 * ap * ap2 + a * ap3
    if n >= 4:
        ap2, ap3, ap4 = a_vector[2], a_vector[3], a_vector[4]
        fa2p[3] = 3 * ap2**2 + 4 * ap * ap3 + a * ap4
    if n >= 5:
        ap2, ap3, ap4, ap5 = (
            a_vector[2],
            a_vector[3],
            a_vector[4],
            a_vector[5],
        )
        fa2p[4] = 10 * ap2 * ap3 + 5 * ap * ap4 + a * ap5
    return fa2p


def derivatives_of_product_wrt_param(
    a_vector, a_d_vector, do_one=False
):
    m = len(a_vector)
    n = m - 1
    if n < 1:
        return np.array([]) if not do_one else 0.0

    a, a_d = a_vector[0], a_d_vector[0]
    ap, ap_d = a_vector[1], a_d_vector[1]

    if do_one:
        if n == 1:
            return a_d * ap + a * ap_d
        ap2, ap2_d = a_vector[2], a_d_vector[2]
        if n == 2:
            return 2 * ap * ap_d + a_d * ap2 + a * ap2_d
        ap3, ap3_d = a_vector[3], a_d_vector[3]
        if n == 3:
            return 3 * ap_d * ap2 + 3 * ap * ap2_d + a_d * ap3 + a * ap3_d
        ap4, ap4_d = a_vector[4], a_d_vector[4]
        if n == 4:
            return (
                6 * ap2 * ap2_d
                + 4 * ap_d * ap3
                + 4 * ap * ap3_d
                + a_d * ap4
                + a * ap4_d
            )
        ap5, ap5_d = a_vector[5], a_d_vector[5]
        if n == 5:
            return (
                10 * ap2_d * ap3
                + 10 * ap2 * ap3_d
                + 5 * ap_d * ap4
                + 5 * ap * ap4_d
                + a_d * ap5
                + a * ap5_d
            )
        raise NotImplementedError

    fa2p_d = np.zeros(n, dtype=a_vector.dtype)
    fa2p_d[0] = a_d * ap + a * ap_d
    if n >= 2:
        ap2, ap2_d = a_vector[2], a_d_vector[2]
        fa2p_d[1] = 2 * ap * ap_d + a_d * ap2 + a * ap2_d
    if n >= 3:
        ap2, ap2_d = a_vector[2], a_d_vector[2]
        ap3, ap3_d = a_vector[3], a_d_vector[3]
        fa2p_d[2] = 3 * ap_d * ap2 + 3 * ap * ap2_d + a_d * ap3 + a * ap3_d
    if n >= 4:
        ap2, ap2_d = a_vector[2], a_d_vector[2]
        ap3, ap3_d = a_vector[3], a_d_vector[3]
        ap4, ap4_d = a_vector[4], a_d_vector[4]
        fa2p_d[3] = (
            6 * ap2 * ap2_d
            + 4 * ap_d * ap3
            + 4 * ap * ap3_d
            + a_d * ap4
            + a * ap4_d
        )
    if n >= 5:
        ap2, ap2_d = a_vector[2], a_d_vector[2]
        ap3, ap3_d = a_vector[3], a_d_vector[3]
        ap4, ap4_d = a_vector[4], a_d_vector[4]
        ap5, ap5_d = a_vector[5], a_d_vector[5]
        fa2p_d[4] = (
            10 * ap2_d * ap3
            + 10 * ap2 * ap3_d
            + 5 * ap_d * ap4
            + 5 * ap * ap4_d
            + a_d * ap5
            + a * ap5_d
        )
    return fa2p_d


__all__ = [
    "derivatives_of_inverse",
    "derivatives_of_inverse_wrt_param",
    "derivatives_of_product",
    "derivatives_of_product_wrt_param",
]
