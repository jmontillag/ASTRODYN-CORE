"""Matrix conversion helpers for Orekit/Hipparchus interop."""

from __future__ import annotations

from typing import Any

import numpy as np


def realmatrix_to_numpy(mat: Any) -> np.ndarray:
    """Convert a Hipparchus/Orekit ``RealMatrix`` to a NumPy array.

    Args:
        mat: Java ``RealMatrix`` instance.

    Returns:
        NumPy array with the same numeric contents.
    """
    from orekit import JArray_double

    return np.array([JArray_double.cast_(r) for r in mat.getData()])


def new_java_double_2d(rows: int, cols: int) -> Any:
    """Allocate a Java ``double[][]`` for Orekit Jacobian APIs.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        Java array object compatible with Orekit Jacobian fill methods.
    """
    from orekit import JArray_double, JArray_object

    out = JArray_object(rows)
    for i in range(rows):
        out[i] = JArray_double(cols)
    return out


def java_double_2d_to_numpy(mat: Any, rows: int) -> np.ndarray:
    """Convert a Java ``double[][]`` into a NumPy array.

    Args:
        mat: Java ``double[][]`` matrix.
        rows: Number of rows to read.

    Returns:
        Array of shape ``(rows, n)`` inferred from the Java matrix rows.
    """
    from orekit import JArray_double

    return np.array([JArray_double.cast_(mat[i]) for i in range(rows)], dtype=np.float64)


def numpy_to_realmatrix(arr: np.ndarray) -> Any:
    """Convert a 2-D NumPy array to a Hipparchus ``RealMatrix``.

    Args:
        arr: Two-dimensional numeric array.

    Returns:
        Java ``RealMatrix`` with copied values.
    """
    from org.hipparchus.linear import MatrixUtils

    rows, cols = arr.shape
    mat = MatrixUtils.createRealMatrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            mat.setEntry(i, j, float(arr[i, j]))
    return mat
