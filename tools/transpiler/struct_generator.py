"""Header and struct generation for GEqOE Taylor order files.

Generates:
- OrderNCoefficients struct (EOM values + partials + map column)
- Order{N-1}Intermediates struct (cross-order fields)
- Function declarations
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from tools.transpiler.ast_parser import (
    EvalScratchWrite,
    EvalStateUpdate,
    EvalStmUpdate,
    MapAssign,
    ParsedFunction,
    ScratchRead,
    ScratchWrite,
)
from tools.transpiler.data_flow import CrossOrderDeps


# ---------------------------------------------------------------------------
# STM fields (same for all orders)
# ---------------------------------------------------------------------------

STM_FIELDS = [
    "nu_nu",
    "q1_nu", "q1_Lr", "q1_q1", "q1_q2", "q1_p1", "q1_p2",
    "q2_nu", "q2_Lr", "q2_q1", "q2_q2", "q2_p1", "q2_p2",
    "p1_nu", "p1_Lr", "p1_q1", "p1_q2", "p1_p1", "p1_p2",
    "p2_nu", "p2_Lr", "p2_q1", "p2_q2", "p2_p1", "p2_p2",
    "Lr_nu", "Lr_Lr", "Lr_q1", "Lr_q2", "Lr_p1", "Lr_p2",
]


# ---------------------------------------------------------------------------
# Extract coefficient struct fields from evaluate function
# ---------------------------------------------------------------------------

def extract_eval_coeff_fields(evaluate: ParsedFunction) -> List[str]:
    """Extract fields needed in the Coefficients struct from evaluate reads.

    The evaluate function reads EOM coefficients from scratch/coeffs.
    These are the fields that need to be in OrderNCoefficients.
    """
    fields: List[str] = []
    stm_set = set(STM_FIELDS)

    for stmt in evaluate.statements:
        if isinstance(stmt, ScratchRead):
            if stmt.key not in stm_set:
                fields.append(stmt.key)

    return fields


def extract_map_column(compute: ParsedFunction) -> Optional[int]:
    """Extract which map column this order writes."""
    for stmt in compute.statements:
        if isinstance(stmt, MapAssign):
            return stmt.column
    return None


# ---------------------------------------------------------------------------
# Intermediates struct fields from cross-order deps
# ---------------------------------------------------------------------------

def extract_intermediates_fields(
    compute: ParsedFunction,
) -> List[str]:
    """Extract scalar scratch write keys from compute (for intermediates struct).

    Vector scratch writes (ending in '_vector') are skipped — vectors are
    tracked logically and don't need physical storage in the struct.
    """
    fields: List[str] = []
    seen: Set[str] = set()
    for stmt in compute.statements:
        if isinstance(stmt, ScratchWrite):
            if stmt.key not in seen and not stmt.key.endswith("_vector"):
                seen.add(stmt.key)
                fields.append(stmt.key)
    return fields


# ---------------------------------------------------------------------------
# Header generation
# ---------------------------------------------------------------------------

def generate_header(
    order: int,
    eval_coeff_fields: List[str],
    map_col: Optional[int],
    intermediates_fields: Optional[List[str]],
    prev_order_header: Optional[str],
) -> str:
    """Generate the complete taylor_order_N.hpp header file."""
    lines: List[str] = []

    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("#include <vector>")
    lines.append("")
    if prev_order_header:
        lines.append(f'#include "{prev_order_header}"')
    else:
        lines.append('#include "propagator_core.hpp"')
    lines.append("")
    lines.append("namespace astrodyn_core {")
    lines.append("namespace geqoe {")
    lines.append("")

    # --- Intermediates struct (scratch writes from this order, for next order) ---
    if intermediates_fields:
        lines.append(f"struct Order{order}Intermediates {{")
        for f in intermediates_fields:
            if f.endswith("_vector"):
                # Vector field — use a fixed-size array
                # Length will be determined from vector tracker
                lines.append(f"    // Vector: {f}")
            else:
                lines.append(f"    double {f};")
        lines.append("};")
        lines.append("")

    # --- Coefficients struct ---
    lines.append(f"struct Order{order}Coefficients {{")
    for f in eval_coeff_fields:
        lines.append(f"    double {f};")
    if map_col is not None:
        lines.append(f"    double map_components_col{map_col}[6];")
    lines.append("};")
    lines.append("")

    # --- Evaluation scratch struct (same 31-vector pattern for all orders) ---
    # Orders > 1 reuse Order1EvaluationScratch
    lines.append(f"// Order {order} reuses Order1EvaluationScratch")
    lines.append("")

    # --- Function declarations ---
    lines.append(f"void compute_coefficients_{order}(")
    lines.append(f"    const double* y0,")
    lines.append(f"    const PropagationConstants& constants,")
    if order > 1:
        for prev in range(1, order):
            lines.append(f"    Order{prev}Coefficients& out{prev},")
            lines.append(f"    Order{prev}Intermediates& inter{prev},")
    lines.append(f"    Order{order}Coefficients& out,")
    lines.append(f"    Order{order}Intermediates& inter")
    lines.append(f");")
    lines.append("")

    lines.append(f"void evaluate_order_{order}(")
    lines.append(f"    const Order{order}Coefficients& coeffs,")
    lines.append(f"    const double* dt_norm,")
    lines.append(f"    std::size_t M,")
    lines.append(f"    double* y_prop,")
    lines.append(f"    Order1EvaluationScratch& scratch")
    lines.append(f");")
    lines.append("")

    lines.append("} // namespace geqoe")
    lines.append("} // namespace astrodyn_core")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Source file generation
# ---------------------------------------------------------------------------

def generate_source(
    order: int,
    inline_helpers: str,
    compute_body: str,
    evaluate_body: str,
    prev_headers: List[str],
    eval_coeff_fields: Optional[List[str]] = None,
    map_col: Optional[int] = None,
) -> str:
    """Generate the complete taylor_order_N.cpp source file."""
    lines: List[str] = []

    lines.append(f'#include "taylor_order_{order}.hpp"')
    lines.append("")
    lines.append("#include <cmath>")
    lines.append("")
    lines.append('#include "math_utils.hpp"')
    lines.append("")
    lines.append("namespace astrodyn_core {")
    lines.append("namespace geqoe {")
    lines.append("")
    lines.append("namespace {")
    lines.append("")
    lines.append("constexpr std::size_t STATE_DIM = 6;")
    lines.append("")
    lines.append("inline std::size_t idx_yprop(std::size_t step, std::size_t comp) {")
    lines.append("    return step * STATE_DIM + comp;")
    lines.append("}")
    lines.append("")

    # Inline helpers
    lines.append(inline_helpers)

    lines.append("} // namespace")
    lines.append("")

    # compute_coefficients_N function
    lines.append(f"void compute_coefficients_{order}(")
    lines.append(f"    const double* y0,")
    lines.append(f"    const PropagationConstants& constants,")
    if order > 1:
        for prev in range(1, order):
            lines.append(f"    Order{prev}Coefficients& out{prev},")
            lines.append(f"    Order{prev}Intermediates& inter{prev},")
    lines.append(f"    Order{order}Coefficients& out,")
    lines.append(f"    Order{order}Intermediates& inter")
    lines.append(f") {{")

    if order > 1:
        # Call previous order's compute first, passing intermediates
        prev = order - 1
        prev_args = ["y0", "constants"]
        for p in range(1, prev):
            prev_args.append(f"out{p}")
            prev_args.append(f"inter{p}")
        prev_args.append(f"out{prev}")
        prev_args.append(f"inter{prev}")  # always pass inter for chaining
        lines.append(f"    compute_coefficients_{prev}({', '.join(prev_args)});")
        lines.append("")

    # Compute body
    lines.append(compute_body)

    # Fill coefficients struct from computed local variables
    if eval_coeff_fields:
        lines.append("")
        lines.append("    // Fill coefficients struct")
        for f in eval_coeff_fields:
            lines.append(f"    out.{f} = {f};")
    # Map column is already written by MapAssign in the compute body

    lines.append("}")
    lines.append("")

    # evaluate_order_N function
    lines.append(f"void evaluate_order_{order}(")
    lines.append(f"    const Order{order}Coefficients& coeffs,")
    lines.append(f"    const double* dt_norm,")
    lines.append(f"    std::size_t M,")
    lines.append(f"    double* y_prop,")
    lines.append(f"    Order1EvaluationScratch& scratch")
    lines.append(f") {{")
    lines.append(f"    for (std::size_t i = 0; i < M; ++i) {{")
    lines.append(f"        const double dt = dt_norm[i];")

    # Evaluate body
    lines.append(evaluate_body)

    lines.append(f"    }}")
    lines.append(f"}}")
    lines.append("")

    lines.append("} // namespace geqoe")
    lines.append("} // namespace astrodyn_core")
    lines.append("")

    return "\n".join(lines)
