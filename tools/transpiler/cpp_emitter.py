"""C++ code emitter for GEqOE Taylor order files.

Translates classified AST statements into C++ source code.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

from tools.transpiler.ast_parser import (
    DerivCallDoOne,
    DerivCallFull,
    DtCompute,
    EvalScratchWrite,
    EvalStateUpdate,
    EvalStmUpdate,
    Ignored,
    MapAssign,
    ParsedFunction,
    ScalarAssignment,
    ScratchRead,
    ScratchWrite,
    Statement,
    VectorConstruct,
    VectorElementExtract,
    VectorExtend,
)
from tools.transpiler.data_flow import (
    CrossOrderDeps,
    DerivCallInfo,
    VectorTracker,
)


# ---------------------------------------------------------------------------
# AST expression → C++ string
# ---------------------------------------------------------------------------

# Python operator → C++ operator
_BINOP_MAP = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}

_UNARYOP_MAP = {
    ast.UAdd: "+",
    ast.USub: "-",
}

# Python operator precedence (lower = binds less tightly)
_PREC = {
    ast.Add: 1,
    ast.Sub: 1,
    ast.Mult: 2,
    ast.Div: 2,
    ast.Pow: 3,
}


def _needs_parens(parent_op, child_op, is_right: bool) -> bool:
    """Determine if child expression needs parentheses inside parent."""
    p_prec = _PREC.get(type(parent_op), 0)
    c_prec = _PREC.get(type(child_op), 0)
    if c_prec < p_prec:
        return True
    if c_prec == p_prec and is_right:
        # For non-commutative ops (Sub, Div), right operand needs parens
        if isinstance(parent_op, (ast.Sub, ast.Div)):
            return True
    return False


def emit_expr(node: ast.expr, parent_op=None, is_right: bool = False) -> str:
    """Recursively convert a Python AST expression to C++ string."""

    # --- Name ---
    if isinstance(node, ast.Name):
        name = node.id
        # GAMMA_ is fine as-is in C++ (valid identifier)
        return name

    # --- Constant ---
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, (int, float)):
            # Convert integers to float literals to avoid integer division
            if isinstance(val, int):
                if val == 0:
                    return "0.0"
                elif val == 1:
                    return "1.0"
                elif val == 2:
                    return "2.0"
                elif val == 3:
                    return "3.0"
                elif val == -1:
                    return "-1.0"
                else:
                    return f"{float(val)}"
            else:
                # Float — format nicely
                if val == int(val) and abs(val) < 1e15:
                    return f"{int(val)}.0"
                return repr(val)
        if isinstance(val, str):
            return f'"{val}"'
        return str(val)

    # --- UnaryOp ---
    if isinstance(node, ast.UnaryOp):
        op_str = _UNARYOP_MAP.get(type(node.op))
        if op_str is not None:
            operand = emit_expr(node.operand)
            if op_str == "-":
                # Avoid double-negative issues
                if operand.startswith("-"):
                    return f"-({operand})"
                # Wrap BinOp operands in parens: -(a + b) must NOT become -a + b
                if isinstance(node.operand, ast.BinOp):
                    return f"-({operand})"
            return f"{op_str}{operand}"

    # --- BinOp ---
    if isinstance(node, ast.BinOp):
        # Special case: ** (power)
        if isinstance(node.op, ast.Pow):
            return _emit_power(node)

        op_str = _BINOP_MAP.get(type(node.op))
        if op_str is not None:
            left = emit_expr(node.left, node.op, False)
            right = emit_expr(node.right, node.op, True)

            # Add parens if needed for precedence
            if (isinstance(node.left, ast.BinOp)
                    and _needs_parens(node.op, node.left.op, False)):
                left = f"({left})"
            if (isinstance(node.right, ast.BinOp)
                    and _needs_parens(node.op, node.right.op, True)):
                right = f"({right})"
            # Also wrap unary negation on right side of subtraction
            if (isinstance(node.right, ast.UnaryOp)
                    and isinstance(node.right.op, ast.USub)
                    and isinstance(node.op, ast.Sub)):
                right = f"({right})"

            return f"{left} {op_str} {right}"

    # --- Call ---
    if isinstance(node, ast.Call):
        return _emit_call(node)

    # --- Subscript: s["key"] or vec[i] ---
    if isinstance(node, ast.Subscript):
        return _emit_subscript(node)

    # --- Attribute: ctx.something or np.pi ---
    if isinstance(node, ast.Attribute):
        return _emit_attribute(node)

    # --- List: [a, b, c] ---
    if isinstance(node, ast.List):
        elts = ", ".join(emit_expr(e) for e in node.elts)
        return f"{{{elts}}}"

    # Fallback
    return f"/* UNHANDLED: {ast.dump(node)} */"


def _emit_power(node: ast.BinOp) -> str:
    """Handle ``x ** n`` expressions."""
    base = emit_expr(node.left)
    exp = node.right

    # x**2 → (x * x)
    if isinstance(exp, ast.Constant):
        val = exp.value
        if val == 2:
            return f"({base} * {base})"
        if val == 3:
            return f"({base} * {base} * {base})"
        if val == 4:
            return f"({base} * {base} * {base} * {base})"
        if val == 0.5:
            return f"std::sqrt({base})"
        if val == -1:
            return f"(1.0 / {base})"
        if val == -2:
            return f"(1.0 / ({base} * {base}))"

    # x**(1/3) → std::cbrt(x)
    if isinstance(exp, ast.BinOp) and isinstance(exp.op, ast.Div):
        if (isinstance(exp.left, ast.Constant) and isinstance(exp.right, ast.Constant)):
            num, den = exp.left.value, exp.right.value
            if num == 1 and den == 3:
                return f"std::cbrt({base})"
            if num == 2 and den == 3:
                return f"std::cbrt({base} * {base})"
            if num == 1 and den == 2:
                return f"std::sqrt({base})"
            # General rational power
            return f"std::pow({base}, {float(num)}/{float(den)})"

    # Negative rational: -(N/D) → std::pow(x, -N.0/D.0)
    if isinstance(exp, ast.UnaryOp) and isinstance(exp.op, ast.USub):
        inner = exp.operand
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div):
            if (isinstance(inner.left, ast.Constant)
                    and isinstance(inner.right, ast.Constant)):
                num, den = inner.left.value, inner.right.value
                return f"std::pow({base}, -{float(num)} / {float(den)})"

    # General power
    exp_str = emit_expr(exp)
    return f"std::pow({base}, {exp_str})"


def _emit_call(node: ast.Call) -> str:
    """Handle function calls."""
    # np.sin, np.cos, np.sqrt, np.abs
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        obj = node.func.value.id
        method = node.func.attr
        if obj == "np":
            np_map = {
                "sin": "std::sin",
                "cos": "std::cos",
                "sqrt": "std::sqrt",
                "abs": "std::abs",
                "arctan2": "std::atan2",
                "mod": "std::fmod",
            }
            if method in np_map:
                args = ", ".join(emit_expr(a) for a in node.args)
                cpp_func = np_map[method]
                # np.mod(x, 2*np.pi) → std::fmod(x, 2.0 * M_PI)
                return f"{cpp_func}({args})"
            if method == "pi":
                return "M_PI"
            if method == "array":
                # This shouldn't be reached in normal emit flow
                return f"/* np.array call */"

    # Direct function calls (derivative functions handled separately)
    if isinstance(node.func, ast.Name):
        fname = node.func.id
        args = ", ".join(emit_expr(a) for a in node.args)
        return f"{fname}({args})"

    # Fallback
    return f"/* CALL: {ast.dump(node)} */"


def _emit_subscript(node: ast.Subscript) -> str:
    """Handle subscript access."""
    # s["key"] — inline scratch read in evaluate expressions
    if isinstance(node.value, ast.Name) and node.value.id == "s":
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return f"coeffs.{sl.value}"
    # vec[i]
    if isinstance(node.value, ast.Name):
        idx = emit_expr(node.slice)
        return f"{node.value.id}[{idx}]"
    return f"/* SUBSCRIPT: {ast.dump(node)} */"


def _emit_attribute(node: ast.Attribute) -> str:
    """Handle attribute access."""
    if isinstance(node.value, ast.Name):
        if node.value.id == "np" and node.attr == "pi":
            return "M_PI"
    # General: obj.attr
    obj = emit_expr(node.value)
    return f"{obj}.{node.attr}"


# ---------------------------------------------------------------------------
# Inline wrapper helpers for derivative calls
# ---------------------------------------------------------------------------

def generate_inline_helpers(calls: List[DerivCallInfo]) -> str:
    """Generate inline wrapper functions for derivative calls.

    These wrap the math_utils compute_* functions with stack-allocated arrays.
    """
    lines: List[str] = []
    seen: set = set()

    for c in sorted(calls, key=lambda x: (x.func_name, x.vector_length, x.do_one, x.has_param)):
        n = c.vector_length
        if c.do_one:
            key = (c.func_name, n, True, c.has_param)
            if key in seen:
                continue
            seen.add(key)
            if "product" in c.func_name:
                lines.append(_gen_product_helper(c, n))
            else:
                lines.append(_gen_inverse_helper(c, n))
        else:
            key = (c.func_name, n, False, c.has_param)
            if key in seen:
                continue
            seen.add(key)
            if "product" in c.func_name:
                lines.append(_gen_product_helper_full(c, n))
            else:
                lines.append(_gen_inverse_helper_full(c, n))

    return "\n".join(lines)


def _helper_name(func_name: str, n: int, do_one: bool = True) -> str:
    """Generate the inline helper function name.

    func_name already includes _wrt_param if applicable, e.g.
    "derivatives_of_inverse_wrt_param".
    """
    if do_one:
        return f"{func_name}_{n}"
    return f"{func_name}_full_{n}"


def _gen_inverse_helper(c: DerivCallInfo, n: int) -> str:
    """Generate inline helper for derivatives_of_inverse or _wrt_param."""
    if c.has_param:
        # derivatives_of_inverse_wrt_param_N(a0, a1, ..., ad0, ad1, ...)
        a_params = ", ".join(f"double a{i}" for i in range(n))
        d_params = ", ".join(f"double ad{i}" for i in range(n))
        a_init = ", ".join(f"a{i}" for i in range(n))
        d_init = ", ".join(f"ad{i}" for i in range(n))
        fname = f"derivatives_of_inverse_wrt_param_{n}"
        math_func = "astrodyn_core::math::compute_derivatives_of_inverse_wrt_param"
        return f"""inline double {fname}({a_params}, {d_params}) {{
    double out = 0.0;
    const double in_a[{n}] = {{{a_init}}};
    const double in_d[{n}] = {{{d_init}}};
    {math_func}(in_a, in_d, &out, {n}, true);
    return out;
}}
"""
    else:
        # derivatives_of_inverse_N(a0, a1, ...)
        a_params = ", ".join(f"double a{i}" for i in range(n))
        a_init = ", ".join(f"a{i}" for i in range(n))
        fname = f"derivatives_of_inverse_{n}"
        math_func = "astrodyn_core::math::compute_derivatives_of_inverse"
        return f"""inline double {fname}({a_params}) {{
    double out = 0.0;
    const double in_a[{n}] = {{{a_init}}};
    {math_func}(in_a, &out, {n}, true);
    return out;
}}
"""


def _gen_inverse_helper_full(c: DerivCallInfo, n: int) -> str:
    """Generate full-return (do_one=False) inverse helper."""
    a_params = ", ".join(f"double a{i}" for i in range(n))
    a_init = ", ".join(f"a{i}" for i in range(n))
    fname = f"derivatives_of_inverse_full_{n}"
    math_func = "astrodyn_core::math::compute_derivatives_of_inverse"
    out_decl = ", ".join(f"out[{i}]" for i in range(n))
    return f"""inline void {fname}({a_params}, double* out) {{
    const double in_a[{n}] = {{{a_init}}};
    {math_func}(in_a, out, {n}, false);
}}
"""


def _gen_product_helper(c: DerivCallInfo, n: int) -> str:
    """Generate inline helper for derivatives_of_product or _wrt_param."""
    if c.has_param:
        # Product wrt param: input is interleaved [a0, a1, ...], [ad0, ad1, ...]
        # For product, the math function takes m = n (vector length)
        a_params = ", ".join(f"double a{i}" for i in range(n))
        d_params = ", ".join(f"double ad{i}" for i in range(n))
        a_init = ", ".join(f"a{i}" for i in range(n))
        d_init = ", ".join(f"ad{i}" for i in range(n))
        fname = f"derivatives_of_product_wrt_param_{n}"
        math_func = "astrodyn_core::math::compute_derivatives_of_product_wrt_param"
        return f"""inline double {fname}({a_params}, {d_params}) {{
    double out = 0.0;
    const double in_a[{n}] = {{{a_init}}};
    const double in_d[{n}] = {{{d_init}}};
    {math_func}(in_a, in_d, &out, {n}, true);
    return out;
}}
"""
    else:
        a_params = ", ".join(f"double a{i}" for i in range(n))
        a_init = ", ".join(f"a{i}" for i in range(n))
        fname = f"derivatives_of_product_{n}"
        math_func = "astrodyn_core::math::compute_derivatives_of_product"
        return f"""inline double {fname}({a_params}) {{
    double out = 0.0;
    const double in_a[{n}] = {{{a_init}}};
    {math_func}(in_a, &out, {n}, true);
    return out;
}}
"""


def _gen_product_helper_full(c: DerivCallInfo, n: int) -> str:
    """Generate full-return product helper."""
    a_params = ", ".join(f"double a{i}" for i in range(n))
    a_init = ", ".join(f"a{i}" for i in range(n))
    out_len = n - 1  # product output is one shorter
    fname = f"derivatives_of_product_full_{n}"
    math_func = "astrodyn_core::math::compute_derivatives_of_product"
    return f"""inline void {fname}({a_params}, double* out) {{
    const double in_a[{n}] = {{{a_init}}};
    {math_func}(in_a, out, {n}, false);
}}
"""


# ---------------------------------------------------------------------------
# Statement → C++ line emitters
# ---------------------------------------------------------------------------

def emit_scratch_read(stmt: ScratchRead, inter_prefix: str) -> str:
    """Emit: ``const double var = inter.key;``"""
    return f"    const double {stmt.target} = {inter_prefix}.{stmt.key};"


def emit_scalar_assignment(stmt: ScalarAssignment, mutable_vars: Set[str]) -> str:
    """Emit: ``const double var = expr;`` or ``var = expr;`` for mutable vars."""
    expr = emit_expr(stmt.value_node)
    if stmt.target in mutable_vars:
        return f"    {stmt.target} = {expr};"
    return f"    const double {stmt.target} = {expr};"


def _get_deriv_elements(stmt, vt: VectorTracker) -> List[str]:
    """Get vector elements for a derivative call, handling inline np.array.

    Returns C++ expression strings for each element.
    """
    if stmt.inline_elements is not None:
        return stmt.inline_elements
    if stmt.vector_arg is not None:
        return vt.get_cpp_elements(stmt.vector_arg)
    raise ValueError(f"Derivative call has neither vector_arg nor inline_elements: {stmt}")


def emit_deriv_call_do_one(stmt: DerivCallDoOne, vt: VectorTracker) -> str:
    """Emit inline wrapper call for do_one=True derivative."""
    elements = _get_deriv_elements(stmt, vt)
    n = len(elements)

    # Function name: func_name already includes _wrt_param if applicable
    fname = f"{stmt.func_name}_{n}"

    if stmt.param_vector_arg:
        param_elements = vt.get_cpp_elements(stmt.param_vector_arg)
        all_args = ", ".join(elements + param_elements)
    else:
        all_args = ", ".join(elements)

    return f"    const double {stmt.target} = {fname}({all_args});"


def emit_deriv_call_full(stmt: DerivCallFull, vt: VectorTracker,
                          extract_stmts: List[VectorElementExtract]) -> List[str]:
    """Emit full derivative call (returns array) + element extractions."""
    elements = _get_deriv_elements(stmt, vt)
    n = len(elements)

    lines: List[str] = []

    if "product" in stmt.func_name:
        out_len = n - 1
        fname = f"{stmt.func_name}_full_{n}"
    else:
        out_len = n
        fname = f"{stmt.func_name}_full_{n}"

    all_args = ", ".join(elements)
    result_name = f"_buf_{stmt.target}"
    lines.append(f"    double {result_name}[{out_len}];")
    lines.append(f"    {fname}({all_args}, {result_name});")

    # Emit element extractions
    for ext in extract_stmts:
        lines.append(f"    const double {ext.target} = {result_name}[{ext.index}];")

    return lines


def emit_scratch_write(stmt: ScratchWrite, out_prefix: str) -> str:
    """Emit: ``out.key = value;``"""
    val = emit_expr(stmt.value_node)
    return f"    {out_prefix}.{stmt.key} = {val};"


def emit_map_assign(stmt: MapAssign, out_var: str) -> List[str]:
    """Emit map_components column assignment."""
    lines: List[str] = []
    col = stmt.column
    for i, elt in enumerate(stmt.elements):
        val = emit_expr(elt)
        lines.append(f"    {out_var}.map_components_col{col}[{i}] = {val};")
    return lines


# ---------------------------------------------------------------------------
# Top-level function generators
# ---------------------------------------------------------------------------

def generate_compute_coefficients(
    order: int,
    func: ParsedFunction,
    vt: VectorTracker,
    cross_deps: Optional[CrossOrderDeps],
    prev_inter_name: Optional[str],
    vec_only_locals: Optional[List[str]] = None,
) -> str:
    """Generate the full ``compute_coefficients_N`` C++ function body."""
    lines: List[str] = []

    # Emit synthetic reads for vector-only locals from the previous order.
    # These are variables stored in vectors but never written to scratch,
    # so they don't get normal scratch-read lines but are needed by
    # derivative calls that reference vector elements.
    if vec_only_locals and prev_inter_name:
        for vname in vec_only_locals:
            lines.append(f"    const double {vname} = {prev_inter_name}.{vname};")

    # Track variables that get reassigned (like fic in Order 2)
    assigned_vars: Set[str] = set()
    mutable_vars: Set[str] = set()

    # Pre-scan for reassignments
    for stmt in func.statements:
        target = None
        if isinstance(stmt, ScalarAssignment):
            target = stmt.target
        elif isinstance(stmt, DerivCallDoOne):
            target = stmt.target
        elif isinstance(stmt, VectorElementExtract):
            target = stmt.target

        if target is not None:
            if target in assigned_vars:
                mutable_vars.add(target)
            assigned_vars.add(target)

    # Also check scratch reads that get overwritten later
    scratch_read_vars = set()
    for stmt in func.statements:
        if isinstance(stmt, ScratchRead):
            scratch_read_vars.add(stmt.target)

    for var in scratch_read_vars & assigned_vars:
        mutable_vars.add(var)

    # Track which variables have been declared (to avoid redeclaration)
    declared_vars: Set[str] = set()

    # Build incremental vector tracker
    live_vt = VectorTracker()
    # Seed from pre-built tracker (for cross-order vectors from scratch)
    if vt is not None:
        live_vt.vectors = {k: list(v) for k, v in vt.vectors.items()}
        live_vt.ast_nodes = dict(vt.ast_nodes)

    # Generate statements
    i = 0
    stmts = func.statements
    while i < len(stmts):
        stmt = stmts[i]

        # Update live tracker for vector operations
        if isinstance(stmt, VectorConstruct):
            live_vt.construct(stmt.target, stmt.elements,
                            getattr(stmt, 'element_nodes', None))
        elif isinstance(stmt, VectorExtend):
            live_vt.extend(stmt.target, stmt.element,
                          getattr(stmt, 'element_node', None))
        elif isinstance(stmt, ScratchRead) and stmt.key.endswith("_vector"):
            if stmt.key in live_vt.vectors:
                live_vt.vectors[stmt.target] = list(live_vt.vectors[stmt.key])
                for (vname, idx), node in list(live_vt.ast_nodes.items()):
                    if vname == stmt.key:
                        live_vt.ast_nodes[(stmt.target, idx)] = node

        if isinstance(stmt, Ignored):
            i += 1
            continue

        if isinstance(stmt, ScratchRead):
            # Skip vector scratch reads — vectors are logical, not physical
            if stmt.key.endswith("_vector"):
                # Vector aliasing handled by live_vt tracker (already done above)
                i += 1
                continue
            inter_prefix = prev_inter_name or "inter1"
            if stmt.target in mutable_vars:
                lines.append(f"    double {stmt.target} = {inter_prefix}.{stmt.key};")
            else:
                lines.append(emit_scratch_read(stmt, inter_prefix))
            declared_vars.add(stmt.target)

        elif isinstance(stmt, ScalarAssignment):
            lines.append(emit_scalar_assignment(stmt, mutable_vars))
            declared_vars.add(stmt.target)

        elif isinstance(stmt, DerivCallDoOne):
            if stmt.target in mutable_vars and stmt.target in declared_vars:
                expr = _build_deriv_do_one_expr(stmt, live_vt)
                lines.append(f"    {stmt.target} = {expr};")
            else:
                lines.append(emit_deriv_call_do_one(stmt, live_vt))
            declared_vars.add(stmt.target)

        elif isinstance(stmt, DerivCallFull):
            # Collect subsequent VectorElementExtract statements
            extracts: List[VectorElementExtract] = []
            j = i + 1
            while j < len(stmts) and isinstance(stmts[j], VectorElementExtract):
                ext = stmts[j]
                if ext.vector_name == stmt.target:
                    extracts.append(ext)
                else:
                    break
                j += 1

            full_lines = emit_deriv_call_full(stmt, live_vt, extracts)
            # Handle mutable/redeclared extracted vars
            for line_idx, ext in enumerate(extracts):
                if ext.target in declared_vars:
                    # Already declared — just reassign (no type)
                    old = f"    const double {ext.target} ="
                    new = f"    {ext.target} ="
                    full_lines[2 + line_idx] = full_lines[2 + line_idx].replace(old, new)
                elif ext.target in mutable_vars:
                    # First declaration but will be reassigned — use non-const
                    old = f"    const double {ext.target} ="
                    new = f"    double {ext.target} ="
                    full_lines[2 + line_idx] = full_lines[2 + line_idx].replace(old, new)
                declared_vars.add(ext.target)

            lines.extend(full_lines)
            i = j  # skip the extracted elements we already handled
            continue

        elif isinstance(stmt, VectorElementExtract):
            # Standalone element extract (not following a DerivCallFull)
            # This happens when extracting from a result named differently
            pass  # handled inline with DerivCallFull above

        elif isinstance(stmt, VectorConstruct):
            # No C++ output — just tracking
            pass

        elif isinstance(stmt, VectorExtend):
            # No C++ output — just tracking
            pass

        elif isinstance(stmt, ScratchWrite):
            # Skip vector scratch writes — vectors are logical, not physical
            if not stmt.key.endswith("_vector"):
                lines.append(emit_scratch_write(stmt, "inter"))

        elif isinstance(stmt, MapAssign):
            lines.extend(emit_map_assign(stmt, "out"))

        i += 1

    return "\n".join(lines)


def _build_deriv_do_one_expr(stmt: DerivCallDoOne, vt: VectorTracker) -> str:
    """Build the expression for a derivative do_one call (without const double prefix)."""
    elements = _get_deriv_elements(stmt, vt)
    n = len(elements)
    fname = f"{stmt.func_name}_{n}"
    if stmt.param_vector_arg:
        param_elements = vt.get_cpp_elements(stmt.param_vector_arg)
        all_args = ", ".join(elements + param_elements)
    else:
        all_args = ", ".join(elements)
    return f"{fname}({all_args})"


def generate_evaluate_order(
    order: int,
    func: ParsedFunction,
) -> str:
    """Generate the full ``evaluate_order_N`` C++ function body.

    The generated code goes inside a ``for (i = 0; i < M; ++i)`` loop.
    """
    lines: List[str] = []

    # Pre-scan: collect keys that are STM accumulators (will be modified via +=)
    stm_targets: Set[str] = set()
    for stmt in func.statements:
        if isinstance(stmt, EvalStmUpdate):
            stm_targets.add(stmt.target)
        if isinstance(stmt, EvalScratchWrite):
            stm_targets.add(stmt.key)

    for stmt in func.statements:
        if isinstance(stmt, Ignored):
            continue

        if isinstance(stmt, ScratchRead):
            # Skip STM accumulator reads — they live in scratch, not coeffs,
            # and are updated via += directly
            if stmt.key in stm_targets:
                continue
            lines.append(f"        const double {stmt.target} = coeffs.{stmt.key};")

        elif isinstance(stmt, DtCompute):
            expr = emit_expr(stmt.value_node)
            # In the loop, dt_norm[i] is the variable 'dt'
            expr = expr.replace("dt_norm", "dt")
            lines.append(f"        const double {stmt.target} = {expr};")

        elif isinstance(stmt, EvalStateUpdate):
            expr = emit_expr(stmt.value_node)
            expr = expr.replace("dt_norm", "dt")
            if stmt.op == "+=":
                lines.append(f"        y_prop[idx_yprop(i, {stmt.column})] += {expr};")
            else:
                lines.append(f"        y_prop[idx_yprop(i, {stmt.column})] = {expr};")

        elif isinstance(stmt, EvalStmUpdate):
            delta_expr = _extract_stm_delta(stmt)
            lines.append(f"        scratch.{stmt.target}[i] += {delta_expr};")

        elif isinstance(stmt, EvalScratchWrite):
            # In C++, scratch is updated in-place via +=. Skip.
            pass

        elif isinstance(stmt, ScalarAssignment):
            # In evaluate, remaining scalar assignments are typically
            # identity constants like nu_nu = 1, or expressions involving
            # inline scratch reads like s["q1p_nu"] * dt_norm
            expr = emit_expr(stmt.value_node)
            # In the loop, dt_norm[i] is the variable 'dt'
            expr = expr.replace("dt_norm", "dt")
            lines.append(f"        scratch.{stmt.target}[i] = {expr};")

    return "\n".join(lines)


def _extract_stm_delta(stmt: EvalStmUpdate) -> str:
    """Extract the delta expression from ``var = var + delta``.

    The Python code writes: ``Lr_nu = Lr_nu + Lrp2_nu*dt2/2``
    We need to emit just ``Lrp2_nu*dt2/2`` as the delta.
    """
    node = stmt.value_node
    target = stmt.target

    # Pattern: var + delta  (BinOp with Add)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        if isinstance(node.left, ast.Name) and node.left.id == target:
            return emit_expr(node.right)
        if isinstance(node.right, ast.Name) and node.right.id == target:
            return emit_expr(node.left)

    # Fallback: emit the full expression and let it be an assignment
    return f"/* DELTA EXTRACT FAILED */ {emit_expr(node)}"
