"""AST-based statement classifier for GEqOE Taylor order Python files.

Parses ``compute_coefficients_N`` and ``evaluate_order_N`` function bodies
and classifies every statement into one of the transpiler's statement types.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Statement data classes
# ---------------------------------------------------------------------------

@dataclass
class ScratchRead:
    """``var = s["key"]``  — read from scratch dict."""
    target: str          # LHS variable name
    key: str             # dict key string


@dataclass
class ScalarAssignment:
    """``var = <arithmetic expr>``  — pure arithmetic (no calls, no dict)."""
    target: str
    value_node: ast.expr  # keep the AST node for the emitter


@dataclass
class DerivCallDoOne:
    """``var = derivatives_of_*(vec, True)`` — returns scalar."""
    target: str
    func_name: str        # e.g. "derivatives_of_inverse"
    vector_arg: Optional[str]       # name of the vector variable (None if inline)
    param_vector_arg: Optional[str]  # for _wrt_param variants
    value_node: ast.expr  # the full Call node
    inline_elements: Optional[List[str]] = None  # elements if vector was inline np.array


@dataclass
class DerivCallFull:
    """``vec = derivatives_of_*(vec)`` — returns array, unpacked later.

    Also covers the pattern where individual elements are extracted via
    ``var = result[0]; var2 = result[1]`` on subsequent lines — those are
    classified as VectorElementExtract.
    """
    target: str           # LHS name (the result vector)
    func_name: str
    vector_arg: Optional[str]
    param_vector_arg: Optional[str]
    value_node: ast.expr
    inline_elements: Optional[List[str]] = None


@dataclass
class VectorElementExtract:
    """``var = vec[idx]`` — extract element from a DerivCallFull result."""
    target: str
    vector_name: str
    index: int


@dataclass
class VectorConstruct:
    """``vec = np.array([a, b, ...])``"""
    target: str
    elements: List[str]   # element expression names/strings
    element_nodes: List[ast.expr]


@dataclass
class VectorExtend:
    """``vec = np.append(vec, elem)``"""
    target: str
    vector_name: str
    element: str
    element_node: ast.expr


@dataclass
class ScratchWrite:
    """``s["key"] = value``"""
    key: str
    value: str            # the variable name being stored
    value_node: ast.expr


@dataclass
class MapAssign:
    """``ctx.map_components[:, col] = [v0, v1, ..., v5]``"""
    column: int
    elements: List[ast.expr]  # 6-element list


@dataclass
class EvalStateUpdate:
    """``ctx.y_prop[:, col] += expr``  (AugAssign in evaluate function)."""
    column: int
    value_node: ast.expr
    op: str  # '+=' or '='


@dataclass
class EvalStmUpdate:
    """``var = var + expr``  — STM accumulator update in evaluate."""
    target: str
    value_node: ast.expr


@dataclass
class EvalScratchWrite:
    """``s["key"] = var`` in evaluate function (write STM accumulators)."""
    key: str
    value: str
    value_node: ast.expr


@dataclass
class ConstantExtract:
    """Extracts like ``mu_norm = ctx.constants.mu_norm`` or ``T = ctx.constants.time_scale``."""
    target: str
    attr_chain: str  # e.g. "ctx.constants.mu_norm"


@dataclass
class InitialStateExtract:
    """``var = ctx.initial_state.field`` or ``var = st.field``."""
    target: str
    field: str


@dataclass
class DtCompute:
    """``dt2 = dt_norm**2`` etc — time polynomial computation in evaluate."""
    target: str
    value_node: ast.expr


@dataclass
class Ignored:
    """Passthrough for lines we intentionally skip (imports, docstrings, etc.)."""
    reason: str
    line: Optional[int] = None


# Union type for all statement types
Statement = (
    ScratchRead | ScalarAssignment | DerivCallDoOne | DerivCallFull |
    VectorElementExtract | VectorConstruct | VectorExtend |
    ScratchWrite | MapAssign | EvalStateUpdate | EvalStmUpdate |
    EvalScratchWrite | ConstantExtract | InitialStateExtract |
    DtCompute | Ignored
)


# ---------------------------------------------------------------------------
# Derivative function names we recognize
# ---------------------------------------------------------------------------
DERIV_FUNCS = {
    "derivatives_of_inverse",
    "derivatives_of_inverse_wrt_param",
    "derivatives_of_product",
    "derivatives_of_product_wrt_param",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_scratch_subscript(node: ast.expr) -> Optional[str]:
    """If ``node`` is ``s["key"]``, return key. Else None."""
    if (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == "s"):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return sl.value
    return None


def _is_np_array(node: ast.expr) -> Optional[List[ast.expr]]:
    """If ``node`` is ``np.array([...])``, return the element list."""
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr == "array"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.List)):
        return node.args[0].elts
    return None


def _is_np_append(node: ast.expr) -> Optional[Tuple[str, ast.expr]]:
    """If ``node`` is ``np.append(vec, elem)``, return (vec_name, elem_node)."""
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr == "append"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)):
        return node.args[0].id, node.args[1]
    return None


def _is_deriv_call(node: ast.expr) -> Optional[Tuple[str, Optional[str], Optional[str], bool, Optional[List[str]]]]:
    """If ``node`` is a derivative function call, return
    (func_name, vec_arg, param_vec_arg, do_one, inline_elements).

    ``inline_elements`` is set when the first argument is an inline
    ``np.array([...])`` rather than a named vector variable.
    """
    if not isinstance(node, ast.Call):
        return None
    # Check if it's a known derivative function
    func_name = None
    if isinstance(node.func, ast.Name) and node.func.id in DERIV_FUNCS:
        func_name = node.func.id
    if func_name is None:
        return None

    args = node.args
    if len(args) < 1:
        return None

    vec_arg = None
    inline_elements = None
    if isinstance(args[0], ast.Name):
        vec_arg = args[0].id
    else:
        # Check for inline np.array([...])
        inline_elts = _is_np_array(args[0])
        if inline_elts is not None:
            inline_elements = []
            for e in inline_elts:
                if isinstance(e, ast.Name):
                    inline_elements.append(e.id)
                else:
                    inline_elements.append(ast.dump(e))

    # Determine if _wrt_param variant
    param_vec_arg = None
    do_one = False

    if "wrt_param" in func_name:
        # derivatives_of_*_wrt_param(vec, param_vec, True/False)
        if len(args) >= 2 and isinstance(args[1], ast.Name):
            param_vec_arg = args[1].id
        if len(args) >= 3 and isinstance(args[2], ast.Constant):
            do_one = bool(args[2].value)
    else:
        # derivatives_of_*(vec, True/False)
        if len(args) >= 2 and isinstance(args[1], ast.Constant):
            do_one = bool(args[1].value)

    return func_name, vec_arg, param_vec_arg, do_one, inline_elements


def _is_vector_subscript(node: ast.expr) -> Optional[Tuple[str, int]]:
    """If ``node`` is ``vec[N]`` (constant integer index), return (vec_name, N)."""
    if (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
            return node.value.id, sl.value
    return None


def _is_map_components_assign(stmt: ast.Assign) -> Optional[Tuple[int, List[ast.expr]]]:
    """Check if stmt is ``ctx.map_components[:, col] = [...]`` or ``map_components[:, col] = [...]``."""
    if len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Subscript):
        return None
    # target.value should be ctx.map_components or just map_components
    is_map = False
    if (isinstance(target.value, ast.Attribute)
            and target.value.attr == "map_components"):
        is_map = True
    elif isinstance(target.value, ast.Name) and target.value.id == "map_components":
        is_map = True
    if not is_map:
        return None
    # target.slice should be Tuple(Slice(), Constant(col))
    sl = target.slice
    if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
        if isinstance(sl.elts[1], ast.Constant):
            col = sl.elts[1].value
            if isinstance(stmt.value, ast.List):
                return col, stmt.value.elts
    return None


def _is_yprop_update(stmt) -> Optional[Tuple[int, ast.expr, str]]:
    """Check if stmt is ``ctx.y_prop[:, col] = expr`` or ``+= expr``."""
    if isinstance(stmt, ast.AugAssign):
        target = stmt.target
        op = "+="
    elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        op = "="
    else:
        return None

    if not isinstance(target, ast.Subscript):
        return None
    if not (isinstance(target.value, ast.Attribute)
            and target.value.attr == "y_prop"):
        return None
    sl = target.slice
    if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
        if isinstance(sl.elts[1], ast.Constant):
            col = sl.elts[1].value
            value = stmt.value
            return col, value, op
    return None


def _get_name(node: ast.expr) -> Optional[str]:
    """Get plain name from AST node."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _expr_contains_name(node: ast.expr, name: str) -> bool:
    """Check if an expression AST tree contains a Name node with the given id."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
    return False


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

@dataclass
class ParsedFunction:
    """Holds the classified statements for one function."""
    name: str
    statements: List[Statement] = field(default_factory=list)
    order: int = 0


def _classify_single_assign(target_name: str, value: ast.expr,
                             is_evaluate: bool) -> Statement:
    """Classify a single ``target = value`` assignment."""
    # 1. Scratch read: s["key"]
    key = _is_scratch_subscript(value)
    if key is not None:
        return ScratchRead(target=target_name, key=key)

    # 2. Derivative call
    dc = _is_deriv_call(value)
    if dc is not None:
        func_name, vec_arg, param_vec_arg, do_one, inline_elts = dc
        if do_one:
            return DerivCallDoOne(
                target=target_name, func_name=func_name,
                vector_arg=vec_arg, param_vector_arg=param_vec_arg,
                value_node=value, inline_elements=inline_elts)
        else:
            return DerivCallFull(
                target=target_name, func_name=func_name,
                vector_arg=vec_arg, param_vector_arg=param_vec_arg,
                value_node=value, inline_elements=inline_elts)

    # 3. np.array([...])
    elts = _is_np_array(value)
    if elts is not None:
        elem_names = []
        for e in elts:
            if isinstance(e, ast.Name):
                elem_names.append(e.id)
            else:
                elem_names.append(ast.dump(e))
        return VectorConstruct(target=target_name, elements=elem_names,
                               element_nodes=elts)

    # 4. np.append(vec, elem)
    app = _is_np_append(value)
    if app is not None:
        vec_name, elem_node = app
        elem_str = _get_name(elem_node) or ast.dump(elem_node)
        return VectorExtend(target=target_name, vector_name=vec_name,
                            element=elem_str, element_node=elem_node)

    # 5. Vector element extract: vec[N]
    ve = _is_vector_subscript(value)
    if ve is not None:
        return VectorElementExtract(target=target_name, vector_name=ve[0],
                                    index=ve[1])

    # 5b. Ignore np.zeros/np.empty/np.ones array creation (Python-only setup)
    if (isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "np"
            and value.func.attr in ("zeros", "empty", "ones")):
        return Ignored(reason="numpy array creation")

    # 6. Ignore setup lines like s = ctx.scratch, dt_norm = ctx.dt_norm
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        if value.value.id == "ctx":
            return Ignored(reason="ctx attribute setup")

    # 6a. In evaluate context, detect dt computation patterns
    if is_evaluate and target_name.startswith("dt"):
        return DtCompute(target=target_name, value_node=value)

    # 6b. In evaluate context, detect STM accumulator self-referential updates
    #     Pattern: var = var + expr  (target appears on RHS)
    if is_evaluate and _expr_contains_name(value, target_name):
        return EvalStmUpdate(target=target_name, value_node=value)

    # 7. Scalar arithmetic assignment
    return ScalarAssignment(target=target_name, value_node=value)


def _classify_stmt(stmt: ast.stmt, is_evaluate: bool,
                   deriv_call_results: set) -> List[Statement]:
    """Classify a single AST statement node. May return multiple statements
    for multi-target or tuple-unpack assignments."""
    results: List[Statement] = []

    # --- ast.Expr (standalone expression, e.g. function call or docstring) ---
    if isinstance(stmt, ast.Expr):
        # call to compute_coefficients_N-1 or evaluate_order_N-1
        if isinstance(stmt.value, ast.Call):
            return [Ignored(reason="chained function call")]
        if isinstance(stmt.value, ast.Constant):
            return [Ignored(reason="docstring")]
        return [Ignored(reason="standalone expression")]

    # --- ast.AugAssign (+=, etc.) ---
    if isinstance(stmt, ast.AugAssign):
        ypu = _is_yprop_update(stmt)
        if ypu is not None:
            col, val, op = ypu
            return [EvalStateUpdate(column=col, value_node=val, op=op)]
        # STM accumulator: var = var + expr  (but written as var += expr)
        # Actually in the Python code it's written as  var = var + expr
        # not var += expr.  But handle both.
        if isinstance(stmt.target, ast.Name):
            return [EvalStmUpdate(target=stmt.target.id, value_node=stmt.value)]
        return [Ignored(reason="unclassified augassign")]

    # --- ast.Assign ---
    if isinstance(stmt, ast.Assign):
        # Check map_components assign first
        mc = _is_map_components_assign(stmt)
        if mc is not None:
            return [MapAssign(column=mc[0], elements=mc[1])]

        # Check y_prop direct assign
        ypu = _is_yprop_update(stmt)
        if ypu is not None:
            col, val, op = ypu
            return [EvalStateUpdate(column=col, value_node=val, op=op)]

        # Multiple targets on one line: a = b = expr
        # (rare in this codebase, but handle)
        if len(stmt.targets) > 1:
            # Handle each target
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    s = _classify_single_assign(t.id, stmt.value, is_evaluate)
                    results.append(s)
            return results if results else [Ignored(reason="multi-target")]

        target = stmt.targets[0]

        # Single name target
        if isinstance(target, ast.Name):
            s = _classify_single_assign(target.id, stmt.value, is_evaluate)
            # Track DerivCallFull results for later element extraction
            if isinstance(s, DerivCallFull):
                deriv_call_results.add(target.id)
            return [s]

        # Tuple unpack: a, b = func(...) or a, b = expr1, expr2
        if isinstance(target, ast.Tuple):
            names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
            # Check if RHS is a derivative call
            dc = _is_deriv_call(stmt.value)
            if dc is not None:
                func_name, vec_arg, param_vec_arg, do_one, inline_elts = dc
                if not do_one and len(names) > 1:
                    # Full derivative call with tuple unpack
                    # e.g. fir, firp = derivatives_of_inverse(r_vector)
                    full = DerivCallFull(
                        target=f"_unpack_{'_'.join(names)}",
                        func_name=func_name, vector_arg=vec_arg,
                        param_vector_arg=param_vec_arg,
                        value_node=stmt.value,
                        inline_elements=inline_elts)
                    results.append(full)
                    for i, n in enumerate(names):
                        results.append(VectorElementExtract(
                            target=n, vector_name=full.target, index=i))
                    return results

            # RHS is a Tuple of expressions — split into individual assigns
            if isinstance(stmt.value, ast.Tuple):
                for name, val in zip(names, stmt.value.elts):
                    results.append(
                        _classify_single_assign(name, val, is_evaluate))
                return results

            return [Ignored(reason="unclassified tuple unpack")]

        # Subscript target: s["key"] = value
        key = _is_scratch_subscript(target)
        if key is not None:
            val_name = _get_name(stmt.value) or ast.dump(stmt.value)
            if is_evaluate:
                return [EvalScratchWrite(key=key, value=val_name,
                                         value_node=stmt.value)]
            return [ScratchWrite(key=key, value=val_name,
                                value_node=stmt.value)]

        # Attribute target: ctx.y_prop = y_prop, ctx.y_y0 = y_y0
        if isinstance(target, ast.Attribute):
            return [Ignored(reason="ctx attribute setup")]

        return [Ignored(reason="unclassified assign target")]

    # --- Imports ---
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return [Ignored(reason="import")]

    # --- Other ---
    return [Ignored(reason=f"unclassified {type(stmt).__name__}")]


# ---------------------------------------------------------------------------
# Top-level parse functions
# ---------------------------------------------------------------------------

def _find_function(tree: ast.Module, func_name: str) -> Optional[ast.FunctionDef]:
    """Find a top-level function definition by name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def parse_function(source: str, func_name: str,
                   is_evaluate: bool = False) -> ParsedFunction:
    """Parse a Python source string and classify all statements in the named function."""
    tree = ast.parse(source)
    func = _find_function(tree, func_name)
    if func is None:
        raise ValueError(f"Function {func_name!r} not found in source")

    result = ParsedFunction(name=func_name)
    deriv_call_results: set = set()

    for stmt in func.body:
        classified = _classify_stmt(stmt, is_evaluate, deriv_call_results)
        result.statements.extend(classified)

    return result


def parse_order_file(filepath: str | Path, order: int) -> Tuple[ParsedFunction, ParsedFunction]:
    """Parse a taylor_order_N.py file and return (compute_func, evaluate_func)."""
    source = Path(filepath).read_text()
    compute = parse_function(source, f"compute_coefficients_{order}", is_evaluate=False)
    compute.order = order
    evaluate = parse_function(source, f"evaluate_order_{order}", is_evaluate=True)
    evaluate.order = order
    return compute, evaluate
